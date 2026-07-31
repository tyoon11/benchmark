"""
NumPy-backed ECG Dataset (EchoNext)
===================================
PhysioNet EchoNext 1.1.0 ships ``.npy`` waveform stacks plus a metadata CSV
rather than per-record files, so it bypasses the H5 store. This loader mirrors
:class:`src.dataset.H5ECGDataset` exactly — same encoder contract
(``input_size`` seconds x ``fs_model`` Hz), same train/val-test chunking, same
band-limited resampling, same lead-order handling — so the trainer and the
prediction-aggregation path are shared.

EchoNext format:
  - ``EchoNext_<split>_waveforms.npy``: ``(N, 1, 2500, 12)`` float64, 250 Hz,
    12-lead, 10 s (already median-filtered, percentile-clipped and
    dataset-wide z-scored by the publishers).
  - ``echonext_metadata_100k.csv``: ``split`` column plus binary echo flags.

The split assignment comes from the metadata ``split`` column, matching the
original (``main_lite_ecg.setup_dataset`` maps val->strat_fold 8, test->9).
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .leads import (STANDARD_12, build_lead_permutation, describe_permutation,
                    resolve_target_order)
from .signal_utils import fit_length, resample_signal

logger = logging.getLogger(__name__)


class EchoNextDataset(Dataset):
    """EchoNext ``.npy`` dataset with the paper-faithful windowing contract.

    Args:
        waveform_npy:  the split's ``.npy`` (``(N, 1, T, C)`` or ``(N, C, T)``).
        metadata_csv:  metadata CSV covering all splits.
        split:         ``'train'`` | ``'val'`` | ``'test'``.
        md_split:      value to match in ``split_col`` (defaults to ``split``).
        label_cols:    binary flag columns.
        source_fs:     native rate of the waveform stack (250 Hz).
        input_size:    window length in seconds (encoder contract).
        fs_model:      rate the encoder expects (encoder contract).
        source_leads:  lead names of the stored array (defaults to standard order).
        lead_order:    order the encoder wants; see :mod:`src.leads`.
    """

    def __init__(
        self,
        waveform_npy: str,
        metadata_csv: str,
        split: str,
        *,
        md_split: str = None,
        split_col: str = "split",
        label_cols: list = None,
        source_fs: int = 250,
        input_size: float = 2.5,
        fs_model: float = 500,
        chunkify_train: bool = False,
        chunk_length_train: float = 1.0,
        stride_fraction_train: float = 1.0,
        stride_fraction_valtest: float = 1.0,
        source_leads=None,
        lead_order="standard",
        normalize: bool = False,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        n_leads: int = 12,
        layout: str = "NHWC",
    ):
        if not label_cols:
            raise ValueError("label_cols must list at least one column.")

        self.split = split
        self.is_train = (split == "train")
        self.source_fs = float(source_fs)
        self.input_size = float(input_size)
        self.fs_model = float(fs_model)
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.n_leads = n_leads
        self.layout = layout
        self.label_cols = list(label_cols)
        self.num_classes = len(self.label_cols)
        self.has_labels = True

        self.output_size = int(round(self.input_size * self.source_fs))
        self.model_seq_len = int(round(self.input_size * self.fs_model))

        # memory-mapped: the stacks are 5-16 GB per split
        self.waveforms = np.load(waveform_npy, mmap_mode="r")
        self._wf_path = os.path.abspath(waveform_npy)

        df = pd.read_csv(metadata_csv, low_memory=False)
        if split_col not in df.columns:
            raise ValueError(f"metadata_csv has no '{split_col}' column.")
        df = df[df[split_col] == (md_split or split)].reset_index(drop=True)

        n_npy = self.waveforms.shape[0]
        if len(df) != n_npy:
            raise ValueError(
                f"split='{split}' row mismatch: csv={len(df)}, npy={n_npy}. "
                f"The .npy for this split is probably incomplete.")

        missing = [c for c in self.label_cols if c not in df.columns]
        if missing:
            raise ValueError(f"label columns absent from metadata: {missing}")

        self.df = df
        self.labels = np.nan_to_num(df[self.label_cols].to_numpy(dtype=np.float32), nan=0.0)

        # ── lead permutation ──────────────────────────────────────────────
        self.source_leads = list(source_leads) if source_leads else list(STANDARD_12)
        self.target_leads = resolve_target_order(lead_order)
        self.lead_perm = build_lead_permutation(self.source_leads, self.target_leads)
        logger.info("  [%s] %s", split,
                    describe_permutation(self.source_leads, self.target_leads, self.lead_perm))

        # ── window index (identical regime to H5ECGDataset) ───────────────
        record_len = int(self.waveforms.shape[2] if self.layout == "NHWC"
                         else self.waveforms.shape[-1])
        self._record_len = record_len

        row_idx, starts = [], []
        if record_len < self.output_size:
            raise RuntimeError(
                f"[{split}] EchoNext records are {record_len} samples but the encoder "
                f"asks for {self.output_size} ({self.input_size}s @ {self.source_fs}Hz).")

        for i in range(len(df)):
            if self.is_train and not chunkify_train:
                row_idx.append(i)
                starts.append(-1)  # sentinel: random crop at __getitem__ time
                continue
            if self.is_train:
                chunk = max(int(chunk_length_train * self.output_size), 1)
                stride = max(int(stride_fraction_train * self.output_size), 1)
            else:
                chunk = self.output_size
                stride = max(int(stride_fraction_valtest * self.output_size), 1)
            for s in range(0, record_len, stride):
                if min(s + chunk, record_len) - s < self.output_size:
                    break
                row_idx.append(i)
                starts.append(s)

        self._row_idx = np.asarray(row_idx, dtype=np.int64)
        self._start = np.asarray(starts, dtype=np.int64)
        logger.info("  [%s] %s windows over %s records (input_size=%.4gs -> %d samples @ %gHz)",
                    split, f"{len(self._row_idx):,}", f"{len(df):,}",
                    self.input_size, self.model_seq_len, self.fs_model)

    def __len__(self):
        return len(self._row_idx)

    def get_id_mapping(self):
        return self._row_idx

    def _read_record(self, idx) -> np.ndarray:
        sig = np.asarray(self.waveforms[idx]).astype(np.float32)

        if self.layout == "NHWC":
            if sig.ndim == 3 and sig.shape[0] == 1:
                sig = sig[0]               # (T, C)
            elif sig.ndim != 2:
                raise ValueError(f"unexpected NHWC sample shape: {sig.shape}")
            sig = sig.T                    # (C, T)
        elif self.layout == "NCT":
            if sig.ndim == 3 and sig.shape[0] == 1:
                sig = sig[0]
        else:
            raise ValueError(f"unsupported layout: {self.layout}")

        if sig.shape[0] != self.n_leads:
            raise ValueError(f"n_leads mismatch: got {sig.shape[0]}, expected {self.n_leads}")
        return sig

    def __getitem__(self, idx):
        row = int(self._row_idx[idx])
        start = int(self._start[idx])

        sig = self._read_record(row)

        if start < 0:  # train, non-chunkified: random crop over the whole record
            span = self._record_len - self.output_size
            start = 0 if span <= 0 else int(np.random.randint(0, span))
        sig = sig[:, start:start + self.output_size]

        if self.lead_perm is not None:
            sig = sig[self.lead_perm]

        sig = resample_signal(sig, self.source_fs, self.fs_model)
        sig = fit_length(sig, self.model_seq_len)

        if self.normalize and self.mean is not None and self.std is not None:
            sig = (sig - np.asarray(self.mean)[:, None]) / (np.asarray(self.std)[:, None] + 1e-8)
        sig = np.nan_to_num(sig, nan=0.0)

        return {
            "signal": torch.from_numpy(np.ascontiguousarray(sig)),
            "label": torch.from_numpy(self.labels[row].copy()),
            "fs": self.source_fs,
            "idx": idx,
            "ecg_id": row,
            # MoRyECG preprocessing-cache keys (mirrors H5ECGDataset); seg_idx is
            # the split-local row index, matching scripts/precompute_moryecg_cache.py.
            "ecg_filepath": self._wf_path,
            "ecg_seg_idx": row,
        }


def build_echonext_dataset(cfg: dict, split: str = "train") -> EchoNextDataset:
    """Instantiate :class:`EchoNextDataset` from a task ``data`` config section."""
    md_split = cfg.get("split_overrides", {}).get(split, split)
    return EchoNextDataset(
        waveform_npy=cfg["waveforms"][split],
        metadata_csv=cfg["metadata_csv"],
        split=split,
        md_split=md_split,
        split_col=cfg.get("split_col", "split"),
        label_cols=cfg["label_cols"],
        source_fs=int(cfg.get("source_fs", 250)),
        input_size=cfg["input_size"],
        fs_model=cfg["fs_model"],
        chunkify_train=bool(cfg.get("chunkify_train", False)),
        chunk_length_train=float(cfg.get("chunk_length_train", 1.0)),
        stride_fraction_train=float(cfg.get("stride_fraction_train", 1.0)),
        stride_fraction_valtest=float(cfg.get("stride_fraction_valtest", 1.0)),
        source_leads=cfg.get("source_leads"),
        lead_order=cfg.get("lead_order", "standard"),
        normalize=bool(cfg.get("normalize", False)),
        mean=cfg.get("mean"),
        std=cfg.get("std"),
        n_leads=int(cfg.get("n_leads", 12)),
        layout=str(cfg.get("layout", "NHWC")),
    )


def build_echonext_dataloaders(cfg: dict, split: str = "train"):
    """Single-process DataLoader (the DDP path lives in ``run.py``)."""
    from torch.utils.data import DataLoader

    ds = build_echonext_dataset(cfg, split)
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=(split == "train"),
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=(split == "train"),
    )
    return ds, loader
