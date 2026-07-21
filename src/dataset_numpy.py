"""
NumPy based ECG Dataset (EchoNext (for))
====================================
PhysioNet EchoNext 1.1.0 like (.npy waveforms + metadata.csv) by ed dataset
H5 convert without directly load.

EchoNext format:
  - EchoNext_<split>_waveforms.npy : (N, 1, 2500, 12) float64, 250Hz, 12-lead, 10s
    (already median-filter + percentile-clip + dataset-wide z-score handling )
  - echonext_metadata_100k.csv     : split column + 11 binary echo flag label

H5ECGDataset and identically (n_leads, target_length) float32 + multi-hot label return
DownstreamWrapper / DownstreamTrainer as-is re-use available.
"""

import logging
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EchoNextDataset(Dataset):
    """
    Args:
        waveform_npy:    the split's .npy path (shape: (N, 1, T, C) or (N, C, T))
        metadata_csv:    all metadatadata CSV
        split:           'train' | 'val' | 'test' | 'no_split'
        split_col:       split column name (default 'split')
        label_cols:      useto label column list (binary flag)
        source_fs:       waveform native fs (250)
        target_fs:       expected by the model fs (None=resample inside )
        target_length:   expected by the model length (samples)
        chunk_length:    encoder window size (target_fs reference: samples ). config
                         random_crop=False (val/test): ⌊target_length/chunk_length⌋
                                                       deterministic non-overlapping chunks
                         random_crop=True  (train): 1 sample/ECG, random offset
                         (paper §3.3 multi-window).
        random_crop:     True (train) / False (val/test). chunk_length configed  only .
        normalize:       True then mean/std by add z-score (EchoNext already normalization )
        mean, std:       per-lead (n_leads,)
        n_leads:         12
        layout:          'NHWC'(=(N,1,T,C), EchoNext default) or 'NCT'(=(N,C,T))
    """

    def __init__(
        self,
        waveform_npy:  str,
        metadata_csv:  str,
        split:         str,
        split_col:     str = "split",
        label_cols:    list = None,
        source_fs:     int = 250,
        target_fs:     int = None,
        target_length: int = None,
        chunk_length:  int = None,
        random_crop:   bool = False,
        normalize:     bool = False,
        mean:          np.ndarray = None,
        std:           np.ndarray = None,
        n_leads:       int = 12,
        layout:        str = "NHWC",
    ):
        if label_cols is None or len(label_cols) == 0:
            raise ValueError("label_cols min 1 or more required.")

        self.source_fs = source_fs
        self.target_fs = target_fs
        self.target_length = target_length
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.n_leads = n_leads
        self.layout = layout
        self.label_cols = list(label_cols)
        self.num_classes = len(self.label_cols)
        self.has_labels = True

        # mmap as .npy load (5-16GB file all  re- )
        self.waveforms = np.load(waveform_npy, mmap_mode="r")
        # Absolute .npy path used as the MoRyECG preprocessing-cache key
        # (matches scripts/precompute_moryecg_cache.py collect_numpy_segments,
        # which keys each split's cache by (npy_abspath, split_local_row_idx)).
        self._wf_path = os.path.abspath(waveform_npy)

        # split filter
        df = pd.read_csv(metadata_csv, low_memory=False)
        if split_col not in df.columns:
            raise ValueError(f"metadata_csv in '{split_col}' column .")
        df = df[df[split_col] == split].reset_index(drop=True)

        n_npy = self.waveforms.shape[0]
        if len(df) != n_npy:
            raise ValueError(
                f"split='{split}' rows match: csv={len(df)}, npy={n_npy}. "
                f"npy file  split download incompletecase ."
            )

        miss = [c for c in self.label_cols if c not in df.columns]
        if miss:
            raise ValueError(f"label column metadata in none: {miss}")

        self.df = df

        # label in advance numpy by extract (binary 0/1, NaN→0)
        labels = df[self.label_cols].values.astype(np.float32)
        labels = np.nan_to_num(labels, nan=0.0)
        self.labels = labels  # (N, num_classes)

        # ── Chunk extension (paper §3.3 multi-window) ──
        # train (random_crop=True):  1 sample/ECG, __getitem__ each  random offset
        # val/test (random_crop=False): ⌊target_length/chunk_length⌋ deterministic chunks
        self.chunk_length = chunk_length
        self.random_crop = random_crop
        if (chunk_length is not None and target_length is not None
                and chunk_length > 0 and chunk_length < target_length):
            if random_crop:
                self.n_chunks_per_ecg = 1
            else:
                self.n_chunks_per_ecg = int(target_length // chunk_length)
            self._random_max_start = int(target_length - chunk_length)
        else:
            self.n_chunks_per_ecg = 1
            self.chunk_length = None
            self._random_max_start = 0

        n_rows = len(self.df)
        if self.n_chunks_per_ecg > 1:
            self._row_idx = np.repeat(np.arange(n_rows), self.n_chunks_per_ecg)
            self._chunk_idx = np.tile(np.arange(self.n_chunks_per_ecg), n_rows)
        else:
            self._row_idx = np.arange(n_rows)
            self._chunk_idx = np.zeros(n_rows, dtype=int)

    def __len__(self):
        return len(self._row_idx)

    def _read_signal(self, idx) -> np.ndarray:
        """(n_leads, target_length) float32 signal return."""
        sig = np.asarray(self.waveforms[idx]).astype(np.float32)

        # layout normalization → (n_leads, T)
        if self.layout == "NHWC":
            # (1, T, C)
            if sig.ndim == 3 and sig.shape[0] == 1:
                sig = sig[0]              # (T, C)
            elif sig.ndim == 2:
                pass                       # (T, C)
            else:
                raise ValueError(f"NHWC exampleabove and  shape: {sig.shape}")
            sig = sig.T                   # (C, T)
        elif self.layout == "NCT":
            if sig.ndim == 3 and sig.shape[0] == 1:
                sig = sig[0]              # (C, T)
        else:
            raise ValueError(f"supportdo not  layout: {self.layout}")

        if sig.shape[0] != self.n_leads:
            raise ValueError(f"n_leads match: got {sig.shape[0]}, expected {self.n_leads}")

        if self.target_fs and self.target_fs != self.source_fs:
            sig = self._resample(sig, self.source_fs, self.target_fs)

        if self.target_length:
            sig = self._adjust_length(sig, self.target_length)

        if self.normalize and self.mean is not None and self.std is not None:
            sig = (sig - self.mean[:, None]) / (self.std[:, None] + 1e-8)

        sig = np.nan_to_num(sig, nan=0.0)
        return sig

    def __getitem__(self, idx):
        table_idx = int(self._row_idx[idx])
        chunk_idx = int(self._chunk_idx[idx])
        sig = self._read_signal(table_idx)

        # Chunk slice — train: random offset, val/test: deterministic chunk_idx
        if self.chunk_length is not None:
            if self.random_crop and self._random_max_start > 0:
                s = int(np.random.randint(0, self._random_max_start + 1))
            else:
                s = chunk_idx * self.chunk_length
            e = s + self.chunk_length
            sig = sig[:, s:e]

        label = self.labels[table_idx]
        return {
            "signal": torch.from_numpy(sig),
            "label":  torch.from_numpy(label),
            "fs":     self.source_fs,
            "idx":    idx,
            "ecg_id": table_idx,
            # MoRyECG preprocessing-cache keys (mirrors H5ECGDataset). seg_idx is
            # the split-local row index; valid because echonext has no chunking /
            # random crop (chunk_length == target_length → one window per record).
            "ecg_filepath": self._wf_path,
            "ecg_seg_idx":  int(table_idx),
        }

    @staticmethod
    def _resample(sig, orig_fs, target_fs):
        from scipy.signal import resample
        n_leads, orig_len = sig.shape
        target_len = int(round(orig_len * target_fs / orig_fs))
        if target_len == orig_len:
            return sig
        return resample(sig, target_len, axis=1).astype(np.float32)

    @staticmethod
    def _adjust_length(sig, target_length):
        n_leads, cur_len = sig.shape
        if cur_len >= target_length:
            return sig[:, :target_length]
        pad = np.zeros((n_leads, target_length - cur_len), dtype=sig.dtype)
        return np.concatenate([sig, pad], axis=1)


def build_echonext_dataloaders(cfg: dict, split: str = "train"):
    """
    EchoNext (for) DataLoader  (single GPU path (for); DDP run.py from sampler add).

    cfg Examples:
      loader_type: echonext_numpy
      metadata_csv: /.../echonext_metadata_100k.csv
      waveforms:
        train: /.../EchoNext_train_waveforms.npy
        val:   /.../EchoNext_val_waveforms.npy
        test:  /.../EchoNext_test_waveforms.npy
      label_cols: [shd_moderate_or_greater_flag]
      source_fs: 250
      target_fs: 250
      target_length: 2500
      normalize: false
      batch_size: 64
      num_workers: 8
    """
    from torch.utils.data import DataLoader

    waveform_npy = cfg["waveforms"][split]
    ds = EchoNextDataset(
        waveform_npy=waveform_npy,
        metadata_csv=cfg["metadata_csv"],
        split=split,
        split_col=cfg.get("split_col", "split"),
        label_cols=cfg["label_cols"],
        source_fs=int(cfg.get("source_fs", 250)),
        target_fs=cfg.get("target_fs"),
        target_length=cfg.get("target_length"),
        chunk_length=cfg.get("chunk_length"),
        random_crop=(split == "train"),
        normalize=bool(cfg.get("normalize", False)),
        mean=cfg.get("mean"),
        std=cfg.get("std"),
        n_leads=int(cfg.get("n_leads", 12)),
        layout=str(cfg.get("layout", "NHWC")),
    )
    loader = DataLoader(
        ds,
        batch_size=int(cfg.get("batch_size", 64)),
        shuffle=(split == "train"),
        num_workers=int(cfg.get("num_workers", 4)),
        pin_memory=True,
        drop_last=(split == "train"),
    )
    return ds, loader
