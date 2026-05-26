"""
H5-backed ECG Dataset
====================
Load signals directly from H5 and join with the label CSV to return multi-hot labels.
"""

import os
import numpy as np
import pandas as pd
import h5py
import torch
from torch.utils.data import Dataset
from pathlib import Path


class H5ECGDataset(Dataset):
    """
    H5 ECG downstream-task dataset.

    Each sample is loaded from a single segment of the H5 file.
    - signal: (n_leads, target_length) float32
    - label:  multi-hot vector (num_classes,)

    Resampling logic:
      even when the fs differs per dataset (200/250/257/400/500/1000 Hz)
      convert to the fixed input expected by the model (target_fs * target_seconds = target_length).

      Examples: when ECG-JEPA expects 500 Hz x 5 s = 2500 samples
        - heedb 500Hz, 5000samples (10 s) → 500Hz keep, first 2500samples crop
        - code15 400Hz, 4096samples (10.24 s) → 500Hz upsample to(5120) → first 2500 crop
        - cpsc2021 200Hz, 2000samples (10 s) → 500Hz upsample to(5000) → first 2500 crop
        - ptb 1000Hz, 10000samples (10 s) → 500Hz downsample to(5000) → first 2500 crop
        - stpetersburg 257Hz → 500Hz upsample to → crop

    Args:
        h5_root:        H5 data/ directory's above directory
        table_csv:      ecg_table.csv (filepath, pid, rid, fs etc.)
        label_csv:      {dataset}_labels.csv (filepath + binary label column)
        label_cols:     useto label column list (None if so, label_csv's all non-key column)
        target_fs:      expected by the model sampling  (None if so, resampling inside )
        target_length:  expected by the model  length (samples, None if so,  inside )
        chunk_length:   encoder  in  window size (target_fs reference: samples ).
                        None if so, chunking none (all ECG as-is return).
                        random_crop=False (val/test): ECG 1
                        ⌊target_length/chunk_length⌋  non-overlapping chunk by extension.
                        random_crop=True  (train)  : ECG 1 per 1 sample, 
                        __getitem__ each  random offset from chunk_length only slice
                        (paper main_lite.py default chunkify_train=False).
        random_crop:    True then train mode (random offset), False then deterministic
                        chunking. (paper §3.3)
        seg_idx:        useto segment isdex (None if so, seg0 only, 'all' if so, all segment)
        normalize:      True then per-lead z-score (dataset mean/std)
        fold_col:       fold column name
        fold_ids:       useto fold ID list (None if so, all)
        mean:           per-lead mean (n_leads,) for normalization
        std:            per-lead std (n_leads,) for normalization
    """

    def __init__(
        self,
        h5_root:       str,
        table_csv:     str,
        label_csv:     str = None,
        label_cols:    list = None,
        target_fs:     int = None,
        target_length: int = None,
        chunk_length:  int = None,
        random_crop:   bool = False,
        seg_idx:       str = None,  # None(=0), 'all', or int
        normalize:     bool = False,
        fold_col:      str = None,
        fold_ids:      list = None,
        mean:          np.ndarray = None,
        std:           np.ndarray = None,
        task_type:     str = "binary",  # 'binary' | 'regression' | 'multi-label-binary' | 'classification_and_regression'
        target_mean:   np.ndarray = None,  # regression z-norm mean (paper-faithful)
        target_std:    np.ndarray = None,  # regression z-norm std
        cls_cols:      list = None,        # joint task only: classification subset (NaN-masked BCE)
        reg_cols:      list = None,        # joint task only: regression subset (NaN-masked L1, z-normed)
    ):
        self.h5_root = Path(h5_root)
        self.target_fs = target_fs
        self.target_length = target_length
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.task_type = task_type
        self.target_mean = np.asarray(target_mean, dtype=np.float32) if target_mean is not None else None
        self.target_std = np.asarray(target_std, dtype=np.float32) if target_std is not None else None
        self.cls_cols = list(cls_cols) if cls_cols is not None else None
        self.reg_cols = list(reg_cols) if reg_cols is not None else None
        self.num_cls = len(self.cls_cols) if self.cls_cols else 0
        self.num_reg = len(self.reg_cols) if self.reg_cols else 0

        # metadata table load
        self.table = pd.read_csv(table_csv, low_memory=False)

        # label CSV load + is
        if label_csv is not None and not os.path.exists(label_csv):
            raise FileNotFoundError(
                f"label_csv  file missing: {label_csv}\n"
                f"  → task yaml's label_csv path confirm. "
                f"(relative pathis  run.py automatically repo root reference as resolve.)"
            )
        self.has_labels = label_csv is not None
        if self.has_labels:
            label_df = pd.read_csv(label_csv, low_memory=False)
            key_cols = ["filepath"]
            n_before = len(self.table)
            # inner join: label ECG only training in use
            # (mimic4_table 800k ECG all, each task's cohort  )
            self.table = self.table.merge(label_df, on=key_cols, how="inner",
                                          suffixes=("_table", ""))
            if n_before != len(self.table):
                import logging
                logging.info(f"  Label join: {n_before:,} → {len(self.table):,} rows "
                             f"(label ECG only)")

            if self.task_type == "classification_and_regression":
                # joint task: explicit cls_cols + reg_cols required
                if not self.cls_cols or not self.reg_cols:
                    raise ValueError(
                        "task_type='classification_and_regression' requires both "
                        "cls_cols and reg_cols to be set in the task config.")
                label_cols = list(self.cls_cols) + list(self.reg_cols)
            elif label_cols is None:
                # key not all column = label
                non_label = {"filepath", "dataset", "pid", "rid", "sid", "oid",
                             "age", "gender", "height", "weight", "fs",
                             "channel_name", "nan_ratio", "amp_mean", "amp_std",
                             "amp_skewness", "amp_kurtosis", "bs_corr", "bs_dtw",
                             "strat_fold", "fold", "split"}
                label_cols = [c for c in label_df.columns if c not in non_label]
            self.label_cols = label_cols
            self.num_classes = len(label_cols)
        else:
            self.label_cols = []
            self.num_classes = 0

        # Fold filtering
        if fold_col and fold_ids is not None:
            self.table = self.table[self.table[fold_col].isin(fold_ids)].reset_index(drop=True)

        # segment extension (all if so, all segment per samples by)
        if seg_idx == "all":
            self._expand_segments()
        else:
            self.seg_indices = [int(seg_idx) if seg_idx is not None else 0] * len(self.table)

        # ── Chunk extension (paper §3.3 multi-window train + test-time aggregation) ──
        # train (random_crop=True):  1 sample/ECG, __getitem__ each  random offset
        # val/test (random_crop=False): ⌊target_length/chunk_length⌋ deterministic chunks
        self.chunk_length = chunk_length
        self.random_crop = random_crop
        if (chunk_length is not None and target_length is not None
                and chunk_length > 0 and chunk_length < target_length):
            if random_crop:
                self.n_chunks_per_ecg = 1   # random offset, 1 view per epoch
            else:
                self.n_chunks_per_ecg = int(target_length // chunk_length)
            self._random_max_start = int(target_length - chunk_length)
        else:
            self.n_chunks_per_ecg = 1
            self.chunk_length = None
            self._random_max_start = 0

        n_rows = len(self.table)
        if self.n_chunks_per_ecg > 1:
            self._row_idx = np.repeat(np.arange(n_rows), self.n_chunks_per_ecg)
            self._chunk_idx = np.tile(np.arange(self.n_chunks_per_ecg), n_rows)
        else:
            self._row_idx = np.arange(n_rows)
            self._chunk_idx = np.zeros(n_rows, dtype=int)

    def _expand_segments(self):
        """all segment per samples by extension."""
        expanded_rows = []
        expanded_segs = []
        for i, row in self.table.iterrows():
            h5_path = self.h5_root / row["filepath"]
            try:
                with h5py.File(h5_path, "r") as f:
                    n_segs = int(f["ECG/segments"].attrs.get("seg_len", 1))
                for s in range(n_segs):
                    expanded_rows.append(i)
                    expanded_segs.append(s)
            except Exception:
                expanded_rows.append(i)
                expanded_segs.append(0)
        self.table = self.table.iloc[expanded_rows].reset_index(drop=True)
        self.seg_indices = expanded_segs

    def __len__(self):
        return len(self._row_idx)

    def __getitem__(self, idx):
        table_idx = int(self._row_idx[idx])
        chunk_idx = int(self._chunk_idx[idx])
        row = self.table.iloc[table_idx]
        seg_i = self.seg_indices[table_idx] if hasattr(self, "seg_indices") else 0
        h5_path = self.h5_root / row["filepath"]

        # H5 from signal load
        with h5py.File(h5_path, "r") as f:
            fs = int(f["ECG/metadata"].attrs.get("fs", 500))
            sig = f[f"ECG/segments/{seg_i}/signal"][()].astype(np.float32)
            # sig: (n_leads, samples)

        # ── resampling + length  ──
        # 1stage: fs  then target_fs by resampling (upsample/downsample)
        if self.target_fs and self.target_fs != fs:
            sig = self._resample(sig, fs, self.target_fs)

        # 2stage: target_length in  crop or pad
        if self.target_length:
            sig = self._adjust_length(sig, self.target_length)

        # 3stage: chunk_length config then window slice
        # train (random_crop=True): random offset; val/test: deterministic chunk_idx
        if self.chunk_length is not None:
            if self.random_crop and self._random_max_start > 0:
                s = int(np.random.randint(0, self._random_max_start + 1))
            else:
                s = chunk_idx * self.chunk_length
            e = s + self.chunk_length
            sig = sig[:, s:e]

        # normalization
        if self.normalize and self.mean is not None and self.std is not None:
            sig = (sig - self.mean[:, None]) / (self.std[:, None] + 1e-8)

        # NaN → 0
        sig = np.nan_to_num(sig, nan=0.0)

        # label — task_type
        if self.has_labels:
            if self.task_type == "classification_and_regression":
                # joint task (paper mimic_preprocessing.py): cls + reg concatenated.
                # cls part: NaN preserved (masked BCE). reg part: NaN preserved, z-normed
                # with train-fold stats (target_mean/target_std cover the reg part only).
                cls_part = np.array([
                    float(row.get(c)) if pd.notna(row.get(c)) else np.nan
                    for c in self.cls_cols
                ], dtype=np.float32)
                reg_part = np.array([
                    float(row.get(c)) if pd.notna(row.get(c)) else np.nan
                    for c in self.reg_cols
                ], dtype=np.float32)
                if self.target_mean is not None and self.target_std is not None:
                    reg_part = (reg_part - self.target_mean) / (self.target_std + 1e-8)
                label = np.concatenate([cls_part, reg_part])
            elif self.task_type == "regression":
                # numeric value float32 by, NaN preserve (paper main_lite_ecg.py:122-133 mask (for))
                label = np.array([
                    float(row.get(c)) if pd.notna(row.get(c)) else np.nan
                    for c in self.label_cols
                ], dtype=np.float32)
                # paper z-normalize: (target - train_mean) / train_std
                if self.target_mean is not None and self.target_std is not None:
                    label = (label - self.target_mean) / (self.target_std + 1e-8)
            elif self.task_type == "multi-label-binary":
                # binary 0/1, NaN preserve (paper:114-118 mask (for) — mds_ed's missing label handling)
                vals = []
                for c in self.label_cols:
                    v = row.get(c)
                    if pd.isna(v):
                        vals.append(np.nan)
                    else:
                        s = str(v).lower()
                        vals.append(1.0 if s in ("true", "1", "1.0") else 0.0)
                label = np.array(vals, dtype=np.float32)
            else:
                # binary (default) — all NaN 0 as ( of-label,  negative by )
                label = np.array([
                    1.0 if str(row.get(c, "")).lower() in ("true", "1", "1.0") else 0.0
                    for c in self.label_cols
                ], dtype=np.float32)
        else:
            label = np.zeros(1, dtype=np.float32)

        return {
            "signal": torch.from_numpy(sig),          # (n_leads, chunk_length or target_length)
            "label":  torch.from_numpy(label),         # (num_classes,)
            "fs":     fs,
            "idx":    idx,
            "ecg_id": table_idx,                      # ECG-level id (eval  key)
            "ecg_filepath": str(h5_path),             # absolute H5 path — moryecg cache key
            "ecg_seg_idx":  int(seg_i),               # H5 segment idx — moryecg cache key
        }

    @staticmethod
    def _resample(sig, orig_fs, target_fs):
        """
        scipy based resampling (upsample/downsample).

        Examples:
          200Hz → 500Hz: upsample ×2.5
          400Hz → 500Hz: upsample ×1.25
          1000Hz → 500Hz: downsample ×0.5
          257Hz → 500Hz: upsample ×1.95
        """
        from scipy.signal import resample
        n_leads, orig_len = sig.shape
        target_len = int(round(orig_len * target_fs / orig_fs))
        if target_len == orig_len:
            return sig
        return resample(sig, target_len, axis=1).astype(np.float32)

    @staticmethod
    def _adjust_length(sig, target_length):
        """
         length crop or zero-pad.

        - length:  from target_length only crop
        - :  in zero-pad
        """
        n_leads, cur_len = sig.shape
        if cur_len >= target_length:
            return sig[:, :target_length]
        else:
            pad = np.zeros((n_leads, target_length - cur_len), dtype=sig.dtype)
            return np.concatenate([sig, pad], axis=1)

    def compute_stats(self):
        """per-lead mean/std compute (normalization (for))"""
        sums = None
        sq_sums = None
        count = 0
        for i in range(min(len(self), 5000)):
            item = self[i]
            sig = item["signal"].numpy()
            if sums is None:
                sums = np.zeros(sig.shape[0], dtype=np.float64)
                sq_sums = np.zeros(sig.shape[0], dtype=np.float64)
            sums += sig.mean(axis=1)
            sq_sums += (sig ** 2).mean(axis=1)
            count += 1
        mean = (sums / count).astype(np.float32)
        std = np.sqrt(sq_sums / count - mean ** 2).astype(np.float32)
        return mean, std


def build_dataloaders(cfg, split="train"):
    """Config from DataLoader generate."""
    from torch.utils.data import DataLoader

    ds = H5ECGDataset(
        h5_root=cfg["h5_root"],
        table_csv=cfg["table_csv"],
        label_csv=cfg.get("label_csv"),
        label_cols=cfg.get("label_cols"),
        target_fs=cfg.get("target_fs"),
        target_length=cfg.get("target_length"),
        chunk_length=cfg.get("chunk_length"),
        random_crop=(split == "train"),
        seg_idx=cfg.get("seg_idx", None),
        normalize=cfg.get("normalize", False),
        fold_col=cfg.get("fold_col"),
        fold_ids=cfg.get(f"{split}_folds"),
        mean=cfg.get("mean"),
        std=cfg.get("std"),
        task_type=cfg.get("task_type", "binary"),
        target_mean=cfg.get("target_mean"),
        target_std=cfg.get("target_std"),
        cls_cols=cfg.get("cls_cols"),
        reg_cols=cfg.get("reg_cols"),
    )
    nw = int(os.environ.get("NUM_WORKERS", cfg.get("num_workers", 4)))
    loader = DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 64),
        shuffle=(split == "train"),
        num_workers=nw,
        pin_memory=True,
        drop_last=(split == "train"),
        persistent_workers=(nw > 0),
        prefetch_factor=4 if nw > 0 else None,
    )
    return ds, loader
