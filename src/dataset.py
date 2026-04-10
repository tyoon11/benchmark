"""
H5 기반 ECG Dataset
====================
H5 파일에서 직접 신호를 로드하고, 라벨 CSV와 조인하여 multi-hot 라벨을 반환합니다.
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
    H5 ECG 다운스트림 태스크 Dataset.

    각 샘플은 H5 파일의 하나의 세그먼트에서 로드됩니다.
    - signal: (n_leads, target_length) float32
    - label:  multi-hot vector (num_classes,)

    리샘플링 로직:
      데이터셋마다 fs가 다르더라도 (200/250/257/400/500/1000 Hz)
      모델이 기대하는 고정 입력 (target_fs × target_seconds = target_length)으로 변환합니다.

      예시: ECG-JEPA가 500Hz × 5초 = 2500 샘플을 기대할 때
        - heedb 500Hz, 5000샘플(10초) → 500Hz로 유지, 앞 2500샘플 crop
        - code15 400Hz, 4096샘플(10.24초) → 500Hz로 upsample(5120) → 앞 2500 crop
        - cpsc2021 200Hz, 2000샘플(10초) → 500Hz로 upsample(5000) → 앞 2500 crop
        - ptb 1000Hz, 10000샘플(10초) → 500Hz로 downsample(5000) → 앞 2500 crop
        - stpetersburg 257Hz → 500Hz로 upsample → crop

    Args:
        h5_root:        H5 data/ 폴더의 상위 디렉토리
        table_csv:      ecg_table.csv (filepath, pid, rid, fs 등)
        label_csv:      {dataset}_labels.csv (filepath + binary 라벨 컬럼)
        label_cols:     사용할 라벨 컬럼 목록 (None이면 label_csv의 모든 non-key 컬럼)
        target_fs:      모델이 기대하는 샘플링 주파수 (None이면 리샘플링 안 함)
        target_length:  모델이 기대하는 시계열 길이 (샘플 수, None이면 조정 안 함)
        seg_idx:        사용할 세그먼트 인덱스 (None이면 seg0만, 'all'이면 모든 세그먼트)
        normalize:      True면 per-lead z-score (dataset mean/std)
        fold_col:       fold 컬럼명
        fold_ids:       사용할 fold ID 리스트 (None이면 전체)
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
        seg_idx:       str = None,  # None(=0), 'all', or int
        normalize:     bool = False,
        fold_col:      str = None,
        fold_ids:      list = None,
        mean:          np.ndarray = None,
        std:           np.ndarray = None,
    ):
        self.h5_root = Path(h5_root)
        self.target_fs = target_fs
        self.target_length = target_length
        self.normalize = normalize
        self.mean = mean
        self.std = std

        # 메타 테이블 로드
        self.table = pd.read_csv(table_csv, low_memory=False)

        # 라벨 CSV 로드 + 조인
        self.has_labels = label_csv is not None and os.path.exists(label_csv)
        if self.has_labels:
            label_df = pd.read_csv(label_csv, low_memory=False)
            key_cols = ["filepath"]
            self.table = self.table.merge(label_df, on=key_cols, how="left",
                                          suffixes=("", "_label"))

            if label_cols is None:
                # key가 아닌 모든 컬럼 = 라벨
                non_label = {"filepath", "dataset", "pid", "rid", "sid", "oid",
                             "age", "gender", "height", "weight", "fs",
                             "channel_name", "nan_ratio", "amp_mean", "amp_std",
                             "amp_skewness", "amp_kurtosis", "bs_corr", "bs_dtw"}
                label_cols = [c for c in label_df.columns if c not in non_label]
            self.label_cols = label_cols
            self.num_classes = len(label_cols)
        else:
            self.label_cols = []
            self.num_classes = 0

        # Fold 필터링
        if fold_col and fold_ids is not None:
            self.table = self.table[self.table[fold_col].isin(fold_ids)].reset_index(drop=True)

        # 세그먼트 확장 (all이면 모든 세그먼트를 개별 샘플로)
        if seg_idx == "all":
            self._expand_segments()
        else:
            self.seg_indices = [int(seg_idx) if seg_idx is not None else 0] * len(self.table)

    def _expand_segments(self):
        """모든 세그먼트를 개별 샘플로 확장합니다."""
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
        return len(self.table)

    def __getitem__(self, idx):
        row = self.table.iloc[idx]
        seg_i = self.seg_indices[idx] if hasattr(self, "seg_indices") else 0
        h5_path = self.h5_root / row["filepath"]

        # H5에서 신호 로드
        with h5py.File(h5_path, "r") as f:
            fs = int(f["ECG/metadata"].attrs.get("fs", 500))
            sig = f[f"ECG/segments/{seg_i}/signal"][()].astype(np.float32)
            # sig: (n_leads, samples)

        # ── 리샘플링 + 길이 조정 ──
        # 1단계: fs가 다르면 target_fs로 리샘플링 (upsample/downsample)
        if self.target_fs and self.target_fs != fs:
            sig = self._resample(sig, fs, self.target_fs)

        # 2단계: target_length에 맞춰 crop 또는 pad
        if self.target_length:
            sig = self._adjust_length(sig, self.target_length)

        # 정규화
        if self.normalize and self.mean is not None and self.std is not None:
            sig = (sig - self.mean[:, None]) / (self.std[:, None] + 1e-8)

        # NaN → 0
        sig = np.nan_to_num(sig, nan=0.0)

        # 라벨
        if self.has_labels:
            label = np.array([
                1.0 if str(row.get(c, "")).lower() in ("true", "1", "1.0") else 0.0
                for c in self.label_cols
            ], dtype=np.float32)
        else:
            label = np.zeros(1, dtype=np.float32)

        return {
            "signal": torch.from_numpy(sig),          # (n_leads, samples)
            "label":  torch.from_numpy(label),         # (num_classes,)
            "fs":     fs,
            "idx":    idx,
        }

    @staticmethod
    def _resample(sig, orig_fs, target_fs):
        """
        scipy 기반 리샘플링 (upsample/downsample).

        예시:
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
        고정 길이로 crop 또는 zero-pad.

        - 길면: 앞에서 target_length만큼 crop
        - 짧으면: 뒤에 zero-pad
        """
        n_leads, cur_len = sig.shape
        if cur_len >= target_length:
            return sig[:, :target_length]
        else:
            pad = np.zeros((n_leads, target_length - cur_len), dtype=sig.dtype)
            return np.concatenate([sig, pad], axis=1)

    def compute_stats(self):
        """per-lead mean/std 계산 (normalization용)"""
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
    """Config에서 DataLoader를 생성합니다."""
    from torch.utils.data import DataLoader

    ds = H5ECGDataset(
        h5_root=cfg["h5_root"],
        table_csv=cfg["table_csv"],
        label_csv=cfg.get("label_csv"),
        label_cols=cfg.get("label_cols"),
        target_fs=cfg.get("target_fs"),
        target_length=cfg.get("target_length"),
        seg_idx=cfg.get("seg_idx", None),
        normalize=cfg.get("normalize", False),
        fold_col=cfg.get("fold_col"),
        fold_ids=cfg.get(f"{split}_folds"),
        mean=cfg.get("mean"),
        std=cfg.get("std"),
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.get("batch_size", 64),
        shuffle=(split == "train"),
        num_workers=cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=(split == "train"),
    )
    return ds, loader
