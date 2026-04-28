"""
NumPy 기반 ECG Dataset (EchoNext용)
====================================
PhysioNet EchoNext 1.1.0 처럼 (.npy waveforms + metadata.csv)로 배포된 데이터셋을
H5 변환 없이 직접 로드합니다.

EchoNext 포맷:
  - EchoNext_<split>_waveforms.npy : (N, 1, 2500, 12) float64, 250Hz, 12-lead, 10초
    (이미 median-filter + percentile-clip + dataset-wide z-score 처리됨)
  - echonext_metadata_100k.csv     : split 컬럼 + 11개 binary echo flag 라벨

H5ECGDataset과 동일하게 (n_leads, target_length) float32 + multi-hot label을 반환하므로
DownstreamWrapper / DownstreamTrainer 그대로 재사용 가능.
"""

import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class EchoNextDataset(Dataset):
    """
    Args:
        waveform_npy:    해당 split의 .npy 경로 (shape: (N, 1, T, C) 또는 (N, C, T))
        metadata_csv:    전체 메타데이터 CSV
        split:           'train' | 'val' | 'test' | 'no_split'
        split_col:       split 컬럼명 (기본 'split')
        label_cols:      사용할 라벨 컬럼 목록 (binary flag들)
        source_fs:       waveform native fs (250)
        target_fs:       모델이 기대하는 fs (None=리샘플 안 함)
        target_length:   모델이 기대하는 길이 (샘플)
        normalize:       True면 mean/std로 추가 z-score (EchoNext는 이미 정규화됨)
        mean, std:       per-lead (n_leads,)
        n_leads:         12
        layout:          'NHWC'(=(N,1,T,C), EchoNext 기본) 또는 'NCT'(=(N,C,T))
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
        normalize:     bool = False,
        mean:          np.ndarray = None,
        std:           np.ndarray = None,
        n_leads:       int = 12,
        layout:        str = "NHWC",
    ):
        if label_cols is None or len(label_cols) == 0:
            raise ValueError("label_cols는 최소 1개 이상 필요합니다.")

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

        # mmap으로 .npy 로드 (5-16GB 파일이라 전체 메모리 적재 불가)
        self.waveforms = np.load(waveform_npy, mmap_mode="r")

        # split 필터
        df = pd.read_csv(metadata_csv, low_memory=False)
        if split_col not in df.columns:
            raise ValueError(f"metadata_csv에 '{split_col}' 컬럼이 없습니다.")
        df = df[df[split_col] == split].reset_index(drop=True)

        n_npy = self.waveforms.shape[0]
        if len(df) != n_npy:
            raise ValueError(
                f"split='{split}' 행 수 불일치: csv={len(df)}, npy={n_npy}. "
                f"npy 파일이 다른 split이거나 download가 incomplete일 수 있습니다."
            )

        miss = [c for c in self.label_cols if c not in df.columns]
        if miss:
            raise ValueError(f"라벨 컬럼이 metadata에 없음: {miss}")

        self.df = df

        # 라벨을 미리 numpy로 추출 (binary 0/1, NaN→0)
        labels = df[self.label_cols].values.astype(np.float32)
        labels = np.nan_to_num(labels, nan=0.0)
        self.labels = labels  # (N, num_classes)

    def __len__(self):
        return len(self.df)

    def _read_signal(self, idx) -> np.ndarray:
        """(n_leads, target_length) float32 신호 반환."""
        sig = np.asarray(self.waveforms[idx]).astype(np.float32)

        # layout 정규화 → (n_leads, T)
        if self.layout == "NHWC":
            # (1, T, C)
            if sig.ndim == 3 and sig.shape[0] == 1:
                sig = sig[0]              # (T, C)
            elif sig.ndim == 2:
                pass                       # (T, C)
            else:
                raise ValueError(f"NHWC 예상과 다른 shape: {sig.shape}")
            sig = sig.T                   # (C, T)
        elif self.layout == "NCT":
            if sig.ndim == 3 and sig.shape[0] == 1:
                sig = sig[0]              # (C, T)
        else:
            raise ValueError(f"지원하지 않는 layout: {self.layout}")

        if sig.shape[0] != self.n_leads:
            raise ValueError(f"n_leads 불일치: got {sig.shape[0]}, expected {self.n_leads}")

        if self.target_fs and self.target_fs != self.source_fs:
            sig = self._resample(sig, self.source_fs, self.target_fs)

        if self.target_length:
            sig = self._adjust_length(sig, self.target_length)

        if self.normalize and self.mean is not None and self.std is not None:
            sig = (sig - self.mean[:, None]) / (self.std[:, None] + 1e-8)

        sig = np.nan_to_num(sig, nan=0.0)
        return sig

    def __getitem__(self, idx):
        sig = self._read_signal(idx)
        label = self.labels[idx]
        return {
            "signal": torch.from_numpy(sig),
            "label":  torch.from_numpy(label),
            "fs":     self.source_fs,
            "idx":    idx,
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
    EchoNext용 DataLoader 빌더 (single GPU 경로용; DDP는 run.py에서 sampler 추가).

    cfg 예시:
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
