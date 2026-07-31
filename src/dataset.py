"""
H5-backed ECG Dataset
=====================
Reproduces the windowing/resampling contract of the original
``ecg-fm-benchmarking`` (``main_lite_base.Main_Lite.setup`` +
``clinical_ts.data.time_series_dataset.TimeSeriesDataset``) on top of the HEEDB
H5 store.

Original contract, restated
---------------------------
``input_size`` (seconds) and ``fs_model`` come from the *model*; ``fs_data`` is
the dataset's native rate. Then::

    output_size          = int(input_size * fs_data)          # crop, in native samples
    chunk_length_train   = 0 if not chunkify_train            # one window per record
    chunk_length_valtest = output_size
    stride_valtest       = stride_fraction_valtest * output_size

* train: one entry per record spanning the whole record; ``__getitem__`` takes a
  random crop of ``output_size``.
* val/test: the record is cut into windows of ``output_size`` at ``stride``;
  any trailing window shorter than ``output_size`` — and everything after it —
  is dropped. Predictions are averaged per record afterwards.
* records shorter than ``output_size`` produce no windows at all, i.e. they are
  dropped (the original never zero-pads).
* the crop is taken at the native rate and only *then* resampled to ``fs_model``
  with a band-limited resampler.

Deviations from the previous implementation of this file, all of which changed
results for the pretrained baselines:

* leads are permuted from the HEEDB order to whatever the encoder declares
  (see :mod:`src.leads`) — previously 9 of 12 leads reached the model in the
  wrong slot;
* resampling is band-limited instead of ``F.interpolate(mode="linear")``;
* windows are cut at the native rate over the *whole* record (all segments)
  rather than over the first 10 s of segment 0;
* short records are dropped instead of zero-padded;
* ``min_data_length`` reproduces the ``data_length >= N`` cohort filters.

``target_fs`` / ``target_length`` from older task configs are ignored (with a
warning): the window is now fully determined by the encoder's ``input_size`` /
``model_fs`` and the dataset's native rate, exactly as in the original.
"""

from __future__ import annotations

import logging
import os
import random
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .leads import (build_lead_permutation, describe_permutation,
                    parse_channel_names, resolve_target_order)
from .signal_utils import fit_length, load_record_lengths, resample_signal

logger = logging.getLogger(__name__)

# Columns of *_table.csv / *_labels.csv that are metadata rather than labels.
NON_LABEL_COLS = {
    "filepath", "dataset", "pid", "rid", "sid", "oid",
    "age", "gender", "height", "weight", "fs",
    "channel_name", "nan_ratio", "amp_mean", "amp_std",
    "amp_skewness", "amp_kurtosis", "bs_corr", "bs_dtw",
    "strat_fold", "fold", "split",
}


# Bumped whenever the MoRyECG preprocessing pipeline changes in a way that
# invalidates precomputed cache entries.
PIPELINE_VERSION = "2026.07.31-parity"
CACHE_STAMP_FILE = "_cache_stamp.json"

_stamp_warned = set()


def cache_stamp(model_fs, lead_order, seg_mode="all") -> dict:
    """Identity of a MoRyECG preprocessing cache.

    Entries are whole-*segment* R-peak/beat/STFT bundles read straight from the
    H5 at ``model_fs`` in the store's own lead order — the benchmark window
    (``input_size``) is deliberately absent, because on a cache hit the encoder
    ignores the windowed tensor and consumes the bundle directly. What does
    invalidate an entry is the preprocessing code, the rate it ran at, or the
    lead order; what makes a cache *incomplete* is ``seg_mode``.
    """
    return {
        "pipeline": PIPELINE_VERSION,
        "model_fs": float(model_fs),
        "lead_order": str(lead_order),
        "seg_mode": str(seg_mode),
    }


def check_cache_stamp(cache_root, model_fs, lead_order, seg_mode="all", split=""):
    """Warn when a MoRyECG preprocessing cache may be stale or incomplete.

    The cache is keyed by ``(filepath, seg_idx)`` only, so a mismatched entry
    still looks like a hit. ``scripts/precompute_moryecg_cache.py`` writes the
    stamp; caches predating it have none, and those were built with
    ``seg_idx=0`` — every segment past the first is missing and will fall back
    to (correct but slow) live preprocessing.
    """
    import json

    stamp_path = Path(cache_root) / CACHE_STAMP_FILE
    want = cache_stamp(model_fs, lead_order, seg_mode)
    key = (str(stamp_path), tuple(sorted(want.items())))
    if key in _stamp_warned:
        return
    _stamp_warned.add(key)

    if not stamp_path.exists():
        logger.warning(
            "[%s] MORYECG_CACHE=%s has no %s. Entry *content* is still valid "
            "(whole-segment bundles are independent of the window), but caches "
            "written before this file only covered seg_idx=0, so multi-segment "
            "records will miss and preprocess live. Top it up with "
            "scripts/precompute_moryecg_cache.py.", split, cache_root, CACHE_STAMP_FILE)
        return

    try:
        have = json.loads(stamp_path.read_text())
    except (OSError, ValueError) as exc:
        logger.warning("[%s] could not read %s: %s", split, stamp_path, exc)
        return

    diff = {k: (have.get(k), v) for k, v in want.items() if have.get(k) != v}
    if diff:
        logger.warning(
            "[%s] MORYECG_CACHE=%s stamp disagrees with this run — %s "
            "(have, want). Rebuild with --force, or unset MORYECG_CACHE.",
            split, cache_root, diff)


class H5ECGDataset(Dataset):
    """H5 ECG dataset with paper-faithful cropping, chunking and resampling.

    Args:
        h5_root:        directory containing the ``data/`` tree referenced by
                        ``filepath`` in the table CSV.
        table_csv:      ``{dataset}_table.csv`` (filepath, fs, channel_name, strat_fold, ...).
        label_csv:      ``{dataset}_labels.csv``; inner-joined on ``filepath``.
        label_cols:     label columns to use (defaults to every non-metadata column).
        split:          ``'train'`` | ``'val'`` | ``'test'`` — selects the
                        chunking regime (random crop vs. strided windows).
        input_size:     window length in **seconds** (encoder contract).
        fs_model:       rate the encoder expects, in Hz (encoder contract).
        fs_data:        override the dataset's native rate; ``None`` uses the
                        per-record ``fs`` from the table/H5 header.
        chunkify_train: original ``--chunkify-train`` (default False).
        chunk_length_train / stride_fraction_train / stride_fraction_valtest:
                        multiples of the window, as in the original CLI.
        min_data_length: drop records whose total native length is below this
                        (reproduces ``df[df.data_length >= N]``).
        seg_mode:       ``'all'`` (every H5 segment, default) or ``'first'``.
        lead_order:     ``'standard'`` (default), ``'heedb'``, ``'native'`` or an
                        explicit list of lead names.
        normalize/mean/std:  optional per-lead z-scoring (original default: off).
        fold_col/fold_ids:   fold-based split selection.
        task_type:      ``binary`` | ``multi-label-binary`` | ``regression`` |
                        ``classification_and_regression``.
        target_mean/target_std: train-fold statistics for regression z-scoring.
        cls_cols/reg_cols:      column split for the joint MIMIC task.
    """

    def __init__(
        self,
        h5_root: str,
        table_csv: str,
        label_csv: str = None,
        label_cols: list = None,
        *,
        split: str = "train",
        input_size: float = 2.5,
        fs_model: float = 500,
        fs_data: float = None,
        chunkify_train: bool = False,
        chunk_length_train: float = 1.0,
        stride_fraction_train: float = 1.0,
        stride_fraction_valtest: float = 1.0,
        min_data_length: int = None,
        seg_mode: str = "all",
        lead_order="standard",
        normalize: bool = False,
        mean: np.ndarray = None,
        std: np.ndarray = None,
        fold_col: str = None,
        fold_ids: list = None,
        task_type: str = "binary",
        target_mean: np.ndarray = None,
        target_std: np.ndarray = None,
        cls_cols: list = None,
        reg_cols: list = None,
        # deprecated, accepted so old configs do not crash
        target_fs: int = None,
        target_length: int = None,
        chunk_length: int = None,
        random_crop: bool = None,
        seg_idx=None,
    ):
        self.h5_root = Path(h5_root)
        self.table_csv = str(table_csv)
        self.split = split
        self.is_train = (split == "train")
        self.input_size = float(input_size)
        self.fs_model = float(fs_model)
        self.fs_data_override = float(fs_data) if fs_data else None
        self.normalize = normalize
        self.mean = mean
        self.std = std
        self.task_type = task_type
        self.target_mean = np.asarray(target_mean, dtype=np.float32) if target_mean is not None else None
        self.target_std = np.asarray(target_std, dtype=np.float32) if target_std is not None else None
        self.cls_cols = list(cls_cols) if cls_cols is not None else None
        self.reg_cols = list(reg_cols) if reg_cols is not None else None

        for name, value in (("target_fs", target_fs), ("target_length", target_length),
                            ("chunk_length", chunk_length), ("seg_idx", seg_idx)):
            if value is not None:
                logger.warning(
                    "H5ECGDataset: '%s' is deprecated and ignored — the window is "
                    "derived from the encoder contract (input_size=%.4gs, fs_model=%g) "
                    "and the record's native rate.", name, self.input_size, self.fs_model)

        # Model-side window length, i.e. what the encoder receives.
        self.model_seq_len = int(round(self.input_size * self.fs_model))

        # ── table + labels ────────────────────────────────────────────────
        self.table = pd.read_csv(table_csv, low_memory=False)
        # Keep the unfiltered filepath list so the length cache is built once for
        # the whole dataset instead of once per split.
        all_filepaths = self.table["filepath"].tolist()

        if label_csv is not None and not os.path.exists(label_csv):
            raise FileNotFoundError(
                f"label_csv not found: {label_csv}\n"
                f"  -> check 'data.label_csv' in the task yaml (relative paths are "
                f"resolved against the benchmark root by run.py).")
        self.has_labels = label_csv is not None
        if self.has_labels:
            label_df = pd.read_csv(label_csv, low_memory=False)
            n_before = len(self.table)
            self.table = self.table.merge(label_df, on=["filepath"], how="inner",
                                          suffixes=("_table", ""))
            if n_before != len(self.table):
                logger.info("  Label join: %s -> %s rows (labelled records only)",
                            f"{n_before:,}", f"{len(self.table):,}")

            if self.task_type == "classification_and_regression":
                if not self.cls_cols or not self.reg_cols:
                    raise ValueError(
                        "task_type='classification_and_regression' requires both "
                        "cls_cols and reg_cols in the task config.")
                label_cols = list(self.cls_cols) + list(self.reg_cols)
            elif label_cols is None:
                label_cols = [c for c in label_df.columns if c not in NON_LABEL_COLS]
            self.label_cols = label_cols
            self.num_classes = len(label_cols)
        else:
            self.label_cols = []
            self.num_classes = 0

        # ── fold filtering ────────────────────────────────────────────────
        if fold_col and fold_ids is not None:
            self.table = self.table[self.table[fold_col].isin(fold_ids)]
        self.table = self.table.reset_index(drop=True)

        if len(self.table) == 0:
            raise RuntimeError(
                f"[{split}] no records left after the label join and fold filter "
                f"(fold_col={fold_col!r}, fold_ids={fold_ids}). Check that those folds "
                f"exist in {table_csv}.")

        # ── lead permutation (H5 order -> encoder order) ──────────────────
        self.target_leads = resolve_target_order(lead_order)
        self.source_leads = None
        channel_col = next((c for c in ("channel_name", "channel_name_table")
                            if c in self.table.columns), None)
        if channel_col:
            self.source_leads = parse_channel_names(self.table[channel_col].iloc[0])
        if self.target_leads and not self.source_leads:
            # Silently skipping the permutation is how the pre-parity pipeline fed
            # 9 mis-ordered leads to every pretrained baseline — make it loud.
            logger.warning(
                "[%s] lead_order=%r was requested but the source order is unknown "
                "(no usable 'channel_name' column in %s). Leads are passed through "
                "UNCHANGED — if the store is not already in %s order the pretrained "
                "encoders will see permuted leads.",
                split, lead_order, Path(table_csv).name, self.target_leads[:6])
        self.lead_perm = build_lead_permutation(self.source_leads, self.target_leads)
        logger.info("  [%s] %s", split,
                    describe_permutation(self.source_leads, self.target_leads, self.lead_perm))

        # ── per-segment lengths ───────────────────────────────────────────
        lengths = load_record_lengths(str(self.h5_root), self.table_csv, all_filepaths)
        self._build_index(lengths, min_data_length, seg_mode,
                          chunkify_train, chunk_length_train,
                          stride_fraction_train, stride_fraction_valtest)

        # ── label matrix (pre-extracted; avoids per-item DataFrame lookups) ─
        self._build_label_matrix()

        # ── optional H5-read skip when the MoRyECG preprocessing cache hits ─
        self._skip_h5_if_cached = os.environ.get("MORYECG_SKIP_H5_IF_CACHED") == "1"
        self._pp_cache_root = os.environ.get("MORYECG_CACHE")
        self._cache_path_fn = None
        if self._pp_cache_root:
            check_cache_stamp(self._pp_cache_root, self.fs_model, lead_order,
                              seg_mode, self.split)
        if self._skip_h5_if_cached and self._pp_cache_root:
            try:
                from src.encoders.moryecg import cache_path as _cp

                self._cache_path_fn = _cp
            except Exception:
                self._skip_h5_if_cached = False

    # ------------------------------------------------------------------
    # index construction
    # ------------------------------------------------------------------
    def _build_index(self, lengths_df, min_data_length, seg_mode,
                     chunkify_train, chunk_length_train,
                     stride_fraction_train, stride_fraction_valtest):
        """Build ``(row, seg, start, end)`` windows, mirroring TimeSeriesDataset."""
        by_path = {}
        for fp, seg, length, fs in lengths_df.itertuples(index=False):
            by_path.setdefault(fp, []).append((int(seg), int(length), int(fs)))
        for segs in by_path.values():
            segs.sort()

        if "fs" in self.table.columns:
            table_fs = pd.to_numeric(self.table["fs"], errors="coerce").to_numpy(dtype=np.float64)
            table_fs = np.nan_to_num(table_fs, nan=0.0)
        else:
            table_fs = np.zeros(len(self.table), dtype=np.float64)

        row_idx, seg_idx, start_idx, end_idx, fs_list, out_sizes = [], [], [], [], [], []
        # CSR-style candidate segments, used only by the random-crop train entries
        cand_offsets, cand_segs, cand_lens = [0], [], []
        n_dropped_short, n_dropped_filter, n_missing = 0, 0, 0
        random_crop_entries = self.is_train and not chunkify_train

        for i, fp in enumerate(self.table["filepath"].to_numpy()):
            segs = by_path.get(fp)
            if not segs:
                n_missing += 1
                continue

            fs = self.fs_data_override
            if not fs:
                fs = table_fs[i] if table_fs[i] else 0
            if not fs:
                fs = next((s[2] for s in segs if s[2]), 0)
            if not fs:
                n_missing += 1
                continue
            fs = float(fs)

            total_length = sum(s[1] for s in segs)
            if min_data_length and total_length < int(min_data_length):
                n_dropped_filter += 1
                continue

            output_size = int(round(self.input_size * fs))
            use_segs = segs if seg_mode == "all" else segs[:1]

            eligible = [(seg, length) for seg, length, _ in use_segs if length >= output_size]
            if not eligible:
                n_dropped_short += 1
                continue

            if random_crop_entries:
                # chunk_length == 0 in the original: exactly ONE training window per
                # record per epoch, cropped at random. The H5 store splits long
                # records into segments, so the segment is drawn here too — weighted
                # by the number of valid start offsets it contains, which makes the
                # crop uniform over the record's timeline just as it is in the
                # original's single contiguous memmap.
                row_idx.append(i)
                seg_idx.append(eligible[0][0])
                start_idx.append(-1)          # sentinel: resolved in __getitem__
                end_idx.append(-1)
                fs_list.append(fs)
                out_sizes.append(output_size)
                for seg, length in eligible:
                    cand_segs.append(seg)
                    cand_lens.append(length)
                cand_offsets.append(len(cand_segs))
                continue

            windows = []
            for seg, length in eligible:
                if self.is_train:
                    chunk = max(int(chunk_length_train * output_size), 1)
                    stride = max(int(stride_fraction_train * output_size), 1)
                else:
                    chunk = output_size
                    stride = max(int(stride_fraction_valtest * output_size), 1)
                for s in range(0, length, stride):
                    e = min(s + chunk, length)
                    if e - s < output_size:
                        break  # original deletes this window and every later one
                    windows.append((seg, s, e))

            if not windows:
                n_dropped_short += 1
                continue

            for seg, s, e in windows:
                row_idx.append(i)
                seg_idx.append(seg)
                start_idx.append(s)
                end_idx.append(e)
                fs_list.append(fs)
                out_sizes.append(output_size)

        self._row_idx = np.asarray(row_idx, dtype=np.int64)
        self._seg_idx = np.asarray(seg_idx, dtype=np.int64)
        self._start_idx = np.asarray(start_idx, dtype=np.int64)
        self._end_idx = np.asarray(end_idx, dtype=np.int64)
        self._fs = np.asarray(fs_list, dtype=np.float64)
        self._output_size = np.asarray(out_sizes, dtype=np.int64)
        self._random_crop_entries = random_crop_entries
        self._cand_offsets = np.asarray(cand_offsets, dtype=np.int64)
        self._cand_segs = np.asarray(cand_segs, dtype=np.int64)
        self._cand_lens = np.asarray(cand_lens, dtype=np.int64)

        n_records = len(np.unique(self._row_idx)) if len(self._row_idx) else 0
        logger.info(
            "  [%s] %s windows over %s records (input_size=%.4gs -> %d samples @ %gHz)"
            "%s%s%s",
            self.split, f"{len(self._row_idx):,}", f"{n_records:,}",
            self.input_size, self.model_seq_len, self.fs_model,
            f" | dropped {n_dropped_short:,} too-short" if n_dropped_short else "",
            f" | dropped {n_dropped_filter:,} by min_data_length" if n_dropped_filter else "",
            f" | {n_missing:,} unreadable" if n_missing else "")

        if len(self._row_idx) == 0:
            reasons = []
            if n_dropped_short:
                reasons.append(f"{n_dropped_short:,} records shorter than the "
                               f"{self.input_size}s window")
            if n_dropped_filter:
                reasons.append(f"{n_dropped_filter:,} dropped by "
                               f"min_data_length={min_data_length}")
            if n_missing:
                reasons.append(f"{n_missing:,} unreadable or missing from the length cache")
            raise RuntimeError(
                f"[{self.split}] no windows produced from {len(self.table):,} records"
                + (" — " + "; ".join(reasons) if reasons else ""))

    def _build_label_matrix(self):
        """Materialise labels as a float32 (n_records, n_labels) array."""
        if not self.has_labels:
            self._labels = np.zeros((len(self.table), 1), dtype=np.float32)
            return

        missing = [c for c in self.label_cols if c not in self.table.columns]
        if missing:
            raise KeyError(
                f"label columns absent from the joined table: {missing[:10]}"
                f"{' ...' if len(missing) > 10 else ''} — check label_csv/label_cols "
                f"in the task config.")

        def numeric(cols):
            frame = self.table.reindex(columns=list(cols))
            return frame.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)

        def binary(cols):
            frame = self.table.reindex(columns=list(cols))
            out = np.zeros((len(frame), len(cols)), dtype=np.float32)
            for j, col in enumerate(cols):
                series = frame[col]
                if series.dtype == bool:
                    out[:, j] = series.to_numpy(dtype=np.float32)
                    continue
                as_num = pd.to_numeric(series, errors="coerce")
                text = series.astype(str).str.strip().str.lower()
                truthy = text.isin(["true", "1", "1.0", "yes"])
                out[:, j] = np.where(as_num.notna(), as_num.fillna(0).to_numpy(), truthy.to_numpy()).astype(np.float32)
            return out

        def binary_with_nan(cols):
            out = binary(cols)
            frame = self.table.reindex(columns=list(cols))
            missing = frame.isna().to_numpy()
            out[missing] = np.nan
            return out

        if self.task_type == "classification_and_regression":
            cls_part = binary_with_nan(self.cls_cols)
            reg_part = numeric(self.reg_cols)
            if self.target_mean is not None and self.target_std is not None:
                reg_part = (reg_part - self.target_mean) / (self.target_std + 1e-8)
            self._labels = np.concatenate([cls_part, reg_part], axis=1)
        elif self.task_type == "regression":
            labels = numeric(self.label_cols)
            if self.target_mean is not None and self.target_std is not None:
                labels = (labels - self.target_mean) / (self.target_std + 1e-8)
            self._labels = labels
        elif self.task_type == "multi-label-binary":
            self._labels = binary_with_nan(self.label_cols)
        else:
            self._labels = np.nan_to_num(binary(self.label_cols), nan=0.0)

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self._row_idx)

    def _resolve_window(self, idx, output_size):
        """Pick the (segment, start, end) this item reads.

        Random-crop train entries carry a sentinel start: the segment is drawn
        with probability proportional to its number of valid offsets, then the
        offset itself is drawn uniformly — together equivalent to the original's
        uniform crop over one contiguous record.
        """
        if not self._random_crop_entries:
            return int(self._seg_idx[idx]), int(self._start_idx[idx]), int(self._end_idx[idx])

        lo, hi = int(self._cand_offsets[idx]), int(self._cand_offsets[idx + 1])
        segs = self._cand_segs[lo:hi]
        lens = self._cand_lens[lo:hi]
        if len(segs) == 1:
            return int(segs[0]), 0, int(lens[0])
        weights = lens - output_size + 1
        draw = random.randrange(int(weights.sum()))
        # side="right": draw == cumsum[i] must fall into segment i+1, otherwise
        # the first segment would be over-sampled by one offset
        pick = min(int(np.searchsorted(np.cumsum(weights), draw, side="right")), len(segs) - 1)
        return int(segs[pick]), 0, int(lens[pick])

    def _read_window(self, filepath, seg, start, end, output_size, fs):
        """Read one native-rate crop and bring it to (n_leads, model_seq_len)."""
        h5_path = self.h5_root / filepath
        timesteps = end - start

        # Original TimeSeriesDataset.__getitem__: random crop for train, centre
        # crop otherwise. random.randint is inclusive, and the original excludes
        # the very last offset — reproduced here verbatim.
        if self.is_train:
            start_rel = 0 if timesteps == output_size else random.randint(0, timesteps - output_size - 1)
        else:
            start_rel = (timesteps - output_size) // 2

        s = int(start + start_rel)
        e = int(s + output_size)

        with h5py.File(h5_path, "r") as f:
            sig = f[f"ECG/segments/{seg}/signal"][:, s:e].astype(np.float32)

        if self.lead_perm is not None:
            sig = sig[self.lead_perm]

        sig = resample_signal(sig, fs, self.fs_model)
        sig = fit_length(sig, self.model_seq_len)

        if self.normalize and self.mean is not None and self.std is not None:
            sig = (sig - np.asarray(self.mean)[:, None]) / (np.asarray(self.std)[:, None] + 1e-8)

        return np.nan_to_num(sig, nan=0.0)

    def __getitem__(self, idx):
        row = int(self._row_idx[idx])
        output_size = int(self._output_size[idx])
        seg, start, end = self._resolve_window(idx, output_size)
        filepath = self.table["filepath"].iat[row]
        h5_path = self.h5_root / filepath

        skip_read = (self._skip_h5_if_cached and self._cache_path_fn is not None
                     and self._cache_path_fn(self._pp_cache_root, str(h5_path), seg).exists())
        if skip_read:
            sig = np.zeros((len(self.target_leads or []) or 12, self.model_seq_len),
                           dtype=np.float32)
        else:
            sig = self._read_window(filepath, seg, start, end,
                                    output_size, float(self._fs[idx]))

        return {
            "signal": torch.from_numpy(np.ascontiguousarray(sig)),
            "label": torch.from_numpy(self._labels[row].copy()),
            "fs": float(self._fs[idx]),
            "idx": idx,
            "ecg_id": row,                       # record-level id -> prediction aggregation
            "ecg_filepath": str(h5_path),        # MoRyECG preprocessing-cache key
            "ecg_seg_idx": seg,
        }

    # ------------------------------------------------------------------
    def get_id_mapping(self):
        """Record id per window — the equivalent of ``TimeSeriesDataset.get_id_mapping``."""
        return self._row_idx

    def compute_stats(self, max_items=5000):
        """Per-lead mean/std over the split (only needed when normalize=True)."""
        sums = sq_sums = None
        count = 0
        for i in range(min(len(self), max_items)):
            sig = self[i]["signal"].numpy()
            if sums is None:
                sums = np.zeros(sig.shape[0], dtype=np.float64)
                sq_sums = np.zeros(sig.shape[0], dtype=np.float64)
            sums += sig.mean(axis=1)
            sq_sums += (sig ** 2).mean(axis=1)
            count += 1
        mean = (sums / count).astype(np.float32)
        std = np.sqrt(sq_sums / count - mean ** 2).astype(np.float32)
        return mean, std


def build_dataset(cfg: dict, split: str = "train") -> H5ECGDataset:
    """Instantiate :class:`H5ECGDataset` from a task ``data`` config section."""
    return H5ECGDataset(
        h5_root=cfg["h5_root"],
        table_csv=cfg["table_csv"],
        label_csv=cfg.get("label_csv"),
        label_cols=cfg.get("label_cols"),
        split=split,
        input_size=cfg["input_size"],
        fs_model=cfg["fs_model"],
        fs_data=cfg.get("fs_data"),
        chunkify_train=bool(cfg.get("chunkify_train", False)),
        chunk_length_train=float(cfg.get("chunk_length_train", 1.0)),
        stride_fraction_train=float(cfg.get("stride_fraction_train", 1.0)),
        stride_fraction_valtest=float(cfg.get("stride_fraction_valtest", 1.0)),
        min_data_length=cfg.get("min_data_length"),
        seg_mode=cfg.get("seg_mode", "all"),
        lead_order=cfg.get("lead_order", "standard"),
        normalize=bool(cfg.get("normalize", False)),
        mean=cfg.get("mean"),
        std=cfg.get("std"),
        fold_col=cfg.get("fold_col"),
        fold_ids=cfg.get(f"{split}_folds"),
        task_type=cfg.get("task_type", "binary"),
        target_mean=cfg.get("target_mean"),
        target_std=cfg.get("target_std"),
        cls_cols=cfg.get("cls_cols"),
        reg_cols=cfg.get("reg_cols"),
    )


def build_dataloaders(cfg, split="train"):
    """Single-process DataLoader (DDP path lives in ``run.py``)."""
    from torch.utils.data import DataLoader

    ds = build_dataset(cfg, split)
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
