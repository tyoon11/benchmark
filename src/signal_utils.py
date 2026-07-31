"""
Signal utilities
================
Resampling and record-length bookkeeping, matched to the original
``ecg-fm-benchmarking`` pipeline.

Resampling
----------
The original applies ``resampy.resample(data, fs_source, fs_target, axis=0)`` as
a torchvision transform *after* the crop has been taken (see
``clinical_ts/data/time_series_dataset_transforms.py::Resample``). resampy is a
band-limited (windowed-sinc, ``kaiser_best``) resampler. Naive linear
interpolation — which this benchmark used previously — aliases badly when
decimating (500 Hz -> 100 Hz for HuBERT-ECG is a 5x decimation), so
:func:`resample_signal` reproduces the resampy path and only falls back to
``scipy.signal.resample_poly`` (Kaiser-windowed polyphase, also band-limited) if
resampy is unavailable.

Length cache
------------
Building the chunk index the way the original does needs the length of every
record up front. Reading 30k+ H5 headers over network storage on every run is
too slow, so lengths are scanned once and cached under ``labels/_cache/lengths``.
The cache is keyed by the table CSV path and invalidated by its mtime/size.
"""

from __future__ import annotations

import hashlib
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:  # pragma: no cover - import-time capability probe
    import resampy

    _HAS_RESAMPY = True
except ImportError:  # pragma: no cover
    resampy = None
    _HAS_RESAMPY = False
    logger.warning(
        "resampy is not installed — falling back to scipy.signal.resample_poly. "
        "Install resampy to match the original benchmark bit-for-bit: pip install resampy")


def resample_signal(sig: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    """Band-limited resample of ``(n_leads, n_samples)`` along the time axis.

    Mirrors the original ``Resample`` transform (resampy, ``kaiser_best``).
    """
    if sig.size == 0 or fs_in is None or fs_out is None:
        return sig
    if int(round(fs_in)) == int(round(fs_out)):
        return sig

    sig = np.ascontiguousarray(sig, dtype=np.float32)

    if _HAS_RESAMPY:
        # The original resamples along axis 0 of a (time, channel) array; our
        # layout is (channel, time), so axis=-1 is the equivalent.
        out = resampy.resample(sig, float(fs_in), float(fs_out), axis=-1)
        return np.ascontiguousarray(out, dtype=np.float32)

    from scipy.signal import resample_poly

    ratio = Fraction(float(fs_out) / float(fs_in)).limit_denominator(1000)
    out = resample_poly(sig, ratio.numerator, ratio.denominator,
                        axis=-1, window=("kaiser", 5.0))
    return np.ascontiguousarray(out, dtype=np.float32)


def expected_resampled_length(n_in: int, fs_in: float, fs_out: float) -> int:
    """Length resampy produces for ``n_in`` samples (ceil of the rate ratio)."""
    if int(round(fs_in)) == int(round(fs_out)):
        return int(n_in)
    return int(np.ceil(n_in * float(fs_out) / float(fs_in)))


def fit_length(sig: np.ndarray, length: int) -> np.ndarray:
    """Force ``(n_leads, length)``.

    Resamplers are off by a sample or two versus ``round(n * fs_out / fs_in)``;
    this trims/edge-pads that residue. It is *not* a substitute for the
    record-level length filtering done in :mod:`src.dataset` — records genuinely
    shorter than one window are dropped there, exactly as the original does.
    """
    cur = sig.shape[-1]
    if cur == length:
        return sig
    if cur > length:
        return sig[..., :length]
    pad = np.zeros((sig.shape[0], length - cur), dtype=sig.dtype)
    if cur > 0:
        pad[:] = sig[:, -1:]  # edge-pad rather than zero-pad for sub-sample residue
    return np.concatenate([sig, pad], axis=-1)


# ---------------------------------------------------------------------------
# Record length cache
# ---------------------------------------------------------------------------
def _cache_key(table_csv: str) -> str:
    p = Path(table_csv)
    try:
        stat = p.stat()
        stamp = f"{stat.st_mtime_ns}:{stat.st_size}"
    except OSError:
        stamp = "missing"
    digest = hashlib.sha1(f"{p.resolve()}|{stamp}".encode()).hexdigest()[:16]
    return f"{p.stem}_{digest}.csv"


def _probe_one(args):
    h5_root, filepath = args
    import h5py

    path = os.path.join(h5_root, filepath)
    rows = []
    try:
        with h5py.File(path, "r") as f:
            fs = int(f["ECG/metadata"].attrs.get("fs", 0)) or None
            segs = f["ECG/segments"]
            n_segs = int(segs.attrs.get("seg_len", len(segs)))
            for s in range(n_segs):
                try:
                    shape = segs[str(s)]["signal"].shape
                except KeyError:
                    continue
                rows.append((filepath, s, int(shape[-1]), fs or 0))
    except Exception as exc:  # unreadable record -> length 0, filtered downstream
        logger.debug("length probe failed for %s: %s", path, exc)
        rows.append((filepath, 0, 0, 0))
    return rows


def load_record_lengths(h5_root: str, table_csv: str, filepaths,
                        cache_dir: str = None, workers: int = 16) -> pd.DataFrame:
    """Return a DataFrame ``[filepath, seg_idx, length, fs]`` for every segment.

    Scans the H5 headers once and caches the result. ``length`` is in samples at
    the record's *native* sampling rate.
    """
    cache_dir = Path(cache_dir or (Path(__file__).resolve().parent.parent / "labels" / "_cache" / "lengths"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / _cache_key(table_csv)

    if cache_file.exists():
        cached = pd.read_csv(cache_file)
        if set(filepaths).issubset(set(cached["filepath"])):
            return cached
        logger.info("Length cache %s is incomplete — rescanning", cache_file)

    unique_paths = list(dict.fromkeys(filepaths))
    logger.info("Scanning record lengths for %s (%d records) — cached to %s",
                Path(table_csv).name, len(unique_paths), cache_file)

    rows = []
    try:
        from tqdm.auto import tqdm

        iterator = tqdm(unique_paths, desc="scan lengths", leave=False)
    except ImportError:  # pragma: no cover
        iterator = unique_paths

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(_probe_one, ((h5_root, fp) for fp in iterator)):
            rows.extend(result)

    df = pd.DataFrame(rows, columns=["filepath", "seg_idx", "length", "fs"])
    tmp = cache_file.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, cache_file)
    return df
