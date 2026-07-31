#!/usr/bin/env python
"""
Precompute MoRyECG R-peak / beat / STFT cache for every H5-based task.

Each (h5_file, seg_idx) is preprocessed once with the same pipeline as
`MoRyECGEncoder` live path (`preprocess_signal` from src.encoders.moryecg)
and saved as a .npz sidecar under `MORYECG_CACHE`. The encoder skips live
preprocessing whenever a cache file is present.

Usage:
    MORYECG_CACHE=/path/to/cache_dir \
    python scripts/precompute_moryecg_cache.py \
        [--tasks ptb ningbo ...]   # default: all H5 tasks
        [--workers 64]              # default: nproc
        [--force]                   # overwrite existing cache files
"""
from __future__ import annotations

import os
# Pin BLAS / OpenMP / NumExpr to 1 thread per worker so that 128 worker
# processes do not multiply into 1000+ threads and crush the scheduler.
# Must run before numpy/scipy import.
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "BLIS_NUM_THREADS", "TBB_NUM_THREADS"):
    os.environ.setdefault(_k, "1")

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

BENCH_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH_DIR))

from src.encoders.moryecg import (  # noqa: E402
    MODEL_FS,
    PREPROC_VERSION,
    _import_pretrain_modules,
    _resolve_repo_root,
    cache_path,
    preprocess_signal,
    save_cache,
)

# Worker globals (set in init_worker)
_WORKER_CACHE_ROOT: Path | None = None
_WORKER_MODS: dict | None = None
_WORKER_FORCE: bool = False


def expand_env(s: str) -> str:
    return os.path.expandvars(s)


def collect_numpy_segments(data_cfg: dict) -> list[tuple[str, int]]:
    """Return [(waveform_npy_abs_path, row_idx), ...] for numpy-loader tasks.

    EchoNext splits share a metadata CSV but each split has its own .npy file;
    each .npy file is keyed by row index (matches the dataset's table_idx).
    """
    waveforms = data_cfg.get("waveforms")
    if not waveforms:
        return []
    metadata_csv = data_cfg.get("metadata_csv")
    split_col = data_cfg.get("split_col", "split")
    if not metadata_csv:
        return []
    metadata_csv = Path(expand_env(metadata_csv))
    if not metadata_csv.exists():
        print(f"[skip] metadata_csv missing: {metadata_csv}", file=sys.stderr)
        return []
    df = pd.read_csv(metadata_csv, low_memory=False)
    out: list[tuple[str, int]] = []
    for split, npy_path in waveforms.items():
        npy_abs = Path(expand_env(npy_path))
        if not npy_abs.exists():
            print(f"[skip] {split} .npy missing: {npy_abs}", file=sys.stderr)
            continue
        # Count rows per split via metadata
        n_split = int((df[split_col] == split).sum())
        for i in range(n_split):
            out.append((str(npy_abs), i))
    return out


def collect_task_segments(task_yaml: Path) -> list[tuple[Path, int]]:
    """Return [(absolute_file_path, seg_idx), ...] for one task. Handles both
    H5 and numpy (echonext) loaders."""
    with open(task_yaml) as f:
        cfg = yaml.safe_load(f)
    data = cfg.get("data", {})
    loader = str(data.get("loader_type", "")).lower()
    if loader == "echonext_numpy":
        return collect_numpy_segments(data)

    h5_root = data.get("h5_root")
    table_csv = data.get("table_csv")
    if not h5_root or not table_csv:
        return []
    h5_root = Path(expand_env(h5_root))
    table_csv = Path(expand_env(table_csv))
    if not table_csv.exists():
        print(f"[skip] table_csv missing: {table_csv}", file=sys.stderr)
        return []

    df = pd.read_csv(table_csv)
    if "filepath" not in df.columns:
        print(f"[skip] no 'filepath' col: {table_csv}", file=sys.stderr)
        return []

    # The dataset defaults to seg_mode="all" (every H5 segment of a record), so
    # the cache must cover them all — 'seg_idx' no longer exists in task configs.
    seg_mode = str(cfg.get("data", {}).get("seg_mode", "all")).lower()
    filepaths = df["filepath"].tolist()
    if seg_mode != "all":
        return [(h5_root / fp, 0) for fp in filepaths]

    # Reuse the shared record-length cache: it enumerates every segment with a
    # thread pool and persists the result under labels/_cache/lengths, so this is
    # a fast lookup after the first pass. Opening 345k H5 files serially here
    # took hours.
    from src.signal_utils import load_record_lengths

    lengths = load_record_lengths(str(h5_root), str(table_csv), filepaths)
    return [(h5_root / fp, int(seg))
            for fp, seg in lengths[["filepath", "seg_idx"]].itertuples(index=False)]


def init_worker(cache_root: str, repo_root: str, force: bool) -> None:
    global _WORKER_CACHE_ROOT, _WORKER_MODS, _WORKER_FORCE
    # Pin torch to 1 thread per worker (env vars handle BLAS/MKL)
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    _WORKER_CACHE_ROOT = Path(cache_root)
    _WORKER_MODS = _import_pretrain_modules(_resolve_repo_root(repo_root))
    _WORKER_FORCE = force


def _resample_to_model_fs(sig: np.ndarray, fs: int) -> np.ndarray:
    if fs == MODEL_FS:
        return sig
    from scipy.signal import resample
    n_leads, n_samples = sig.shape
    target_len = int(round(n_samples * MODEL_FS / fs))
    return resample(sig, target_len, axis=1).astype(np.float32)


# Cache mmap'd .npy handles per worker so we don't reopen 100k× for echonext
_WORKER_NPY_CACHE: dict = {}


def _load_signal(file_path: Path, seg_idx: int) -> np.ndarray:
    """Load a 12-lead signal at MODEL_FS=500Hz from either H5 or numpy file."""
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        # EchoNext layout: (N, 1, T, C) NHWC float64 @ 250Hz
        arr = _WORKER_NPY_CACHE.get(str(file_path))
        if arr is None:
            arr = np.load(str(file_path), mmap_mode="r")
            _WORKER_NPY_CACHE[str(file_path)] = arr
        rec = np.asarray(arr[seg_idx]).astype(np.float32)
        # (1, T, C) → (C, T)
        if rec.ndim == 3 and rec.shape[0] == 1:
            rec = rec[0].transpose(1, 0)
        elif rec.ndim == 2 and rec.shape[0] != 12:
            # already (T, C)
            rec = rec.transpose(1, 0)
        # EchoNext source_fs = 250 Hz
        return _resample_to_model_fs(rec, 250)
    # default: H5
    with h5py.File(file_path, "r") as f:
        fs = int(f["ECG/metadata"].attrs.get("fs", 500))
        sig = f[f"ECG/segments/{seg_idx}/signal"][()].astype(np.float32)
    return _resample_to_model_fs(sig, fs)


def process_one(args: tuple[str, int]) -> tuple[int, str | None]:
    """Compute + save cache for one (file_path, seg_idx). Returns (status, err)."""
    file_path_str, seg_idx = args
    file_path = Path(file_path_str)
    cp = cache_path(str(_WORKER_CACHE_ROOT), str(file_path), seg_idx)
    if cp.exists() and not _WORKER_FORCE:
        return 1, None  # already cached
    try:
        sig = _load_signal(file_path, seg_idx)
        sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
        bundle = preprocess_signal(sig, _WORKER_MODS)
        save_cache(cp, bundle)
        return 2, None  # newly cached
    except Exception as e:
        return 0, f"{file_path}::{seg_idx}: {e}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache_root", default=os.environ.get("MORYECG_CACHE"),
                    help="Cache root (default: $MORYECG_CACHE)")
    ap.add_argument("--repo_root", default=os.environ.get(
        "MORYECG_REPO", "/home/irteam/local-node-d/tykim/MoryECG"))
    ap.add_argument("--tasks", nargs="*", default=None,
                    help="Task yaml stems (default: all H5 tasks)")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    if not args.cache_root:
        print("ERROR: MORYECG_CACHE not set and --cache_root not given", file=sys.stderr)
        return 2
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    # Stamp the window this cache is valid for. The cache key is only
    # (filepath, seg_idx), so without this a cache built for a different
    # window/lead order would look like a hit at training time.
    import json

    from src.dataset import CACHE_STAMP_FILE, cache_stamp
    from src.encoders.moryecg import MoRyECGEncoder

    stamp = cache_stamp(MODEL_FS, MoRyECGEncoder.lead_order, seg_mode="all")
    (cache_root / CACHE_STAMP_FILE).write_text(json.dumps(stamp, indent=2))
    print(f"[info] cache stamp  = {stamp}")
    print(f"[info] cache_root = {cache_root}")
    print(f"[info] preproc_version = {PREPROC_VERSION}  model_fs = {MODEL_FS}")
    print(f"[info] workers = {args.workers}")

    # Discover tasks
    task_dir = BENCH_DIR / "configs" / "tasks"
    if args.tasks:
        yamls = [task_dir / f"{t}.yaml" for t in args.tasks]
    else:
        yamls = sorted(task_dir.glob("*.yaml"))

    # Collect all (h5_path, seg_idx) tuples and dedupe across tasks
    pairs: set[tuple[str, int]] = set()
    per_task_counts: list[tuple[str, int]] = []
    for y in yamls:
        try:
            segs = collect_task_segments(y)
        except Exception as e:
            print(f"[warn] failed to parse {y}: {e}", file=sys.stderr)
            continue
        per_task_counts.append((y.stem, len(segs)))
        for h5_path, s in segs:
            pairs.add((str(h5_path), int(s)))

    print(f"[info] per-task segment counts:")
    for name, n in per_task_counts:
        print(f"    {name:30s} {n:>8d}")
    print(f"[info] unique segments across all tasks: {len(pairs)}")

    work = sorted(pairs)
    t0 = time.time()
    counts = {0: 0, 1: 0, 2: 0}  # error, skipped, new
    errors: list[str] = []

    ctx = mp.get_context("forkserver")
    with ctx.Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(str(cache_root), args.repo_root, args.force),
    ) as pool:
        for status, err in tqdm(
            pool.imap_unordered(process_one, work, chunksize=8),
            total=len(work),
            desc="cache",
        ):
            counts[status] += 1
            if err:
                errors.append(err)

    dt = time.time() - t0
    print(f"\n[done] {dt:.1f}s  "
          f"new={counts[2]}  skipped={counts[1]}  errors={counts[0]}")
    if errors[:5]:
        print("first errors:")
        for e in errors[:5]:
            print(f"  {e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
