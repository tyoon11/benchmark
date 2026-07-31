"""
Empirical bootstrap (n=1000) — single-model 95% CI
===================================================
saveed preds.npy/targets.npy  primary metric's
 bootstrap 95% CI compute.

Primary metric:
  -  (binary / multi-label-binary): macro-averaged AUROC
  - time (regression):                  z-normalized MAE
    (preds/targets already z-normalized  justorder MAE = z-norm MAE)

CI   (paper's empirical_bootstrap reproduction):
  diffs   = bootstrap_scores - point_estimate
  CI_low  = point + percentile(diffs, 2.5)
  CI_high = point + percentile(diffs, 97.5)

Usage:
  python scripts/bootstrap_ci.py --result_dir <DIR>
  python scripts/bootstrap_ci.py --root <RESULT_ROOT> [--n_iters 1000]

output:
  <result_dir>/bootstrap.json   point/ci_low/ci_high/n_iters
  <root>/bootstrap_summary.csv  all summary (model, task, mode, point, ci_low, ci_high)
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, mean_absolute_error
from sklearn.utils import resample

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

logger = logging.getLogger("bootstrap_ci")


# ──────────────────────────────────────────────────────────────────
# Primary metric functions (sample-axis bootstrappable)
# ──────────────────────────────────────────────────────────────────
def macro_auroc(targets, preds):
    """Multi-label macro AUROC, matching the original ``multiclass_roc_curve``.

    Mean of the per-class AUCs over **all** classes, with classes that cannot be
    scored in this bootstrap resample counted as 0.5 — the same definition the
    trainer writes as ``auroc_macro``. Using the skip-unscoreable variant here
    would make the CIs inconsistent with the point estimates.
    """
    from src.metrics import multiclass_roc_curve

    _, _, roc = multiclass_roc_curve(targets, preds)
    return float(roc["macro"])


def macro_auroc_skipnan(targets, preds):
    """Mean over scoreable classes only (diagnostic; see ``auroc_macro_skipnan``)."""
    n_classes = targets.shape[1]
    aucs = []
    for i in range(n_classes):
        col = targets[:, i]
        valid = ~np.isnan(col)
        pos = np.nansum(col)
        if 0 < pos < valid.sum():
            try:
                aucs.append(roc_auc_score(col[valid], preds[valid, i]))
            except ValueError:
                pass
    return float(np.mean(aucs)) if aucs else float("nan")


def znorm_mae(targets, preds):
    """multivariate time: per-target MAE's macro . NaN masking."""
    n_targets = targets.shape[1]
    maes = []
    for i in range(n_targets):
        valid = ~np.isnan(targets[:, i])
        if valid.sum() < 2:
            continue
        maes.append(mean_absolute_error(targets[valid, i], preds[valid, i]))
    return float(np.mean(maes)) if maes else float("nan")


def get_metric_fn(task_type: str):
    if task_type == "regression":
        return znorm_mae, "znorm_mae"
    return macro_auroc, "macro_auroc"


# ──────────────────────────────────────────────────────────────────
# Empirical bootstrap (paper's clinical_ts.utils.bootstrap_utils reproduction)
# ──────────────────────────────────────────────────────────────────
def empirical_bootstrap(targets, preds, score_fn, n_iters=1000, alpha=0.95, seed=0):
    """
     bootstrap (resample with replacement).

    Returns: dict(point, ci_low, ci_high, scores)
    """
    n = len(targets)
    point = score_fn(targets, preds)

    rng = np.random.default_rng(seed)
    scores = np.empty(n_iters, dtype=np.float64)
    for it in range(n_iters):
        idx = rng.integers(0, n, size=n)
        scores[it] = score_fn(targets[idx], preds[idx])

    diffs = scores - point
    lo = point + np.nanpercentile(diffs, (1 - alpha) / 2 * 100)
    hi = point + np.nanpercentile(diffs, (alpha + (1 - alpha) / 2) * 100)
    return {
        "point":   float(point),
        "ci_low":  float(lo),
        "ci_high": float(hi),
        "n_iters": int(n_iters),
        "alpha":   float(alpha),
    }


# ──────────────────────────────────────────────────────────────────
# directory handling
# ──────────────────────────────────────────────────────────────────
def process(result_dir: Path, n_iters: int, seed: int, force: bool):
    out_path = result_dir / "bootstrap.json"
    if out_path.exists() and not force:
        return json.loads(out_path.read_text())

    preds_p = result_dir / "preds.npy"
    targets_p = result_dir / "targets.npy"
    meta_p = result_dir / "preds_meta.json"
    if not (preds_p.exists() and targets_p.exists() and meta_p.exists()):
        logger.warning(f"[SKIP] missing preds/targets/meta in {result_dir}")
        return None

    meta = json.loads(meta_p.read_text())
    preds = np.load(preds_p)
    targets = np.load(targets_p)
    if preds.ndim == 1: preds = preds[:, None]
    if targets.ndim == 1: targets = targets[:, None]

    fn, metric_name = get_metric_fn(meta.get("task_type", "binary"))
    res = empirical_bootstrap(targets, preds, fn, n_iters=n_iters, seed=seed)
    res.update({
        "metric": metric_name,
        "model":  meta["model"],
        "task":   meta["task"],
        "mode":   meta["mode"],
        "task_type": meta["task_type"],
        "n_test": meta["n_test"],
    })
    out_path.write_text(json.dumps(res, indent=2))
    logger.info(f"  {result_dir.name}: {metric_name}={res['point']:.4f} "
                f"[{res['ci_low']:.4f}, {res['ci_high']:.4f}]")
    return res


def _process_worker(args):
    """Pickle-safe wrapper for ProcessPoolExecutor."""
    result_dir, n_iters, seed, force = args
    try:
        return result_dir.name, process(result_dir, n_iters, seed, force), None
    except Exception as e:
        import traceback
        return result_dir.name, None, traceback.format_exc()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--result_dir", type=str, default=None)
    p.add_argument("--root", type=str, default=None)
    p.add_argument("--filter", type=str, default=None)
    p.add_argument("--n_iters", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--force", action="store_true")
    p.add_argument("--workers", type=int, default=1,
                   help="parallel worker (default 1=).  .")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.result_dir:
        process(Path(args.result_dir), args.n_iters, args.seed, args.force)
        return

    if not args.root:
        p.error("--result_dir or --root  of one required")

    root = Path(args.root)
    dirs = sorted(d for d in root.iterdir() if d.is_dir() and (d / "preds.npy").exists())
    if args.filter:
        dirs = [d for d in dirs if args.filter in d.name]
    logger.info(f"Bootstrap dirs: {len(dirs)} (n_iters={args.n_iters}, workers={args.workers})")

    rows = []
    if args.workers <= 1:
        for d in dirs:
            try:
                r = process(d, args.n_iters, args.seed, args.force)
                if r: rows.append(r)
            except Exception as e:
                logger.error(f"[FAIL] {d.name}: {e}", exc_info=True)
    else:
        tasks = [(d, args.n_iters, args.seed, args.force) for d in dirs]
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for name, r, err in ex.map(_process_worker, tasks, chunksize=1):
                if err:
                    logger.error(f"[FAIL] {name}: {err.splitlines()[-1]}")
                elif r:
                    rows.append(r)

    if rows:
        df = pd.DataFrame(rows)[["model", "task", "mode", "task_type", "metric",
                                  "point", "ci_low", "ci_high", "n_test", "n_iters"]]
        df = df.sort_values(["task", "mode", "model"]).reset_index(drop=True)
        out_csv = root / "bootstrap_summary.csv"
        df.to_csv(out_csv, index=False)
        logger.info(f"Summary written → {out_csv} ({len(df)} rows)")


if __name__ == "__main__":
    main()
