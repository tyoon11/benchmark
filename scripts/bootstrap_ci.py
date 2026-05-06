"""
Empirical bootstrap (n=1000) — single-model 95% CI
===================================================
저장된 preds.npy/targets.npy 를 읽어서 primary metric의
경험적 부트스트랩 95% CI를 계산합니다.

Primary metric:
  - 분류 (binary / multi-label-binary): macro-averaged AUROC
  - 회귀 (regression):                  z-normalized MAE
    (preds/targets 가 이미 z-normalized 공간이므로 단순 MAE = z-norm MAE)

CI 산출 방식 (paper의 empirical_bootstrap 재현):
  diffs   = bootstrap_scores - point_estimate
  CI_low  = point + percentile(diffs, 2.5)
  CI_high = point + percentile(diffs, 97.5)

사용법:
  python scripts/bootstrap_ci.py --result_dir <DIR>
  python scripts/bootstrap_ci.py --root <RESULT_ROOT> [--n_iters 1000]

출력:
  <result_dir>/bootstrap.json   point/ci_low/ci_high/n_iters
  <root>/bootstrap_summary.csv  전체 요약 (model, task, mode, point, ci_low, ci_high)
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
    """Multi-label macro AUROC. 양성/음성이 모두 존재하는 클래스만 평균."""
    n_classes = targets.shape[1]
    aucs = []
    for i in range(n_classes):
        col = targets[:, i]
        pos = np.nansum(col)
        if 0 < pos < (~np.isnan(col)).sum():
            valid = ~np.isnan(col)
            try:
                aucs.append(roc_auc_score(col[valid], preds[valid, i]))
            except ValueError:
                pass
    return float(np.mean(aucs)) if aucs else float("nan")


def znorm_mae(targets, preds):
    """다변량 회귀: per-target MAE의 macro 평균. NaN 마스킹."""
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
# Empirical bootstrap (paper의 clinical_ts.utils.bootstrap_utils 재현)
# ──────────────────────────────────────────────────────────────────
def empirical_bootstrap(targets, preds, score_fn, n_iters=1000, alpha=0.95, seed=0):
    """
    경험적 부트스트랩 (resample with replacement).

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
# 한 폴더 처리
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
                   help="병렬 worker 수 (기본 1=직렬). 코어수까지 권장.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.result_dir:
        process(Path(args.result_dir), args.n_iters, args.seed, args.force)
        return

    if not args.root:
        p.error("--result_dir 또는 --root 중 하나는 필요")

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
