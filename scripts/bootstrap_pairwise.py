"""
Pairwise empirical bootstrap (n=1000) — 모델 간 유의차 / tied-rank
====================================================================
같은 (task, mode) 안의 두 모델 A, B에 대해
  diff_b = score(A | resample_b) - score(B | resample_b)        (paired bootstrap)
  CI_95 = [point + pct(diffs-diff_pt, 2.5), point + pct(diffs-diff_pt, 97.5)]
  유의차 = 0 ∉ CI_95
  유의차가 없으면 동순위 (tied).

각 (task, mode)에 대해
  - pairwise_diff_<task>_<mode>.csv : (model_a, model_b, diff, ci_low, ci_high, sig)
  - tied_groups_<task>_<mode>.txt   : best_metric 기준 모델 순위 + 동순위 그룹

전체 요약: pairwise_summary.csv  (모든 task·mode 합본)

사용법:
  python scripts/bootstrap_pairwise.py --root <RESULT_ROOT> [--n_iters 1000]

전제:
  - 같은 (task, mode) 의 모델들은 동일한 test split을 쓰므로 N과 정렬이 같다고 가정.
  - ids.npy 가 있으면 정렬 키로 사용해 정합성 보장 (주로 multi-window 케이스).
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.bootstrap_ci import get_metric_fn, macro_auroc, znorm_mae

logger = logging.getLogger("bootstrap_pairwise")


# ──────────────────────────────────────────────────────────────────
# 한 (task, mode) 그룹 처리
# ──────────────────────────────────────────────────────────────────
def load_group(dirs):
    """returns: list of dict(model, preds, targets, ids, task_type)"""
    out = []
    for d in dirs:
        meta = json.loads((d / "preds_meta.json").read_text())
        preds = np.load(d / "preds.npy")
        targets = np.load(d / "targets.npy")
        ids = np.load(d / "ids.npy") if (d / "ids.npy").exists() else np.arange(len(preds))
        if preds.ndim == 1: preds = preds[:, None]
        if targets.ndim == 1: targets = targets[:, None]
        out.append({
            "model": meta["model"], "preds": preds, "targets": targets,
            "ids": ids, "task_type": meta["task_type"],
        })
    return out


def align_by_ids(group):
    """ids.npy 기준으로 모든 모델을 같은 순서로 정렬. 공통 id만 유지."""
    common = None
    for g in group:
        s = set(g["ids"].tolist())
        common = s if common is None else (common & s)
    if not common:
        raise ValueError("no common ids across models")
    common = np.array(sorted(common))
    for g in group:
        order = {int(v): i for i, v in enumerate(g["ids"])}
        idx = np.array([order[int(c)] for c in common], dtype=np.int64)
        g["preds"] = g["preds"][idx]
        g["targets"] = g["targets"][idx]
        g["ids"] = g["ids"][idx]
    return group


def pairwise_bootstrap(group, n_iters=1000, seed=0, alpha=0.95):
    """공유 부트스트랩 인덱스로 paired difference CI."""
    task_type = group[0]["task_type"]
    score_fn, metric_name = get_metric_fn(task_type)
    higher_is_better = (task_type != "regression")

    # 공유된 targets — 모든 모델이 동일해야 함 (sanity check)
    targets = group[0]["targets"]
    for g in group[1:]:
        if not np.allclose(np.nan_to_num(g["targets"]), np.nan_to_num(targets)):
            logger.warning(f"  targets mismatch: {group[0]['model']} vs {g['model']} — proceeding")

    n = len(targets)
    rng = np.random.default_rng(seed)
    boot_idx = rng.integers(0, n, size=(n_iters, n))

    # 모델별 부트스트랩 score 행렬: (n_iters,)
    point_scores = {}
    boot_scores = {}
    for g in group:
        point_scores[g["model"]] = score_fn(g["targets"], g["preds"])
        scores = np.empty(n_iters, dtype=np.float64)
        for it in range(n_iters):
            idx = boot_idx[it]
            scores[it] = score_fn(g["targets"][idx], g["preds"][idx])
        boot_scores[g["model"]] = scores

    # Pairwise diff
    models = [g["model"] for g in group]
    rows = []
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b = models[i], models[j]
            diff_pt = point_scores[a] - point_scores[b]
            diff_boot = boot_scores[a] - boot_scores[b]
            diffs = diff_boot - diff_pt
            lo = diff_pt + np.nanpercentile(diffs, (1 - alpha) / 2 * 100)
            hi = diff_pt + np.nanpercentile(diffs, (alpha + (1 - alpha) / 2) * 100)
            sig = (lo > 0) or (hi < 0)
            rows.append({
                "model_a": a, "model_b": b,
                "score_a": float(point_scores[a]),
                "score_b": float(point_scores[b]),
                "diff": float(diff_pt),
                "ci_low": float(lo),
                "ci_high": float(hi),
                "significant": bool(sig),
            })

    return point_scores, rows, metric_name, higher_is_better


def tied_rank_groups(point_scores, pair_rows, higher_is_better):
    """
    유의차 없는 모델끼리 같은 순위 그룹으로 묶기.

    Equivalence classes: A ~ B if pair (A,B) is NOT significant.
    실제로는 transitive 가 보장 안 되지만 (paper와 동일하게) 단순 union-find로 처리.
    """
    models = sorted(point_scores.keys(),
                    key=lambda m: -point_scores[m] if higher_is_better else point_scores[m])

    # union-find
    parent = {m: m for m in models}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: parent[ra] = rb

    for r in pair_rows:
        if not r["significant"]:
            union(r["model_a"], r["model_b"])

    groups = defaultdict(list)
    for m in models:
        groups[find(m)].append(m)

    # 그룹 순서: 그룹 best score 순
    def grp_key(grp):
        s = max(point_scores[m] for m in grp) if higher_is_better else min(point_scores[m] for m in grp)
        return -s if higher_is_better else s
    ordered = sorted(groups.values(), key=grp_key)

    rank_map = {}
    for rank, grp in enumerate(ordered, start=1):
        for m in grp: rank_map[m] = rank
    return ordered, rank_map


# ──────────────────────────────────────────────────────────────────
# 메인
# ──────────────────────────────────────────────────────────────────
def _process_group(payload):
    """One (task, mode) group → (rows_for_summary, pair_rows, tied_groups, point_scores, metric_name, hib).
    Pickle-safe; runs inside ProcessPoolExecutor worker."""
    task, mode, dirs, n_iters, seed, alpha = payload
    try:
        grp = load_group(dirs)
        grp = align_by_ids(grp)
        point_scores, pair_rows, metric_name, hib = pairwise_bootstrap(
            grp, n_iters=n_iters, seed=seed, alpha=alpha
        )
        tied_groups, rank_map = tied_rank_groups(point_scores, pair_rows, hib)
        return {
            "task": task, "mode": mode, "ok": True,
            "point_scores": point_scores, "pair_rows": pair_rows,
            "tied_groups": tied_groups, "rank_map": rank_map,
            "metric_name": metric_name, "hib": hib,
            "n_dirs": len(dirs),
        }
    except Exception:
        import traceback
        return {"task": task, "mode": mode, "ok": False,
                "err": traceback.format_exc()}


def _write_group_outputs(out_dir, res, n_iters):
    task, mode = res["task"], res["mode"]
    metric_name, hib = res["metric_name"], res["hib"]
    point_scores = res["point_scores"]

    df = pd.DataFrame(res["pair_rows"])
    df.insert(0, "task", task); df.insert(1, "mode", mode); df.insert(2, "metric", metric_name)
    df.to_csv(out_dir / f"pairwise_diff_{task}_{mode}.csv", index=False)

    with open(out_dir / f"tied_groups_{task}_{mode}.txt", "w") as f:
        f.write(f"# Task: {task} | Mode: {mode} | Metric: {metric_name} "
                f"(higher_is_better={hib}) | n_iters={n_iters}\n")
        for rank, grp_models in enumerate(res["tied_groups"], start=1):
            scored = sorted(grp_models, key=lambda m: -point_scores[m] if hib else point_scores[m])
            items = "  ".join(f"{m}={point_scores[m]:.4f}" for m in scored)
            f.write(f"Rank {rank}: {items}\n")

    rows = []
    for m, sc in point_scores.items():
        rows.append({
            "task": task, "mode": mode, "metric": metric_name,
            "model": m, "score": float(sc), "rank": res["rank_map"][m],
            "n_models": len(point_scores),
        })
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--n_iters", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.95)
    p.add_argument("--out_subdir", type=str, default="pairwise")
    p.add_argument("--workers", type=int, default=1,
                   help="(task, mode) 그룹 병렬 worker 수. 그룹 ≈ 50–68개 → 코어수까지 권장.")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    root = Path(args.root)
    out_dir = root / args.out_subdir
    out_dir.mkdir(exist_ok=True)

    # parse_dirname: extract_predictions 의 함수 재사용
    from scripts.extract_predictions import parse_dirname

    # group by (task, mode)
    groups = defaultdict(list)
    for d in sorted(root.iterdir()):
        if not d.is_dir(): continue
        if not (d / "preds.npy").exists(): continue
        parsed = parse_dirname(d.name)
        if parsed is None: continue
        _, task, mode = parsed
        groups[(task, mode)].append(d)

    logger.info(f"(task, mode) groups: {len(groups)} (workers={args.workers})")

    payloads = [
        (task, mode, dirs, args.n_iters, args.seed, args.alpha)
        for (task, mode), dirs in sorted(groups.items()) if len(dirs) >= 2
    ]
    skipped = [(t, m) for (t, m), ds in sorted(groups.items()) if len(ds) < 2]
    for t, m in skipped:
        logger.info(f"  [SKIP <2 models] {t}/{m}")

    summary = []

    def _handle(res):
        if not res["ok"]:
            logger.error(f"[FAIL] {res['task']}/{res['mode']}: "
                         f"{res['err'].splitlines()[-1]}")
            return
        summary.extend(_write_group_outputs(out_dir, res, args.n_iters))
        logger.info(f"  [{res['task']}/{res['mode']}] {res['n_dirs']} models "
                    f"→ {len(res['tied_groups'])} ranks")

    if args.workers <= 1:
        for pl in payloads:
            _handle(_process_group(pl))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            for res in ex.map(_process_group, payloads, chunksize=1):
                _handle(res)

    if summary:
        pd.DataFrame(summary).sort_values(
            ["task", "mode", "rank", "score"]
        ).to_csv(out_dir / "pairwise_summary.csv", index=False)
        logger.info(f"Summary → {out_dir / 'pairwise_summary.csv'}")


if __name__ == "__main__":
    main()
