"""
Paper-style 결과 표 생성
=========================
pairwise_summary.csv + bootstrap_summary.csv 를 읽어서
mode별 표(task × model) 만들고 마킹:
  - **bold**     : 점추정 best (수치 최고)
  - __underline__: best와 통계적 유의차 없음 (rank 1 동순위 그룹 안)

분류 task: macro-AUROC ↑ (높을수록 좋음)
회귀 task: z-norm MAE  ↓ (낮을수록 좋음)

사용법:
  python scripts/make_summary_table.py --root <RESULT_ROOT>

산출물 (root/pairwise/):
  - summary_<mode>.csv              raw scores
  - summary_<mode>_marked.csv       마킹된 스코어 ("**0.862**", "__0.862__")
  - summary_<mode>.md               markdown 표 (bold/underline 렌더링)
  - summary_<mode>_ci.csv           점추정 + 95% CI (long format)
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("summary_table")

# ──────────────────────────────────────────────────────────────────
# 카테고리 매핑 (paper Table 1 순서)
# ──────────────────────────────────────────────────────────────────
TASK_CATEGORY = {
    # Adult ECG interpretation
    "ptb":            ("Adult ECG", "PTB"),
    "ningbo":         ("Adult ECG", "Ningbo"),
    "cpsc2018":       ("Adult ECG", "CPSC2018"),
    "cpsc_extra":     ("Adult ECG", "CPSC-Extra"),
    "georgia":        ("Adult ECG", "Georgia"),
    "chapman":        ("Adult ECG", "Chapman"),
    "chapman_rhythm": ("Adult ECG", "Chapman (rhythm)"),
    "sph_diag":       ("Adult ECG", "SPH"),
    "code15":         ("Adult ECG", "CODE-15%"),
    "ptbxl_all":      ("Adult ECG", "PTB-XL (all)"),
    "ptbxl_diag":     ("Adult ECG", "PTB-XL (diag)"),
    "ptbxl_form":     ("Adult ECG", "PTB-XL (form)"),
    "ptbxl_rhythm":   ("Adult ECG", "PTB-XL (rhythm)"),
    "ptbxl_sub":      ("Adult ECG", "PTB-XL (sub)"),
    "ptbxl_super":    ("Adult ECG", "PTB-XL (super)"),
    # Pediatric
    "zzu_pecg":       ("Ped. ECG", "ZZU pECG"),
    # Cardiac structure
    "echonext":       ("Cardiac struct.", "EchoNext"),
    # MIMIC (회귀/분류 혼재)
    "mimic_cardiac":      ("Cardiac out.",     "MIMIC (Cardiac)"),
    "mimic_noncardiac":   ("Non-cardiac out.", "MIMIC (Non-cardiac)"),
    "mimic_deterioration":("Acute care",       "MIMIC (Deterioration)"),
    "mimic_mortality":    ("Acute care",       "MIMIC (Mortality)"),
    "mimic_icu_admission":("Acute care",       "MIMIC (ICU)"),
    "mimic_sex":          ("Patient char.",    "MIMIC (Sex)"),
    "mimic_age":          ("Patient char.",    "MIMIC (Age)"),
    "mimic_biometrics":   ("Patient char.",    "MIMIC (Biometrics)"),
    "mimic_ecg_features": ("Patient char.",    "MIMIC (ECG Features)"),
    "mimic_labvalues":    ("Patient char.",    "MIMIC (Lab Values)"),
    "mimic_vitals":       ("Patient char.",    "MIMIC (Vital Signs)"),
}

# Paper의 모델 컬럼 순서 (왼→오)
MODEL_ORDER = [
    "ecg_founder", "ecg_jepa", "st_mem", "merl",
    "ecgfm_ked", "hubert_ecg", "ecg_fm", "cpc",
]
MODEL_DISPLAY = {
    "ecg_founder": "ECGFounder",
    "ecg_jepa":    "ECG-JEPA",
    "st_mem":      "ST-MEM",
    "merl":        "MERL",
    "ecgfm_ked":   "ECGFM-KED",
    "hubert_ecg":  "HuBERT-ECG",
    "ecg_fm":      "ECG-FM",
    "cpc":         "ECG-CPC",
}

CATEGORY_ORDER = [
    "Adult ECG", "Ped. ECG", "Cardiac struct.",
    "Cardiac out.", "Non-cardiac out.", "Acute care", "Patient char.",
]


# ──────────────────────────────────────────────────────────────────
# Marking: best(bold) / tied(underline)
# ──────────────────────────────────────────────────────────────────
def mark_cell(score, is_best, is_tied, fmt="{:.3f}"):
    if pd.isna(score): return "—"
    s = fmt.format(score)
    if is_best:  return f"**{s}**"      # 단독 1등
    if is_tied:  return f"__{s}__"      # 1등과 통계적 동순위
    return s


# ──────────────────────────────────────────────────────────────────
# 한 mode의 표 만들기
# ──────────────────────────────────────────────────────────────────
def build_table(df_mode, mode, out_dir):
    """
    df_mode: pairwise_summary.csv 에서 한 mode만 필터링한 DataFrame.
             cols: task, mode, metric, model, score, rank, n_models
    """
    tasks = sorted(df_mode["task"].unique())
    models_present = [m for m in MODEL_ORDER if m in df_mode["model"].unique()]
    extra_models = sorted(set(df_mode["model"]) - set(models_present))
    models = models_present + extra_models

    # task → metric (방향 결정용)
    task_metric = {t: df_mode[df_mode.task == t]["metric"].iloc[0] for t in tasks}
    # task_type 추론: regression이면 metric=znorm_mae
    task_dir = {t: ("↓" if task_metric[t] == "znorm_mae" else "↑") for t in tasks}

    # raw / marked / score-pivot 만들기
    score_mat = pd.DataFrame(index=tasks, columns=models, dtype=float)
    rank_mat  = pd.DataFrame(index=tasks, columns=models, dtype=float)
    for _, r in df_mode.iterrows():
        score_mat.loc[r["task"], r["model"]] = r["score"]
        rank_mat.loc [r["task"], r["model"]] = r["rank"]

    marked_mat = pd.DataFrame(index=tasks, columns=models, dtype=object)
    for t in tasks:
        higher_better = (task_dir[t] == "↑")
        scores = score_mat.loc[t]
        ranks  = rank_mat.loc[t]
        rank1_models = ranks[ranks == 1].index.tolist()
        if not rank1_models:
            for m in models: marked_mat.loc[t, m] = mark_cell(scores[m], False, False)
            continue
        # 단독 best (점추정 최대/최소) — bold
        if higher_better:
            best_score = scores[rank1_models].max()
        else:
            best_score = scores[rank1_models].min()
        for m in models:
            v = scores[m]
            in_rank1 = m in rank1_models
            is_best  = in_rank1 and (not pd.isna(v)) and np.isclose(v, best_score)
            is_tied  = in_rank1 and not is_best
            # 단독 1등이면 다른 모델에는 underline 안 들어감 (paper 관례)
            marked_mat.loc[t, m] = mark_cell(v, is_best, is_tied)

    # 카테고리/순서 정렬
    rows = []
    for t in tasks:
        cat, disp = TASK_CATEGORY.get(t, ("Other", t))
        rows.append({"category": cat, "task": t, "display": disp,
                     "direction": task_dir[t]})
    meta = pd.DataFrame(rows)

    cat_rank = {c: i for i, c in enumerate(CATEGORY_ORDER)}
    meta["cat_order"] = meta["category"].map(lambda c: cat_rank.get(c, 999))
    meta = meta.sort_values(["cat_order", "display"]).reset_index(drop=True)

    # ── CSV: raw scores ──
    raw = score_mat.loc[meta["task"]].copy()
    raw.insert(0, "Task", [f"{d} {dr}" for d, dr in zip(meta["display"], meta["direction"])])
    raw.insert(0, "Category", meta["category"].values)
    raw.columns = ["Category", "Task"] + [MODEL_DISPLAY.get(m, m) for m in models]
    raw.to_csv(out_dir / f"summary_{mode}.csv", index=False, float_format="%.3f")

    # ── CSV: marked ──
    mk = marked_mat.loc[meta["task"]].copy()
    mk.insert(0, "Task", [f"{d} {dr}" for d, dr in zip(meta["display"], meta["direction"])])
    mk.insert(0, "Category", meta["category"].values)
    mk.columns = ["Category", "Task"] + [MODEL_DISPLAY.get(m, m) for m in models]
    mk.to_csv(out_dir / f"summary_{mode}_marked.csv", index=False)

    # ── Markdown ──
    md_lines = []
    md_lines.append(f"# Benchmark Results — `{mode}`")
    md_lines.append("")
    md_lines.append("**bold** = best (point estimate). __underline__ = tied with best "
                    "(95% paired bootstrap CI of difference contains 0). "
                    "↑ higher is better (macro-AUROC), ↓ lower is better (z-norm MAE).")
    md_lines.append("")
    header = ["Category", "Task"] + [MODEL_DISPLAY.get(m, m) for m in models]
    md_lines.append("| " + " | ".join(header) + " |")
    md_lines.append("|" + "|".join(["---"] * len(header)) + "|")
    last_cat = None
    for _, row in meta.iterrows():
        cat = row["category"] if row["category"] != last_cat else ""
        last_cat = row["category"]
        task_label = f"{row['display']} {row['direction']}"
        cells = [marked_mat.loc[row["task"], m] for m in models]
        md_lines.append("| " + " | ".join([cat, task_label] + cells) + " |")
    (out_dir / f"summary_{mode}.md").write_text("\n".join(md_lines) + "\n")

    return len(meta)


# ──────────────────────────────────────────────────────────────────
# CI 합본 (long format)
# ──────────────────────────────────────────────────────────────────
def write_ci_table(boot_csv, pair_csv, out_dir):
    if not boot_csv.exists():
        logger.warning(f"no bootstrap_summary.csv → CI 표 skip")
        return
    df = pd.read_csv(boot_csv)
    if pair_csv.exists():
        rank_df = pd.read_csv(pair_csv)[["task", "mode", "model", "rank"]]
        df = df.merge(rank_df, on=["task", "mode", "model"], how="left")

    # 카테고리 추가
    df["category"]      = df["task"].map(lambda t: TASK_CATEGORY.get(t, ("Other", t))[0])
    df["task_display"]  = df["task"].map(lambda t: TASK_CATEGORY.get(t, ("Other", t))[1])
    df["model_display"] = df["model"].map(lambda m: MODEL_DISPLAY.get(m, m))
    df["score_str"]     = df.apply(
        lambda r: f"{r['point']:.3f} [{r['ci_low']:.3f}, {r['ci_high']:.3f}]", axis=1
    )

    cols = ["category", "task_display", "task", "mode", "metric",
            "model_display", "model", "point", "ci_low", "ci_high",
            "rank", "n_test", "score_str"]
    cols = [c for c in cols if c in df.columns]
    df = df[cols].sort_values(["mode", "task", "model"])
    df.to_csv(out_dir / "summary_ci_long.csv", index=False, float_format="%.4f")
    logger.info(f"CI long-format → {out_dir / 'summary_ci_long.csv'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--out_subdir", type=str, default="pairwise")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    root = Path(args.root)
    out_dir = root / args.out_subdir
    out_dir.mkdir(exist_ok=True)

    pair_csv = out_dir / "pairwise_summary.csv"
    boot_csv = root / "bootstrap_summary.csv"
    if not pair_csv.exists():
        raise FileNotFoundError(f"{pair_csv} 없음 — bootstrap_pairwise.py 먼저 실행")

    df = pd.read_csv(pair_csv)
    for mode in sorted(df["mode"].unique()):
        n = build_table(df[df["mode"] == mode], mode, out_dir)
        logger.info(f"  [{mode}] {n} tasks → summary_{mode}.csv / .md / _marked.csv")

    write_ci_table(boot_csv, pair_csv, out_dir)
    logger.info(f"Done → {out_dir}")


if __name__ == "__main__":
    main()
