#!/usr/bin/env python
"""
Category-means benchmark plot (reference layout): 3 eval settings (rows) x
{classification (higher better), regression (lower better)} (cols), grouped
bars over models. Values are per-category means read from each run's
test_metrics.txt. MIMIC is decomposed into its report sub-groups.

Baselines / MoRyECG / A5 come from --old_root (all 3 modes). The new
A5-RVQ comes from --new_root (linear_probe only) and therefore appears only
in the Linear Probe row.

Usage:
  python scripts/plot_rvqgnn_category_means.py \
      --old_root results/20260511_172035 \
      --new_root results/20260722_160603 \
      --out results/20260722_160603/rvqgnn_category_means.png
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ── models (reference 6 + A5-RVQ), display + color ───────────────────────────
MODELS = [
    ("ecg_founder",      "ECGFounder",     "#2ca02c", False),
    ("ecg_jepa",         "ECG-JEPA",       "#ff7f0e", False),
    ("st_mem",           "ST-MEM",         "#9467bd", False),
    ("cpc",              "ECG-CPC",        "#e377c2", False),
    ("moryecg_cb1024",   "MoRyECG",        "#1f77b4", False),
    ("moryecg_a5",       "MoRyECG-A5",     "#d62728", False),
    ("moryecg_a5_rvqgnn","MoRyECG-A5-RVQ", "#8c1d1d", True),   # hatched = new
]
MODES = [
    ("linear_probe",    "Linear Probe"),
    ("attention_probe", "Attention Probe"),
    ("finetune_linear", "Fine-tune"),
]

# regular classification tasks -> category (macro AUROC via key 'auroc_macro')
ADULT = ["ptb","ningbo","cpsc2018","cpsc_extra","georgia","chapman","chapman_rhythm",
         "code15","ptbxl_all","ptbxl_super","ptbxl_diag","ptbxl_sub","ptbxl_form",
         "ptbxl_rhythm","sph_diag"]
CLS_CATS = ["Adult ECG","Ped ECG","Cardiac\nstruct.","Cardiac\nout.",
            "Non-card.\nout.","Acute\ncare","Sex"]
REG_CATS = ["Age","Biometr.","ECG feat.","Lab","Vitals"]

# mimic sub-group metric keys (from joint mimic test_metrics.txt)
MIMIC_CLS = {  # category -> list of mimic metric keys to average
    "Cardiac\nout.":   ["cardiac_auroc_macro"],
    "Non-card.\nout.": ["noncardiac_auroc_macro"],
    "Acute\ncare":     ["deterioration_auroc_macro","mortality_auroc_macro","icu_auroc_macro"],
    "Sex":             ["sex_auroc_macro"],
}
MIMIC_REG = {  # regression category -> mimic metric key
    "Age":       "age_mae_macro",
    "Biometr.":  "biometrics_mae_macro",
    "ECG feat.": "ecg_features_mae_macro",
    "Lab":       "labvalues_mae_macro",
    "Vitals":    "vitals_mae_macro",
}


def read_metrics(path):
    d = {}
    if not path.exists():
        return d
    for line in path.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                d[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return d


def root_for(model, new_root, old_root):
    return new_root if model == "moryecg_a5_rvqgnn" else old_root


def cls_category_means(model, mode, new_root, old_root):
    """Return dict cat -> mean AUROC (or np.nan if no data)."""
    root = root_for(model, new_root, old_root)
    out = {}
    # Adult ECG
    vals = [read_metrics(root / f"{model}_{t}_{mode}" / "test_metrics.txt").get("auroc_macro")
            for t in ADULT]
    vals = [v for v in vals if v is not None]
    out["Adult ECG"] = float(np.mean(vals)) if vals else np.nan
    # Ped ECG (zzu), Cardiac struct (echonext)
    for cat, task in [("Ped ECG", "zzu_pecg"), ("Cardiac\nstruct.", "echonext")]:
        v = read_metrics(root / f"{model}_{task}_{mode}" / "test_metrics.txt").get("auroc_macro")
        out[cat] = v if v is not None else np.nan
    # mimic-derived cls categories
    mm = read_metrics(root / f"{model}_mimic_{mode}" / "test_metrics.txt")
    for cat, keys in MIMIC_CLS.items():
        vs = [mm[k] for k in keys if k in mm]
        out[cat] = float(np.mean(vs)) if vs else np.nan
    return out


def reg_category_means(model, mode, new_root, old_root):
    root = root_for(model, new_root, old_root)
    mm = read_metrics(root / f"{model}_mimic_{mode}" / "test_metrics.txt")
    return {cat: mm.get(key, np.nan) for cat, key in MIMIC_REG.items()}


def grouped_bars(ax, cats, per_model_vals, ylabel, ylim):
    n_models = len(MODELS)
    x = np.arange(len(cats))
    w = 0.8 / n_models
    for mi, (mkey, mdisp, color, hatch) in enumerate(MODELS):
        vals = per_model_vals[mkey]
        heights = [0 if (v is None or np.isnan(v)) else v for v in vals]
        mask = [not (v is None or np.isnan(v)) for v in vals]
        xpos = x - 0.4 + w * (mi + 0.5)
        bars = ax.bar(xpos, heights, w, color=color, edgecolor="black",
                      linewidth=0.4, hatch="///" if hatch else None,
                      label=mdisp)
        # hide bars with no data (nan)
        for b, ok in zip(bars, mask):
            if not ok:
                b.set_height(0)
                b.set_alpha(0)
    ax.set_xticks(x)
    ax.set_xticklabels(cats, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_root", required=True)
    ap.add_argument("--new_root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    old_root = Path(args.old_root); new_root = Path(args.new_root)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    for ri, (mode_key, mode_disp) in enumerate(MODES):
        # classification
        cls_vals = {m[0]: [cls_category_means(m[0], mode_key, new_root, old_root).get(c, np.nan)
                           for c in CLS_CATS] for m in MODELS}
        grouped_bars(axes[ri, 0], CLS_CATS, cls_vals,
                     "mean macro-AUROC ↑", (0.5, 1.0))
        axes[ri, 0].set_title(f"{mode_disp}  ·  classification (higher better)",
                              fontsize=11)
        # regression
        reg_vals = {m[0]: [reg_category_means(m[0], mode_key, new_root, old_root).get(c, np.nan)
                           for c in REG_CATS] for m in MODELS}
        grouped_bars(axes[ri, 1], REG_CATS, reg_vals,
                     "mean z-norm MAE ↓", (0.0, 0.8))
        axes[ri, 1].set_title(f"{mode_disp}  ·  regression (lower better)",
                              fontsize=11)

    # shared legend (top)
    handles = [Patch(facecolor=c, edgecolor="black",
                     hatch="///" if h else None, label=d)
               for (_, d, c, h) in MODELS]
    fig.legend(handles=handles, loc="upper center", ncol=len(MODELS),
               fontsize=9, frameon=False, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("MoRyECG-A5-RVQ benchmark — category means by evaluation setting\n"
                 "(A5-RVQ = linear-probe only; baselines/MoRyECG/A5 = all settings)",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = Path(args.out)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
