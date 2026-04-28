"""
기존 heedb_umap_*_n*.npz의 coords를 재사용해서 age 기준 재채색.

같은 seed/n_per_class로 재샘플링 → row 순서 일치 → age만 merge하여 시각화.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

HEEDB_ROOT = Path("/home/irteam/ddn-opendata1/h5/heedb/v4.0")
TABLE_CSV = HEEDB_ROOT / "heedb_table.csv"
LABEL_CSV = HEEDB_ROOT / "heedb_labels.csv"

AGE_SCALE = 100.0  # 0.64 → 64세

ENCODER_TITLE = {
    "founder": "ECG-Founder",
    "cpc":     "CPC",
}


def rebuild_rows(n_per_class: int, seed: int = 42) -> pd.DataFrame:
    labels = pd.read_csv(
        LABEL_CSV,
        usecols=["filepath", "NORMAL_ECG", "ABNORMAL_ECG"],
    )
    normal = labels[labels["NORMAL_ECG"] & ~labels["ABNORMAL_ECG"]]
    abnormal = labels[labels["ABNORMAL_ECG"] & ~labels["NORMAL_ECG"]]
    n = min(n_per_class, len(normal), len(abnormal))
    normal_s = normal.sample(n=n, random_state=seed)[["filepath"]].copy()
    normal_s["is_normal"] = 1
    abnormal_s = abnormal.sample(n=n, random_state=seed)[["filepath"]].copy()
    abnormal_s["is_normal"] = 0
    sampled = pd.concat([normal_s, abnormal_s], ignore_index=True)

    table = pd.read_csv(TABLE_CSV,
                        usecols=["filepath", "fs", "sid", "age"],
                        low_memory=False)
    merged = sampled.merge(table, on="filepath", how="inner")
    return merged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_class", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--npz", type=str, required=True)
    parser.add_argument("--encoder_name", type=str, default=None,
                        help="title에 쓸 모델명. 미지정 시 npz 파일명에서 추론")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    npz_path = Path(args.npz)
    # 파일명에서 encoder 추론 (heedb_umap_<enc>_n<N>.npz)
    stem = npz_path.stem  # heedb_umap_cpc_n10000
    enc_key = None
    for k in ENCODER_TITLE:
        if f"_{k}_" in f"_{stem}_":
            enc_key = k
            break
    if args.encoder_name:
        enc_display = args.encoder_name
    elif enc_key:
        enc_display = ENCODER_TITLE[enc_key]
    else:
        enc_display = stem

    if args.output is None:
        args.output = str(npz_path.with_name(npz_path.stem + "_age.png"))

    # 1) npz 로드
    logging.info(f"load {npz_path}")
    data = np.load(npz_path)
    coords = data["coords"]
    labels = data["labels"]
    logging.info(f"  coords={coords.shape}, labels={labels.shape}")

    # 2) 동일 seed로 rows 재생성 → age 취득
    rows = rebuild_rows(args.n_per_class, seed=args.seed)
    logging.info(f"  rebuilt rows: {len(rows)}")

    assert len(rows) == len(coords), \
        f"row mismatch: {len(rows)} vs {len(coords)}"
    assert (rows["is_normal"].values == labels).all(), \
        "is_normal order mismatch - sampling reproducibility broken"

    age_years = rows["age"].values * AGE_SCALE
    has_age = ~np.isnan(age_years)
    logging.info(
        f"  age available: {has_age.sum()}/{len(rows)} "
        f"(median={np.nanmedian(age_years):.1f}, "
        f"iqr=[{np.nanpercentile(age_years,25):.1f}, "
        f"{np.nanpercentile(age_years,75):.1f}])"
    )

    # 3) plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # 좌: 연속 컬러
    ax = axes[0]
    ax.scatter(coords[~has_age, 0], coords[~has_age, 1],
               c="lightgray", s=4, alpha=0.3, label="age N/A")
    sc = ax.scatter(
        coords[has_age, 0], coords[has_age, 1],
        c=age_years[has_age], cmap="viridis",
        vmin=0, vmax=100, s=5, alpha=0.6,
    )
    cbar = plt.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("age (years)")
    ax.set_title(
        f"HEEDB × {enc_display} UMAP — continuous age\n"
        f"n={len(coords)} (age avail {has_age.sum()})",
        fontsize=12,
    )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="upper right", fontsize=9, markerscale=2)
    ax.grid(alpha=0.2)

    # 우: 5 bin
    ax = axes[1]
    bin_edges = [0, 20, 40, 60, 80, 120]
    bin_labels = ["<20", "20-40", "40-60", "60-80", "80+"]
    colors = ["#6a3d9a", "#1f78b4", "#33a02c", "#ff7f00", "#e31a1c"]

    bin_idx = np.digitize(age_years, bin_edges) - 1
    bin_idx = np.clip(bin_idx, 0, len(bin_labels) - 1)

    ax.scatter(coords[~has_age, 0], coords[~has_age, 1],
               c="lightgray", s=4, alpha=0.3, label="age N/A")
    for i, (lbl, col) in enumerate(zip(bin_labels, colors)):
        m = has_age & (bin_idx == i)
        ax.scatter(coords[m, 0], coords[m, 1],
                   c=col, s=5, alpha=0.55,
                   label=f"{lbl} (n={m.sum()})")
    ax.set_title(f"Age groups — {enc_display}", fontsize=12)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend(loc="upper right", fontsize=9, markerscale=2)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    logging.info(f"saved: {out}")


if __name__ == "__main__":
    main()
