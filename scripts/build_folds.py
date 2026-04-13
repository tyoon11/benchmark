"""
Stratified Fold 생성
=====================
각 데이터셋의 table CSV에 strat_fold 컬럼을 추가합니다.
ecg-fm-benchmarking과 동일한 방식: label 기반 stratified k-fold.

Split 규칙 (ecg-fm-benchmarking 동일):
  train = strat_fold < max_fold - 1
  val   = strat_fold == max_fold - 1
  test  = strat_fold == max_fold

실행:
  python scripts/build_folds.py --all
  python scripts/build_folds.py --dataset ptbxl
"""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

H5_ROOT = Path("/home/irteam/ddn-opendata1/h5")
LABEL_DIR = SCRIPT_DIR / "labels"

DATASETS = {
    "ptbxl": {
        "table": H5_ROOT / "physionet/v2.0/ptbxl_table.csv",
        "label": LABEL_DIR / "ptbxl_all_bench_labels.csv",
        "n_folds": 10,
    },
    "chapman": {
        "table": H5_ROOT / "physionet/v2.0/chapman_table.csv",
        "label": LABEL_DIR / "chapman_bench_labels.csv",
        "n_folds": 10,
    },
    "cpsc2018": {
        "table": H5_ROOT / "physionet/v2.0/cpsc2018_table.csv",
        "label": LABEL_DIR / "cpsc2018_bench_labels.csv",
        "n_folds": 10,
    },
    "cpsc_extra": {
        "table": H5_ROOT / "physionet/v2.0/cpsc_extra_table.csv",
        "label": LABEL_DIR / "cpsc_extra_bench_labels.csv",
        "n_folds": 10,
    },
    "georgia": {
        "table": H5_ROOT / "physionet/v2.0/georgia_table.csv",
        "label": LABEL_DIR / "georgia_bench_labels.csv",
        "n_folds": 10,
    },
    "ningbo": {
        "table": H5_ROOT / "physionet/v2.0/ningbo_table.csv",
        "label": LABEL_DIR / "ningbo_bench_labels.csv",
        "n_folds": 10,
    },
    "ptb": {
        "table": H5_ROOT / "physionet/v2.0/ptb_table.csv",
        "label": LABEL_DIR / "ptb_bench_labels.csv",
        "n_folds": 5,  # 516개라 5-fold
    },
    "code15": {
        "table": H5_ROOT / "code15/v2.0/code15_table.csv",
        "label": LABEL_DIR / "code15_bench_labels.csv",
        "n_folds": 10,
    },
    "zzu_pecg": {
        "table": H5_ROOT / "ZZU-pECG/v2.0/ecg_table.csv",
        "label": LABEL_DIR / "zzu_bench_labels.csv",
        "n_folds": 10,
    },
}


def build_strat_fold(table_csv, label_csv, n_folds=10, seed=42):
    """
    라벨 기반 stratified k-fold를 table CSV에 추가합니다.

    가장 빈도 높은 라벨을 stratification key로 사용.
    """
    table = pd.read_csv(table_csv, low_memory=False)
    labels = pd.read_csv(label_csv, low_memory=False)

    # 조인
    merged = table.merge(labels, on="filepath", how="left", suffixes=("", "_label"))

    # 라벨 컬럼 추출
    key_cols = {"filepath", "dataset", "pid", "rid", "sid", "oid",
                "age", "gender", "height", "weight", "fs", "channel_name",
                "nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis",
                "bs_corr", "bs_dtw"}
    label_cols = [c for c in labels.columns if c not in key_cols]

    # Stratification key: 가장 빈번한 라벨의 binary 값
    # multi-label이면 첫 번째 True인 라벨의 인덱스 사용
    strat_key = np.zeros(len(merged), dtype=int)
    for i, row in merged.iterrows():
        for j, col in enumerate(label_cols):
            val = row.get(col, False)
            if str(val).lower() in ("true", "1", "1.0"):
                strat_key[i] = j + 1
                break

    # n_folds보다 클래스가 적은 경우 조정
    unique_classes = len(np.unique(strat_key))
    actual_folds = min(n_folds, unique_classes)
    if actual_folds < n_folds:
        logging.warning(f"  클래스 수({unique_classes}) < n_folds({n_folds}) → {actual_folds}-fold로 조정")

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=actual_folds, shuffle=True, random_state=seed)
    table["strat_fold"] = -1
    for fold_idx, (_, test_idx) in enumerate(skf.split(np.zeros(len(table)), strat_key)):
        table.loc[test_idx, "strat_fold"] = fold_idx

    # 저장
    table.to_csv(table_csv, index=False)
    return actual_folds


def main():
    parser = argparse.ArgumentParser(description="Stratified Fold 생성")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if args.all:
        targets = list(DATASETS.keys())
    elif args.dataset:
        targets = [args.dataset]
    else:
        parser.print_help()
        return

    for ds in targets:
        cfg = DATASETS[ds]
        logging.info(f"\n=== {ds} ===")
        if not cfg["table"].exists():
            logging.warning(f"  table CSV 없음: {cfg['table']}")
            continue
        if not cfg["label"].exists():
            logging.warning(f"  label CSV 없음: {cfg['label']}")
            continue

        n_folds = build_strat_fold(cfg["table"], cfg["label"], cfg["n_folds"])

        # 분포 확인
        df = pd.read_csv(cfg["table"], usecols=["strat_fold"])
        dist = df["strat_fold"].value_counts().sort_index()
        max_fold = dist.index.max()
        logging.info(f"  {n_folds}-fold 생성 (max_fold={max_fold})")
        logging.info(f"  train: fold 0~{max_fold-2} ({df[df.strat_fold < max_fold-1].shape[0]:,})")
        logging.info(f"  val:   fold {max_fold-1} ({df[df.strat_fold == max_fold-1].shape[0]:,})")
        logging.info(f"  test:  fold {max_fold} ({df[df.strat_fold == max_fold].shape[0]:,})")

    logging.info("\n완료!")


if __name__ == "__main__":
    main()
