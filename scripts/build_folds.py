"""
Stratified Fold 생성 — ecg-fm-benchmarking 논문과 동일
=========================================================
논문의 stratify() 함수를 그대로 사용하여 각 데이터셋의 table CSV에
strat_fold 컬럼을 추가합니다.

전략 (논문과 동일):
  - physionet 데이터셋 (ningbo, cpsc2018, cpsc_extra, georgia, chapman, ptb):
    → 파일 기반 라벨 stratified split (10-fold)
    → "does not incorporate patient-level split" (논문 코드 주석)
  - ptbxl:
    → 원본 ptbxl_database.csv의 strat_fold 1~10 사용 (환자 기반, 원본 제공)
    → 없으면 파일 기반 fallback
  - code15:
    → 환자(id_patient) 기반 stratified split (논문의 stratify_batched)
  - zzu:
    → 파일 기반 라벨 stratified split (10-fold)

Split 규칙:
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
from tqdm.auto import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

# ecg-fm-benchmarking의 stratify 함수 import
sys.path.insert(0, str(Path("/home/irteam/local-node-d/tykim/ecg-fm-benchmarking/code")))
from clinical_ts.utils.stratify import stratify

H5_ROOT = Path("/home/irteam/ddn-opendata1/h5")
LABEL_DIR = SCRIPT_DIR / "labels"

DATASETS = {
    # physionet: 파일 기반 라벨 stratify (논문 동일)
    "chapman":      {"table": H5_ROOT / "physionet/v2.0/chapman_table.csv",      "label": LABEL_DIR / "chapman_bench_labels.csv",      "n_folds": 10, "method": "label"},
    "cpsc2018":     {"table": H5_ROOT / "physionet/v2.0/cpsc2018_table.csv",     "label": LABEL_DIR / "cpsc2018_bench_labels.csv",     "n_folds": 10, "method": "label"},
    "cpsc_extra":   {"table": H5_ROOT / "physionet/v2.0/cpsc_extra_table.csv",   "label": LABEL_DIR / "cpsc_extra_bench_labels.csv",   "n_folds": 10, "method": "label"},
    "georgia":      {"table": H5_ROOT / "physionet/v2.0/georgia_table.csv",      "label": LABEL_DIR / "georgia_bench_labels.csv",      "n_folds": 10, "method": "label"},
    "ningbo":       {"table": H5_ROOT / "physionet/v2.0/ningbo_table.csv",       "label": LABEL_DIR / "ningbo_bench_labels.csv",       "n_folds": 10, "method": "label"},
    "ptb":          {"table": H5_ROOT / "physionet/v2.0/ptb_table.csv",          "label": LABEL_DIR / "ptb_bench_labels.csv",          "n_folds": 5,  "method": "label"},
    # ptbxl: 원본 strat_fold 사용
    "ptbxl":        {"table": H5_ROOT / "physionet/v2.0/ptbxl_table.csv",        "label": LABEL_DIR / "ptbxl_all_bench_labels.csv",    "n_folds": 10, "method": "ptbxl_original"},
    # code15: 환자 기반 stratify
    "code15":       {"table": H5_ROOT / "code15/v2.0/code15_table.csv",          "label": LABEL_DIR / "code15_bench_labels.csv",       "n_folds": 10, "method": "patient"},
    # zzu: 파일 기반
    "zzu_pecg":     {"table": H5_ROOT / "ZZU-pECG/v2.0/ecg_table.csv",          "label": LABEL_DIR / "zzu_bench_labels.csv",          "n_folds": 10, "method": "label"},
}


def get_label_lists(table_df, label_df):
    """라벨 CSV에서 multi-label 리스트를 추출합니다."""
    merged = table_df.merge(label_df, on="filepath", how="left", suffixes=("", "_label"))
    key_cols = {"filepath", "dataset", "pid", "rid", "sid", "oid",
                "age", "gender", "height", "weight", "fs", "channel_name",
                "nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis",
                "bs_corr", "bs_dtw"}
    label_cols = [c for c in label_df.columns if c not in key_cols]

    # 각 샘플의 라벨 인덱스 리스트
    data = []
    for _, row in merged.iterrows():
        labels = []
        for j, col in enumerate(label_cols):
            val = row.get(col, False)
            if str(val).lower() in ("true", "1", "1.0"):
                labels.append(j)
        data.append(labels)

    classes = list(range(len(label_cols)))
    return data, classes, label_cols


def build_fold_label_stratify(table_csv, label_csv, n_folds=10):
    """
    논문의 stratify() 함수로 파일 기반 라벨 stratified fold 생성.
    Ningbo, CPSC2018, CPSC-Extra, Georgia, Chapman, PTB, ZZU와 동일.
    """
    table = pd.read_csv(table_csv, low_memory=False)
    labels = pd.read_csv(label_csv, low_memory=False)
    data, classes, label_cols = get_label_lists(table, labels)

    ratios = [1.0 / n_folds] * n_folds
    stratified_ids = stratify(data, classes, ratios, random_seed=0)

    table["strat_fold"] = -1
    for fold_idx, indices in enumerate(stratified_ids):
        table.loc[list(indices), "strat_fold"] = fold_idx

    table.to_csv(table_csv, index=False)
    return n_folds


def build_fold_ptbxl_original(table_csv, label_csv, n_folds=10):
    """
    PTB-XL: 원본 ptbxl_database.csv의 strat_fold 사용.
    원본이 없으면 WFDB 파일명에서 ecg_id 추출 후 매핑.
    """
    import glob

    table = pd.read_csv(table_csv, low_memory=False)

    # 원본 ptbxl_database.csv 찾기
    ptbxl_db_candidates = [
        Path("/home/irteam/ddn-opendata1/raw/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"),
        Path("/home/irteam/ddn-opendata1/raw/physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv"),
    ]

    ptbxl_db = None
    for p in ptbxl_db_candidates:
        if p.exists():
            ptbxl_db = p
            break

    if ptbxl_db is not None:
        logging.info(f"  PTB-XL 원본 strat_fold 사용: {ptbxl_db}")
        db = pd.read_csv(ptbxl_db, index_col="ecg_id")

        # file_name.csv에서 original_filename → h5 filepath 매핑
        fn_csv = H5_ROOT / "physionet/v2.0/file_name.csv"
        fn_df = pd.read_csv(fn_csv)
        fn_df = fn_df[fn_df["dataset"] == "ptbxl"]

        # original_filename (예: HR00001) → ptbxl_database의 filename_hr에서 매칭
        # ptbxl_database filename_hr: records500/00000/00001_hr
        ecg_id_map = {}
        for ecg_id, row in db.iterrows():
            fn_hr = str(row.get("filename_hr", ""))
            # 끝부분에서 파일명 추출: records500/00000/00001_hr → 00001
            stem = Path(fn_hr).stem.replace("_hr", "").replace("_lr", "")
            # HR00001 형식으로 변환
            hr_name = f"HR{stem}"
            ecg_id_map[hr_name] = int(row.get("strat_fold", -1))

        # h5 filepath → original_filename → strat_fold 매핑
        orig_map = dict(zip(fn_df["h5_filepath"], fn_df["original_filename"]))
        table["strat_fold"] = table["filepath"].apply(
            lambda fp: ecg_id_map.get(orig_map.get(fp, ""), -1)
        )

        # 매핑 확인
        matched = (table["strat_fold"] >= 0).sum()
        logging.info(f"  원본 fold 매핑: {matched:,}/{len(table):,}")

        if matched < len(table) * 0.5:
            logging.warning("  매핑률 낮음 → fallback to label stratify")
            return build_fold_label_stratify(table_csv, label_csv, n_folds)
    else:
        logging.warning("  ptbxl_database.csv 없음 → fallback to label stratify")
        return build_fold_label_stratify(table_csv, label_csv, n_folds)

    # fold 범위를 0-based로 변환 (원본은 1~10)
    if table["strat_fold"].min() == 1:
        table["strat_fold"] = table["strat_fold"] - 1

    table.to_csv(table_csv, index=False)
    return n_folds


def build_fold_patient_stratify(table_csv, label_csv, n_folds=10):
    """
    CODE-15%: 환자(pid) 기반 stratified fold.
    논문의 stratify_batched와 동일한 접근.
    """
    table = pd.read_csv(table_csv, low_memory=False)
    labels = pd.read_csv(label_csv, low_memory=False)
    merged = table.merge(labels, on="filepath", how="left", suffixes=("", "_label"))

    key_cols = {"filepath", "dataset", "pid", "rid", "sid", "oid",
                "age", "gender", "height", "weight", "fs", "channel_name",
                "nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis",
                "bs_corr", "bs_dtw"}
    label_cols = [c for c in labels.columns if c not in key_cols]

    # 환자별 라벨 집약
    def get_patient_labels(group):
        all_labels = []
        for _, row in group.iterrows():
            for j, col in enumerate(label_cols):
                if str(row.get(col, False)).lower() in ("true", "1", "1.0"):
                    all_labels.append(j)
        return all_labels

    patient_groups = merged.groupby("pid")
    patient_ids = list(patient_groups.groups.keys())
    patient_labels = [get_patient_labels(patient_groups.get_group(pid)) for pid in patient_ids]
    patient_counts = [len(patient_groups.get_group(pid)) for pid in patient_ids]

    classes = list(range(len(label_cols)))
    ratios = [1.0 / n_folds] * n_folds
    stratified_ids = stratify(patient_labels, classes, ratios,
                              samples_per_group=patient_counts, random_seed=0)

    # 환자 ID → fold 매핑
    patient_fold = {}
    for fold_idx, indices in enumerate(stratified_ids):
        for idx in indices:
            patient_fold[patient_ids[idx]] = fold_idx

    table["strat_fold"] = table["pid"].apply(lambda x: patient_fold.get(x, -1))
    table.to_csv(table_csv, index=False)
    return n_folds


METHOD_MAP = {
    "label":          build_fold_label_stratify,
    "ptbxl_original": build_fold_ptbxl_original,
    "patient":        build_fold_patient_stratify,
}


def main():
    parser = argparse.ArgumentParser(description="Stratified Fold 생성 (논문 동일)")
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
        logging.info(f"\n=== {ds} (method={cfg['method']}) ===")
        if not cfg["table"].exists():
            logging.warning(f"  table CSV 없음: {cfg['table']}")
            continue
        if not cfg["label"].exists():
            logging.warning(f"  label CSV 없음: {cfg['label']}")
            continue

        builder = METHOD_MAP[cfg["method"]]
        n_folds = builder(cfg["table"], cfg["label"], cfg["n_folds"])

        # 분포 확인
        df = pd.read_csv(cfg["table"], usecols=["strat_fold"])
        dist = df["strat_fold"].value_counts().sort_index()
        max_fold = int(dist.index.max())
        train_n = len(df[df.strat_fold < max_fold - 1])
        val_n = len(df[df.strat_fold == max_fold - 1])
        test_n = len(df[df.strat_fold == max_fold])
        logging.info(f"  {n_folds}-fold (max={max_fold})")
        logging.info(f"  train: {train_n:,} / val: {val_n:,} / test: {test_n:,}")

    logging.info("\n완료!")


if __name__ == "__main__":
    main()
