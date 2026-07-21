"""
Stratified Fold generate — ecg-fm-benchmarking paper and identical
=========================================================
paper's stratify() function as-is use each dataset's table CSV in
strat_fold column add.

before (paper and identical):
  - physionet dataset (ningbo, cpsc2018, cpsc_extra, georgia, chapman, ptb):
    → file based label stratified split (10-fold)
    → "does not incorporate patient-level split" (paper code comment)
  - ptbxl:
    → original ptbxl_database.csv's strat_fold 1~10 use (patient based, original )
    → if absent, file based fallback
  - code15:
    → patient(id_patient) based stratified split (paper's stratify_batched)
  - zzu:
    → file based label stratified split (10-fold)

Split :
  train = strat_fold < max_fold - 1
  val   = strat_fold == max_fold - 1
  test  = strat_fold == max_fold

run:
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

# ecg-fm-benchmarking's stratify function import
sys.path.insert(0, str(Path("/path/to/workspace/ecg-fm-benchmarking/code")))
from clinical_ts.utils.stratify import stratify

H5_ROOT = Path("/path/to/ecg_data/h5")
LABEL_DIR = SCRIPT_DIR / "labels"

DATASETS = {
    # physionet: file based label stratify (paper identical)
    "chapman":      {"table": H5_ROOT / "physionet/v2.0/chapman_table.csv",      "label": LABEL_DIR / "chapman_paper_labels.csv",      "n_folds": 10, "method": "label"},
    "cpsc2018":     {"table": H5_ROOT / "physionet/v2.0/cpsc2018_table.csv",     "label": LABEL_DIR / "cpsc2018_paper_labels.csv",     "n_folds": 10, "method": "label"},
    "cpsc_extra":   {"table": H5_ROOT / "physionet/v2.0/cpsc_extra_table.csv",   "label": LABEL_DIR / "cpsc_extra_paper_labels.csv",   "n_folds": 10, "method": "label"},
    "georgia":      {"table": H5_ROOT / "physionet/v2.0/georgia_table.csv",      "label": LABEL_DIR / "georgia_paper_labels.csv",      "n_folds": 10, "method": "label"},
    "ningbo":       {"table": H5_ROOT / "physionet/v2.0/ningbo_table.csv",       "label": LABEL_DIR / "ningbo_paper_labels.csv",       "n_folds": 10, "method": "label"},
    "ptb":          {"table": H5_ROOT / "physionet/v2.0/ptb_table.csv",          "label": LABEL_DIR / "ptb_paper_labels.csv",          "n_folds": 5,  "method": "label"},
    # ptbxl: original strat_fold use
    "ptbxl":        {"table": H5_ROOT / "physionet/v2.0/ptbxl_table.csv",        "label": LABEL_DIR / "ptbxl_all_paper_labels.csv",    "n_folds": 10, "method": "ptbxl_original"},
    # code15: patient based stratify
    "code15":       {"table": H5_ROOT / "code15/v2.0/code15_table.csv",          "label": LABEL_DIR / "code15_paper_labels.csv",       "n_folds": 10, "method": "patient"},
    # zzu: file based
    "zzu_pecg":     {"table": H5_ROOT / "ZZU-pECG/v2.0/ecg_table.csv",          "label": LABEL_DIR / "zzu_paper_labels.csv",          "n_folds": 10, "method": "label"},
    # sph: patient based (prepare_data_sph and identical: stratified 10-fold by patient_id)
    "sph":          {"table": H5_ROOT / "sph/v2.0/sph_table.csv",               "label": LABEL_DIR / "sph_paper_labels.csv",          "n_folds": 10, "method": "patient"},
}


def get_label_lists(table_df, label_df):
    """label CSV from multi-label list extract."""
    merged = table_df.merge(label_df, on="filepath", how="left", suffixes=("", "_label"))
    key_cols = {"filepath", "dataset", "pid", "rid", "sid", "oid",
                "age", "gender", "height", "weight", "fs", "channel_name",
                "nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis",
                "bs_corr", "bs_dtw"}
    label_cols = [c for c in label_df.columns if c not in key_cols]

    # each samples's label isdex list
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
    paper's stratify() function file based label stratified fold generate.
    Ningbo, CPSC2018, CPSC-Extra, Georgia, Chapman, PTB, ZZU and identical.
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
    PTB-XL: original ptbxl_database.csv's strat_fold use.
    original if absent, WFDB file name from ecg_id extract after mapping.
    """
    import glob

    table = pd.read_csv(table_csv, low_memory=False)

    # original ptbxl_database.csv 
    ptbxl_db_candidates = [
        Path("/path/to/ecg_data/raw/physionet.org/files/ptb-xl/1.0.3/ptbxl_database.csv"),
        Path("/path/to/ecg_data/raw/physionet.org/files/ptb-xl/1.0.1/ptbxl_database.csv"),
    ]

    ptbxl_db = None
    for p in ptbxl_db_candidates:
        if p.exists():
            ptbxl_db = p
            break

    if ptbxl_db is not None:
        logging.info(f"  PTB-XL original strat_fold use: {ptbxl_db}")
        db = pd.read_csv(ptbxl_db, index_col="ecg_id")

        # file_name.csv from original_filename → h5 filepath mapping
        fn_csv = H5_ROOT / "physionet/v2.0/file_name.csv"
        fn_df = pd.read_csv(fn_csv)
        fn_df = fn_df[fn_df["dataset"] == "ptbxl"]

        # original_filename (example: HR00001) → ptbxl_database's filename_hr from matching
        # ptbxl_database filename_hr: records500/00000/00001_hr
        ecg_id_map = {}
        for ecg_id, row in db.iterrows():
            fn_hr = str(row.get("filename_hr", ""))
            #  from file name extract: records500/00000/00001_hr → 00001
            stem = Path(fn_hr).stem.replace("_hr", "").replace("_lr", "")
            # HR00001  as convert
            hr_name = f"HR{stem}"
            ecg_id_map[hr_name] = int(row.get("strat_fold", -1))

        # h5 filepath → original_filename → strat_fold mapping
        orig_map = dict(zip(fn_df["h5_filepath"], fn_df["original_filename"]))
        table["strat_fold"] = table["filepath"].apply(
            lambda fp: ecg_id_map.get(orig_map.get(fp, ""), -1)
        )

        # mapping confirm
        matched = (table["strat_fold"] >= 0).sum()
        logging.info(f"  original fold mapping: {matched:,}/{len(table):,}")

        if matched < len(table) * 0.5:
            logging.warning("  mapping rate  → fallback to label stratify")
            return build_fold_label_stratify(table_csv, label_csv, n_folds)
    else:
        logging.warning("  ptbxl_database.csv none → fallback to label stratify")
        return build_fold_label_stratify(table_csv, label_csv, n_folds)

    # fold above 0-based by convert (original 1~10)
    if table["strat_fold"].min() == 1:
        table["strat_fold"] = table["strat_fold"] - 1

    table.to_csv(table_csv, index=False)
    return n_folds


def build_fold_patient_stratify(table_csv, label_csv, n_folds=10):
    """
    CODE-15%: patient(pid) based stratified fold.
    paper's stratify_batched and identical .
    """
    table = pd.read_csv(table_csv, low_memory=False)
    labels = pd.read_csv(label_csv, low_memory=False)
    merged = table.merge(labels, on="filepath", how="left", suffixes=("", "_label"))

    key_cols = {"filepath", "dataset", "pid", "rid", "sid", "oid",
                "age", "gender", "height", "weight", "fs", "channel_name",
                "nan_ratio", "amp_mean", "amp_std", "amp_skewness", "amp_kurtosis",
                "bs_corr", "bs_dtw"}
    label_cols = [c for c in labels.columns if c not in key_cols]

    # patient per label 
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

    # patient ID → fold mapping
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
    parser = argparse.ArgumentParser(description="Stratified Fold generate (paper identical)")
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
            logging.warning(f"  table CSV none: {cfg['table']}")
            continue
        if not cfg["label"].exists():
            logging.warning(f"  label CSV none: {cfg['label']}")
            continue

        builder = METHOD_MAP[cfg["method"]]
        n_folds = builder(cfg["table"], cfg["label"], cfg["n_folds"])

        #  confirm
        df = pd.read_csv(cfg["table"], usecols=["strat_fold"])
        dist = df["strat_fold"].value_counts().sort_index()
        max_fold = int(dist.index.max())
        train_n = len(df[df.strat_fold < max_fold - 1])
        val_n = len(df[df.strat_fold == max_fold - 1])
        test_n = len(df[df.strat_fold == max_fold])
        logging.info(f"  {n_folds}-fold (max={max_fold})")
        logging.info(f"  train: {train_n:,} / val: {val_n:,} / test: {test_n:,}")

    logging.info("\ndone!")


if __name__ == "__main__":
    main()
