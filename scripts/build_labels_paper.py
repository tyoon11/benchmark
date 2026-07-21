"""
Paper-canonical label build script
==============================
Faithfully reproduces ecg-fm-benchmarking's prepare_data_*() functions.

- physionet (ningbo/cpsc2018/cpsc_extra/georgia/ptb/chapman/stpetersburg):
    WFDB .hea's # Dx: SNOMED → Label Mappings xlsx → diagnosis (label list)
- PTB-XL:
    ptbxl_database.csv + scp_statements.csv → 6 subtask
    (label_all, label_diag, label_form, label_rhythm,
     label_diag_subclass, label_diag_superclass)
- ZZU:
    AttributesDictionary.csv's AHA_code/CHN_code (original identical)

all label min_cnt=10 as filtering (paper identical).

output:
  labels/{dataset}_paper_labels.csv  (key column + binary label)
  labels/{dataset}_paper_labels.json (label name definitions)

run:
  python scripts/build_labels_paper.py --all
  python scripts/build_labels_paper.py --dataset ptbxl
"""

import os
import sys
import glob
import argparse
import logging
import json
import numpy as np
import pandas as pd
import wfdb
from pathlib import Path
from collections import Counter

# ═══════════════════════════════════════════════════════════════
# path
# ═══════════════════════════════════════════════════════════════
H5_ROOT = Path("/path/to/ecg_data/h5")
RAW_ROOT = Path("/path/to/ecg_data/raw/physionet.org/files")
CHALLENGE_BASE = RAW_ROOT / "challenge-2021/1.0.3/training"
LABEL_XLSX = Path("/path/to/workspace/ecg-fm-benchmarking/Label mappings 2021.xlsx")
PTBXL_META = Path("/path/to/ecg_data/raw/ptbxl_metadata")
BENCHMARK_DIR = Path("/path/to/workspace/benchmark")
OUT_DIR = BENCHMARK_DIR / "labels"

MIN_CNT = 10  # paper identical


# ═══════════════════════════════════════════════════════════════
# map_and_filter_labels — paper as-is reproduction
# ═══════════════════════════════════════════════════════════════
def map_and_filter_labels(df, min_cnt, lbl_cols):
    """
    original ecg_utils.map_and_filter_labels() reproduction.

    each lbl_col in for:
      1. all label 
      2. min_cnt or moreis label only  {col}_filtered column generate
      3. {col}_filtered numeric as is {col}_filtered_numeric column generate

    Returns:
        df:        extensioned DataFrame
        lbl_itos:  {col_filtered: [label1, label2, ...]} — label order
    """
    lbl_itos = {}
    for col in lbl_cols:
        # all label flatten
        all_labels = [item for sublist in df[col] for item in sublist]
        unique, cnt = np.unique(all_labels, return_counts=True)
        selected = set(unique[cnt >= min_cnt])

        # filtered column
        df[col + "_filtered"] = df[col].apply(lambda x: [y for y in x if y in selected])

        # numeric mapping (sorted order)
        lbl_list = sorted(selected)
        lbl_itos[col + "_filtered"] = lbl_list
        lbl_stoi = {s: i for i, s in enumerate(lbl_list)}
        df[col + "_filtered_numeric"] = df[col + "_filtered"].apply(
            lambda x: [lbl_stoi[y] for y in x]
        )
    return df, lbl_itos


# ═══════════════════════════════════════════════════════════════
# PhysioNet SNOMED dataset (paper prepare_data_*() reproduction)
# ═══════════════════════════════════════════════════════════════
SNOMED_DATASETS = {
    "chapman":      {"wfdb_dir": CHALLENGE_BASE / "chapman_shaoxing", "sheet": "Chapman"},
    "cpsc2018":     {"wfdb_dir": CHALLENGE_BASE / "cpsc_2018",        "sheet": "CPSC"},
    "cpsc_extra":   {"wfdb_dir": CHALLENGE_BASE / "cpsc_2018_extra",  "sheet": "CPSC-Extra"},
    "georgia":      {"wfdb_dir": CHALLENGE_BASE / "georgia",          "sheet": "G12EC"},
    "ningbo":       {"wfdb_dir": CHALLENGE_BASE / "ningbo",           "sheet": "Ningbo",
                     "extra_map": {"106068003": "ARH"}},  # paper ningbo prepare's  add
    "ptb":          {"wfdb_dir": CHALLENGE_BASE / "ptb",              "sheet": "PTB"},
    "stpetersburg": {"wfdb_dir": CHALLENGE_BASE / "st_petersburg_incart", "sheet": "INCART"},
}


def load_snomed_mapping(sheet_name: str, extra_map: dict = None) -> dict:
    """Label Mappings xlsx load."""
    df = pd.read_excel(LABEL_XLSX, sheet_name=sheet_name, dtype={"SNOMED code": str})
    df = df.dropna(subset=["SNOMED code"])
    mapping = {}
    for _, row in df.iterrows():
        code = str(row["SNOMED code"]).strip()
        diag = str(row["Diagnosis in the dataset"]).strip()
        if code not in mapping:   # paper's   of at first keep
            mapping[code] = diag
    if extra_map:
        mapping.update(extra_map)
    return mapping


def parse_wfdb_dx(hea_path: str) -> list:
    """WFDB .hea from # Dx: SNOMED code list parsing (paper identical)."""
    try:
        rec = wfdb.rdheader(hea_path)
        for c in (rec.comments or []):
            cl = c.strip()
            if cl.lower().startswith("dx:"):
                # paper: codes = [str(int(x)) for x in arrs[1].split(',')]
                codes = []
                for x in cl.split(":", 1)[1].split(","):
                    x = x.strip()
                    try:
                        codes.append(str(int(x)))
                    except ValueError:
                        pass
                return codes
    except Exception:
        pass
    return []


def build_snomed_dataset(dataset_name: str):
    """
    paper prepare_data_ningbo/cpsc2018/chapman etc. reproduction.

    - df["label"] = [SNOMED diagnosis names (from xlsx mapping)]
    - map_and_filter_labels(min_cnt=10)
    - df["label_filtered"], df["label_filtered_numeric"] generate
    """
    cfg = SNOMED_DATASETS[dataset_name]
    wfdb_dir = cfg["wfdb_dir"]
    snomed_map = load_snomed_mapping(cfg["sheet"], cfg.get("extra_map"))

    logging.info(f"  SNOMED mapping: {len(snomed_map)} entries ({cfg['sheet']})")

    # file_name.csv by h5 filepath mapping
    fn_csv = H5_ROOT / "physionet/v2.0/file_name.csv"
    fn_df = pd.read_csv(fn_csv)
    fn_df = fn_df[fn_df["dataset"] == dataset_name]
    orig_to_h5fp = dict(zip(fn_df["original_filename"].astype(str),
                            fn_df["h5_filepath"].astype(str)))

    hea_files = sorted(glob.glob(str(wfdb_dir / "g*" / "*.hea")))
    logging.info(f"  .hea files: {len(hea_files)}")

    records = []
    for hea in hea_files:
        rec_name = os.path.basename(hea).replace(".hea", "")
        codes = parse_wfdb_dx(hea[:-4])
        # paper: labels = [dx_mapping_snomed[code] for code in codes] (mapping failure skip)
        labels = [snomed_map[c] for c in codes if c in snomed_map]
        h5_fp = orig_to_h5fp.get(rec_name)
        if h5_fp:
            records.append({"filepath": h5_fp, "label": labels})

    logging.info(f"  Mapped records: {len(records)}")
    df = pd.DataFrame(records)

    # paper identical: map_and_filter_labels
    df, lbl_itos = map_and_filter_labels(df, min_cnt=MIN_CNT, lbl_cols=["label"])
    labels_itos = lbl_itos["label_filtered"]
    logging.info(f"  Labels (≥{MIN_CNT}): {len(labels_itos)}")

    return df, labels_itos


# ═══════════════════════════════════════════════════════════════
# PTB-XL — paper prepare_data_ptb_xl() reproduction
# ═══════════════════════════════════════════════════════════════
def build_ptbxl_dataset():
    """
    paper prepare_data_ptb_xl() as-is reproduction.

    subtask:
      - label_all         (all SCP code)
      - label_diag        (scp_statements's diagnostic>0)
      - label_form        (scp_statements's form>0)
      - label_rhythm      (scp_statements's rhythm>0)
      - label_diag_subclass    (diag → diagnostic_subclass)
      - label_diag_superclass  (diag → diagnostic_class)
    """
    logging.info("  Loading ptbxl_database.csv + scp_statements.csv")

    ptbxl_db_path = PTBXL_META / "ptbxl_database.csv"
    scp_path = PTBXL_META / "scp_statements.csv"

    df = pd.read_csv(ptbxl_db_path, index_col="ecg_id")
    # scp_codes dict string
    df["scp_codes"] = df["scp_codes"].apply(lambda x: eval(x.replace("nan", "np.nan")))

    scp = pd.read_csv(scp_path)
    scp = scp.set_index(scp.columns[0])

    # paper identical
    diag_codes = scp[scp["diagnostic"] > 0]
    form_codes = scp[scp["form"] > 0]
    rhythm_codes = scp[scp["rhythm"] > 0]

    diag_class_map = {}
    diag_subclass_map = {}
    for id_, row in diag_codes.iterrows():
        if isinstance(row.get("diagnostic_class"), str):
            diag_class_map[id_] = row["diagnostic_class"]
        if isinstance(row.get("diagnostic_subclass"), str):
            diag_subclass_map[id_] = row["diagnostic_subclass"]

    df["label_all"]        = df.scp_codes.apply(lambda x: list(x.keys()))
    df["label_diag"]       = df.scp_codes.apply(lambda x: [y for y in x.keys() if y in diag_codes.index])
    df["label_form"]       = df.scp_codes.apply(lambda x: [y for y in x.keys() if y in form_codes.index])
    df["label_rhythm"]     = df.scp_codes.apply(lambda x: [y for y in x.keys() if y in rhythm_codes.index])
    df["label_diag_subclass"]   = df["label_diag"].apply(lambda x: [diag_subclass_map[y] for y in x if y in diag_subclass_map])
    df["label_diag_superclass"] = df["label_diag"].apply(lambda x: [diag_class_map[y] for y in x if y in diag_class_map])

    # paper min_cnt=10 filtering
    df, lbl_itos = map_and_filter_labels(
        df, min_cnt=MIN_CNT,
        lbl_cols=["label_all", "label_diag", "label_form", "label_rhythm",
                  "label_diag_subclass", "label_diag_superclass"],
    )

    # original strat_fold also 
    # ptbxl_database's filename_hr: records500/00000/00001_hr
    # H5's original_filename: HR00001  (convert_to_h5.py handling results)
    fn_csv = H5_ROOT / "physionet/v2.0/file_name.csv"
    fn_df = pd.read_csv(fn_csv)
    fn_df = fn_df[fn_df["dataset"] == "ptbxl"]
    orig_to_h5fp = dict(zip(fn_df["original_filename"].astype(str),
                            fn_df["h5_filepath"].astype(str)))

    # ecg_id → HR{padded} mapping
    def ecg_id_to_hr(ecg_id):
        return f"HR{int(ecg_id):05d}"

    df["filepath"] = df.index.to_series().apply(lambda x: orig_to_h5fp.get(ecg_id_to_hr(x), None))
    df_mapped = df[df["filepath"].notna()].copy()
    logging.info(f"  PTB-XL records mapped to H5: {len(df_mapped)}/{len(df)}")

    return df_mapped, lbl_itos


# ═══════════════════════════════════════════════════════════════
# ZZU — original prepare_data_zzu_pecg() reproduction (AHA/CHN/ICD-10 description mapping)
# ═══════════════════════════════════════════════════════════════
ZZU_LABEL_COLS = ["icd10_disease_category", "aha_description", "chn_description"]


def build_zzu_dataset():
    """
    original ecg_utils.prepare_data_zzu_pecg() reproduction.

    - AttributesDictionary.csv: samples metadata(Patient_ID, Lead, AHA_code, CHN_code, ICD-10)
    - DiseaseCode.csv:          ICD-10 → (disease type, disease category)
    - ECGCode.csv:              AHA/CHN code → description

    Lead==12 filter,  label column all map_and_filter_labels(min_cnt=10) apply.
    """
    raw_root = Path("/path/to/ecg_data/raw/ZZU-pECG")
    df = pd.read_csv(raw_root / "AttributesDictionary.csv")
    df_disease = pd.read_csv(raw_root / "DiseaseCode.csv")
    df_ecg = pd.read_csv(raw_root / "ECGCode.csv")

    df.columns = df.columns.str.lower()
    df_disease.columns = df_disease.columns.str.lower()
    df_ecg.columns = df_ecg.columns.str.lower()

    # code list parsing (original identical)
    for col in ["aha_code", "chn_code", "icd-10 code"]:
        df[col] = df[col].apply(
            lambda x: [] if pd.isna(x) or x == "Null"
            else [c.strip().replace("'", "") for c in str(x).split(";") if c.strip()]
        )

    # ICD-10 → disease type / category
    type_map, cat_map = {}, {}
    for _, row in df_disease.iterrows():
        for code in str(row["icd-10 code"]).split(";"):
            code = code.strip()
            if code:
                type_map[code] = row["disease type"]
                cat_map[code] = row["disease category"]

    df["icd10_disease_type"] = df["icd-10 code"].apply(
        lambda codes: [type_map[c] for c in codes if c in type_map]
    )
    df["icd10_disease_category"] = df["icd-10 code"].apply(
        lambda codes: [cat_map[c] for c in codes if c in cat_map]
    )

    # AHA/CHN code → description (original: ECGCode.csv)
    aha_map, chn_map = {}, {}
    for _, row in df_ecg.iterrows():
        desc = str(row["description"]).strip()
        aha = str(row["aha(category&code)"]).strip()
        chn = str(row["chn(category&code)"]).strip()
        if aha not in ["N/A", "nan"]:
            aha_map[aha] = desc
        if chn not in ["N/A", "nan"]:
            chn_map[chn] = desc

    df["aha_description"] = df["aha_code"].apply(
        lambda codes: [aha_map[c] for c in codes if c in aha_map]
    )
    df["chn_description"] = df["chn_code"].apply(
        lambda codes: [chn_map[c] for c in codes if c in chn_map]
    )

    # Lead==12 filter (original identical)
    n_all = len(df)
    df = df[df["lead"] == 12].copy()
    logging.info(f"  ZZU 12-lead filter: {len(df)}/{n_all}")

    # H5 filepath mapping
    fn_csv = H5_ROOT / "ZZU-pECG/v2.0/file_name.csv"
    fn_df = pd.read_csv(fn_csv)
    orig_to_h5 = dict(zip(fn_df["original_filename"].astype(str),
                          fn_df["h5_filepath"].astype(str)))

    df["filepath"] = df["filename"].apply(
        lambda x: orig_to_h5.get(str(x).split("/")[-1], None)
    )
    df = df[df["filepath"].notna()].copy()
    logging.info(f"  ZZU after H5 mapping: {len(df)}")

    #  label column all filtering (original identical)
    df, lbl_itos = map_and_filter_labels(df, min_cnt=MIN_CNT, lbl_cols=ZZU_LABEL_COLS)
    for col in ZZU_LABEL_COLS:
        logging.info(f"  {col}: {len(lbl_itos[col + '_filtered'])} labels (≥{MIN_CNT})")
    return df, lbl_itos


def save_zzu_subtasks(df, lbl_itos):
    """
    ZZU 3 subtask save.
      - zzu_paper_labels.csv              (aha_description, default — yaml and match)
      - zzu_icd10_paper_labels.csv        (icd10_disease_category)
      - zzu_chn_paper_labels.csv          (chn_description)
    """
    subtask_to_filename = {
        "aha_description":        "zzu",
        "icd10_disease_category": "zzu_icd10",
        "chn_description":        "zzu_chn",
    }

    for task_col, out_name in subtask_to_filename.items():
        labels_list = list(lbl_itos[task_col + "_filtered"])
        numeric_col = task_col + "_filtered_numeric"

        label_cols = []
        for lbl in labels_list:
            col = str(lbl).replace(" ", "_").replace(",", "").replace("-", "_") \
                          .replace("(", "").replace(")", "").replace("'", "") \
                          .replace("/", "_").replace(":", "_")
            label_cols.append(col)

        out_df = df[["filepath"]].copy()
        for j, col in enumerate(label_cols):
            out_df[col] = df[numeric_col].apply(
                lambda x: j in x if isinstance(x, list) else False
            )

        out_csv = OUT_DIR / f"{out_name}_paper_labels.csv"
        out_df.to_csv(out_csv, index=False)

        out_json = OUT_DIR / f"{out_name}_paper_labels.json"
        with open(out_json, "w") as f:
            json.dump({
                "dataset": out_name,
                "source_col": task_col,
                "n_labels": len(labels_list),
                "labels": {col: str(lbl) for col, lbl in zip(label_cols, labels_list)},
            }, f, indent=2, ensure_ascii=False)

        logging.info(f"  Saved: {out_csv.name} ({len(out_df)} rows, {len(label_cols)} labels)")


# ═══════════════════════════════════════════════════════════════
# CODE-15% — original CSV keep (already paper and identical)
# ═══════════════════════════════════════════════════════════════
def copy_code15_labels():
    """code15 change without existing label CSV """
    import shutil
    src = OUT_DIR / "code15_bench_labels.csv"
    dst = OUT_DIR / "code15_paper_labels.csv"
    if src.exists():
        shutil.copy(src, dst)
        logging.info(f"  Copied: {dst.name}")
        return True
    return False


# ═══════════════════════════════════════════════════════════════
# DataFrame → labels CSV convert
# ═══════════════════════════════════════════════════════════════
def save_label_csv(dataset: str, df: pd.DataFrame, labels_itos, suffix=""):
    """
    df['label_filtered_numeric'] list binary column as extension CSV save.

    Returns: save path
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{dataset}{suffix}_paper_labels.csv"

    # label column name sanitize
    label_cols = []
    for lbl in labels_itos:
        col = str(lbl).replace(" ", "_").replace(",", "").replace("-", "_") \
                      .replace("(", "").replace(")", "").replace("'", "") \
                      .replace("/", "_")
        label_cols.append(col)

    # binary column  only
    out_df = df[["filepath"]].copy()
    for j, col in enumerate(label_cols):
        out_df[col] = df["label_filtered_numeric" if "label_filtered_numeric" in df.columns else f"label_filtered_numeric"].apply(
            lambda x: j in x if isinstance(x, list) else False
        )
    out_df.to_csv(out_csv, index=False)

    # JSON label definition save
    out_json = OUT_DIR / f"{dataset}{suffix}_paper_labels.json"
    with open(out_json, "w") as f:
        json.dump({
            "dataset": f"{dataset}{suffix}",
            "n_labels": len(labels_itos),
            "labels": {col: lbl for col, lbl in zip(label_cols, labels_itos)},
        }, f, indent=2, ensure_ascii=False)

    logging.info(f"  Saved: {out_csv.name} ({len(out_df)} rows, {len(label_cols)} labels)")
    return out_csv


def save_ptbxl_subtasks(df, lbl_itos):
    """PTB-XL 6 subtask each save"""
    for task in ["label_all", "label_diag", "label_form", "label_rhythm",
                 "label_diag_subclass", "label_diag_superclass"]:
        labels_list = lbl_itos[task + "_filtered"]

        # task per by label_filtered_numeric column binary by
        label_cols = []
        for lbl in labels_list:
            col = str(lbl).replace(" ", "_").replace(",", "").replace("-", "_") \
                          .replace("(", "").replace(")", "").replace("'", "") \
                          .replace("/", "_")
            label_cols.append(col)

        out_df = df[["filepath"]].copy()
        numeric_col = task + "_filtered_numeric"
        for j, col in enumerate(label_cols):
            out_df[col] = df[numeric_col].apply(lambda x: j in x if isinstance(x, list) else False)

        # task name mapping
        task_suffix = task.replace("label_diag_superclass", "ptbxl_super") \
                          .replace("label_diag_subclass", "ptbxl_sub") \
                          .replace("label_rhythm", "ptbxl_rhythm") \
                          .replace("label_form", "ptbxl_form") \
                          .replace("label_diag", "ptbxl_diag") \
                          .replace("label_all", "ptbxl_all")

        out_csv = OUT_DIR / f"{task_suffix}_paper_labels.csv"
        out_df.to_csv(out_csv, index=False)

        out_json = OUT_DIR / f"{task_suffix}_paper_labels.json"
        with open(out_json, "w") as f:
            json.dump({
                "dataset": task_suffix,
                "n_labels": len(labels_list),
                "labels": {col: lbl for col, lbl in zip(label_cols, labels_list)},
            }, f, indent=2, ensure_ascii=False)

        logging.info(f"  Saved: {out_csv.name} ({len(out_df)} rows, {len(label_cols)} labels)")


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    targets = []
    if args.all:
        targets = list(SNOMED_DATASETS.keys()) + ["ptbxl", "zzu", "code15"]
    elif args.dataset:
        targets = [args.dataset]
    else:
        parser.print_help()
        return

    for ds in targets:
        logging.info(f"\n{'='*60}")
        logging.info(f"  {ds}")
        logging.info(f"{'='*60}")

        if ds in SNOMED_DATASETS:
            df, labels_itos = build_snomed_dataset(ds)
            save_label_csv(ds, df, labels_itos)
        elif ds == "ptbxl":
            df, lbl_itos = build_ptbxl_dataset()
            save_ptbxl_subtasks(df, lbl_itos)
        elif ds == "zzu":
            df, lbl_itos = build_zzu_dataset()
            save_zzu_subtasks(df, lbl_itos)
        elif ds == "code15":
            copy_code15_labels()

    logging.info("\ndone!")


if __name__ == "__main__":
    main()
