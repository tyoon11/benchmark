"""
Benchmark label build script
=============================
Build label CSVs matching the benchmark tasks of the ecg-fm-benchmarking paper.
WFDB .hea file from SNOMED code directly parsingand, Label Mappings xlsx by diagnosis mapping.

generate label CSV:
  physionet/{dataset}_bench_labels.csv  — SNOMED multi-label (per-dataset)
  ptbxl_bench_labels_{task}.csv         — PTB-XL subtask (super/sub/all/diag/form/rhythm)
  zzu_bench_labels.csv                  — ZZU AHA based
  code15_bench_labels.csv               — CODE-15% 6-class
  cpsc2021_bench_labels.csv             — CPSC2021 AF 3-class

run:
  python scripts/build_benchmark_labels.py --all
  python scripts/build_benchmark_labels.py --dataset ptbxl
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
BENCHMARK_DIR = Path("/path/to/workspace/benchmark")

MIN_CNT = 10  # min positive 

# per-dataset WFDB path + xlsx 
SNOMED_DATASETS = {
    "chapman":      {"wfdb_dir": CHALLENGE_BASE / "chapman_shaoxing", "sheet": "Chapman"},
    "cpsc2018":     {"wfdb_dir": CHALLENGE_BASE / "cpsc_2018",        "sheet": "CPSC"},
    "cpsc_extra":   {"wfdb_dir": CHALLENGE_BASE / "cpsc_2018_extra",  "sheet": "CPSC-Extra"},
    "georgia":      {"wfdb_dir": CHALLENGE_BASE / "georgia",          "sheet": "G12EC"},
    "ningbo":       {"wfdb_dir": CHALLENGE_BASE / "ningbo",           "sheet": "Ningbo"},
    "ptb":          {"wfdb_dir": CHALLENGE_BASE / "ptb",              "sheet": "PTB"},
    "ptbxl":        {"wfdb_dir": CHALLENGE_BASE / "ptb-xl",           "sheet": "PTBxl"},
    "stpetersburg": {"wfdb_dir": CHALLENGE_BASE / "st_petersburg_incart", "sheet": "INCART"},
}


# ═══════════════════════════════════════════════════════════════
# SNOMED label extract (physionet common)
# ═══════════════════════════════════════════════════════════════
def load_snomed_mapping(sheet_name: str) -> dict:
    """Label Mappings xlsx from SNOMED code → diagnosis name mapping load"""
    df = pd.read_excel(LABEL_XLSX, sheet_name=sheet_name, dtype={"SNOMED code": str})
    df = df.dropna(subset=["SNOMED code"])
    mapping = {}
    for _, row in df.iterrows():
        code = str(row["SNOMED code"]).strip()
        diag = str(row["Diagnosis in the dataset"]).strip()
        mapping[code] = diag
    return mapping


def parse_wfdb_dx(hea_path: str) -> list:
    """WFDB .hea file from # Dx: SNOMED code list extract"""
    try:
        rec = wfdb.rdheader(hea_path)
        for c in (rec.comments or []):
            cl = c.strip()
            if cl.lower().startswith("dx:"):
                codes = [s.strip() for s in cl.split(":", 1)[1].split(",") if s.strip()]
                return codes
    except Exception:
        pass
    return []


def build_snomed_labels(dataset_name: str, min_cnt: int = MIN_CNT):
    """
    physionet dataset's SNOMED based multi-label CSV generate.

    Returns: (DataFrame, label_cols, lbl_itos)
    """
    cfg = SNOMED_DATASETS[dataset_name]
    wfdb_dir = cfg["wfdb_dir"]
    snomed_map = load_snomed_mapping(cfg["sheet"])

    logging.info(f"  SNOMED mapping: {len(snomed_map)} ({cfg['sheet']})")

    # file_name.csv by h5 filepath → original_filename mapping
    fn_csv = H5_ROOT / "physionet/v2.0/file_name.csv"
    fn_df = pd.read_csv(fn_csv)
    fn_df = fn_df[fn_df["dataset"] == dataset_name]
    orig_to_h5fp = dict(zip(fn_df["original_filename"].astype(str),
                            fn_df["h5_filepath"].astype(str)))

    # all .hea file from SNOMED code parsing
    hea_files = sorted(glob.glob(str(wfdb_dir / "g*" / "*.hea")))
    logging.info(f"  .hea file: {len(hea_files)}")

    records = []
    for hea in hea_files:
        rec_name = os.path.basename(hea).replace(".hea", "")
        codes = parse_wfdb_dx(hea[:-4])
        diags = []
        for code in codes:
            if code in snomed_map:
                diags.append(snomed_map[code])
        h5_fp = orig_to_h5fp.get(rec_name)
        if h5_fp:
            records.append({"filepath": h5_fp, "record": rec_name, "diags": diags})

    logging.info(f"  mappinged record: {len(records)}")

    #  based label 
    diag_freq = Counter()
    for r in records:
        for d in r["diags"]:
            diag_freq[d] += 1

    selected = [d for d, cnt in diag_freq.most_common() if cnt >= min_cnt]
    logging.info(f"  label (≥{min_cnt}): {len(selected)}")

    # DataFrame 
    label_cols = [d.replace(" ", "_").replace(",", "").replace("-", "_")
                  .replace("(", "").replace(")", "").replace("'", "")
                  for d in selected]
    diag_to_col = dict(zip(selected, label_cols))

    rows = []
    for r in records:
        row = {"filepath": r["filepath"]}
        active = set(r["diags"])
        for diag, col in diag_to_col.items():
            row[col] = diag in active
        rows.append(row)

    df = pd.DataFrame(rows)

    # lbl_itos 
    lbl_itos = {col: diag for diag, col in diag_to_col.items()}

    return df, label_cols, lbl_itos


# ═══════════════════════════════════════════════════════════════
# PTB-XL subtask (SCP code based)
# ═══════════════════════════════════════════════════════════════
def build_ptbxl_subtask_labels(min_cnt: int = MIN_CNT):
    """
    PTB-XL's WFDB header from SNOMED code based label  only,
    PTB-XL before (for) subtask also SNOMED mapping from .

    return: dict of task_name → (DataFrame, label_cols)
    """
    # first, default SNOMED label generate (ptbxl all)
    df_all, all_cols, all_itos = build_snomed_labels("ptbxl", min_cnt=min_cnt)

    # PTB-XL subtask definitions — SNOMED abbreviation code based directly mapping
    # (all_itos abbreviation→abbreviation by mapping  matching  code directly )
    SUPERCLASS_CODE_MAP = {
        "NORM": ["SR"],
        "MI":   ["AMI", "PMI", "ISCIL", "ISCIN", "ISCLA", "ISCAN"],
        "STTC": ["STD_", "STE_", "NST_", "INVT", "TAB_", "STTC"],
        "CD":   ["CLBBB", "CRBBB", "IRBBB", "ILBBB", "AVB", "2AVB", "3AVB",
                 "LAFB/LPFB", "LPFB", "IVCD", "LPR", "WPW"],
        "HYP":  ["VCLVH", "RVH", "SEHYP", "LAO/LAE", "RAO/RAE", "HVOLT"],
    }

    RHYTHM_CODES = ["SR", "AFIB", "AFLT", "STACH", "SBRAD", "SARRH", "SVARR",
                    "SVTAC", "PSVT", "PAC", "PVC", "PACE"]
    FORM_CODES = ["STD_", "STE_", "NST_", "INVT", "TAB_", "STTC", "QWAVE",
                  "VCLVH", "RVH", "SEHYP", "LAO/LAE", "RAO/RAE", "HVOLT",
                  "LVOLT", "LAD", "RAD", "LNGQT"]

    # subtask per label generate
    tasks = {}

    # 1. all — all SNOMED label
    tasks["ptbxl_all"] = (df_all.copy(), all_cols)

    # 2. super — 5 superclass (abbreviation code directly matching)
    super_cols = list(SUPERCLASS_CODE_MAP.keys())
    df_super = df_all[["filepath"]].copy()
    for sclass, codes in SUPERCLASS_CODE_MAP.items():
        matching = [c for c in codes if c in all_cols]
        df_super[sclass] = df_all[matching].any(axis=1) if matching else False
    # NORM : SR  MI/STTC/CD/HYP   no  only Normal
    pathology_cols = []
    for s in ["MI", "STTC", "CD", "HYP"]:
        pathology_cols.extend([c for c in SUPERCLASS_CODE_MAP[s] if c in all_cols])
    has_pathology = df_all[pathology_cols].any(axis=1) if pathology_cols else False
    df_super["NORM"] = df_super["NORM"] & ~has_pathology
    tasks["ptbxl_super"] = (df_super, super_cols)

    # 3. rhythm —   label only
    rhythm_cols = [c for c in RHYTHM_CODES if c in all_cols]
    if rhythm_cols:
        tasks["ptbxl_rhythm"] = (df_all[["filepath"] + rhythm_cols].copy(), rhythm_cols)

    # 4. form —   label only
    form_cols = [c for c in FORM_CODES if c in all_cols]
    if form_cols:
        tasks["ptbxl_form"] = (df_all[["filepath"] + form_cols].copy(), form_cols)

    # 5. diag — diagnosis label (rhythm/form exclude)
    diag_cols = [c for c in all_cols if c not in rhythm_cols and c not in form_cols]
    if diag_cols:
        tasks["ptbxl_diag"] = (df_all[["filepath"] + diag_cols].copy(), diag_cols)

    # 6. sub — diag's class (diag and identically use)
    tasks["ptbxl_sub"] = tasks.get("ptbxl_diag", (df_all[["filepath"]].copy(), []))

    return tasks


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="benchmark label generate")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", type=str, default=None,
                        help="specific dataset(s) (chapman, ptbxl, etc)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    out_dir = BENCHMARK_DIR / "labels"
    os.makedirs(out_dir, exist_ok=True)

    if args.all:
        targets = list(SNOMED_DATASETS.keys()) + ["zzu", "code15", "cpsc2021", "sph"]
    elif args.dataset:
        targets = [args.dataset]
    else:
        parser.print_help()
        return

    for ds in targets:
        logging.info(f"\n{'='*50}")
        logging.info(f"  {ds}")
        logging.info(f"{'='*50}")

        if ds in SNOMED_DATASETS:
            df, label_cols, lbl_itos = build_snomed_labels(ds, min_cnt=MIN_CNT)
            csv_path = out_dir / f"{ds}_bench_labels.csv"
            df.to_csv(csv_path, index=False)
            logging.info(f"  save: {csv_path.name} ({len(df):,}rows, {len(label_cols)} label)")

            # lbl_itos JSON
            json_path = out_dir / f"{ds}_bench_labels.json"
            with open(json_path, "w") as f:
                json.dump({"dataset": ds, "n_labels": len(label_cols),
                           "labels": lbl_itos}, f, indent=2, ensure_ascii=False)

            # PTB-XL subtask
            if ds == "ptbxl":
                logging.info("  PTB-XL subtask generate...")
                ptbxl_tasks = build_ptbxl_subtask_labels(min_cnt=MIN_CNT)
                for task_name, (task_df, task_cols) in ptbxl_tasks.items():
                    csv_p = out_dir / f"{task_name}_bench_labels.csv"
                    task_df.to_csv(csv_p, index=False)
                    logging.info(f"    {task_name}: {csv_p.name} ({len(task_df):,}rows, {len(task_cols)} label)")
                    json_p = out_dir / f"{task_name}_bench_labels.json"
                    with open(json_p, "w") as f:
                        json.dump({"dataset": task_name, "n_labels": len(task_cols),
                                   "labels": task_cols}, f, indent=2, ensure_ascii=False)

        elif ds == "zzu":
            # existing label CSV 
            src = H5_ROOT / "ZZU-pECG/v2.0/zzu_labels.csv"
            if src.exists():
                import shutil
                dst = out_dir / "zzu_bench_labels.csv"
                shutil.copy(src, dst)
                df = pd.read_csv(dst, nrows=0)
                key = {"filepath","dataset","pid","rid","oid"}
                n = len([c for c in df.columns if c not in key])
                logging.info(f"  save: {dst.name} ({n} label)")

        elif ds == "code15":
            src = H5_ROOT / "code15/v2.0/code15_labels.csv"
            if src.exists():
                import shutil
                dst = out_dir / "code15_bench_labels.csv"
                shutil.copy(src, dst)
                logging.info(f"  save: {dst.name} (6 label)")

        elif ds == "cpsc2021":
            src = H5_ROOT / "cpsc2021/v2.0/cpsc2021_labels.csv"
            if src.exists():
                import shutil
                dst = out_dir / "cpsc2021_bench_labels.csv"
                shutil.copy(src, dst)
                logging.info(f"  save: {dst.name} (3 label)")

        elif ds == "sph":
            # SPH: convert_h5's sph_labels.csv as-is paper/bench  in .
            # convert_h5/append_labels.py's map_sph ecg-fm-benchmarking
            # map_and_filter_labels(min_cnt=10) identical 35 primary AHA code use.
            src = H5_ROOT / "sph/v2.0/sph_labels.csv"
            if src.exists():
                df = pd.read_csv(src, low_memory=False)
                key = {"filepath","dataset","pid","rid","oid"}
                label_cols = [c for c in df.columns if c not in key]
                # benchmark  filepath + label column only 
                paper_df = df[["filepath"] + label_cols]
                for suffix in ("paper", "bench"):
                    dst = out_dir / f"sph_{suffix}_labels.csv"
                    paper_df.to_csv(dst, index=False)
                    json_p = out_dir / f"sph_{suffix}_labels.json"
                    with open(json_p, "w") as f:
                        json.dump({"dataset": "sph", "n_labels": len(label_cols),
                                   "labels": label_cols}, f, indent=2, ensure_ascii=False)
                    logging.info(f"  save: {dst.name} ({len(paper_df):,}rows, {len(label_cols)} label)")
            else:
                logging.error(f"  original label none: {src} (first, append_labels.py --dataset sph)")

    logging.info(f"\ndone! label directory: {out_dir}")


if __name__ == "__main__":
    main()
