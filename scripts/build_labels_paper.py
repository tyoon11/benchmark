"""
논문 동일 라벨 생성 스크립트
==============================
ecg-fm-benchmarking의 prepare_data_*() 함수를 그대로 재현합니다.

- physionet (ningbo/cpsc2018/cpsc_extra/georgia/ptb/chapman/stpetersburg):
    WFDB .hea의 # Dx: SNOMED → Label Mappings xlsx → 진단명 (label 리스트)
- PTB-XL:
    ptbxl_database.csv + scp_statements.csv → 6 서브태스크
    (label_all, label_diag, label_form, label_rhythm,
     label_diag_subclass, label_diag_superclass)
- ZZU:
    AttributesDictionary.csv의 AHA_code/CHN_code (원본 동일)

모든 라벨은 min_cnt=10으로 필터링 (논문 동일).

출력:
  labels/{dataset}_paper_labels.csv  (key 컬럼 + binary 라벨)
  labels/{dataset}_paper_labels.json (라벨명 정의)

실행:
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
# 경로
# ═══════════════════════════════════════════════════════════════
H5_ROOT = Path("/home/irteam/ddn-opendata1/h5")
RAW_ROOT = Path("/home/irteam/ddn-opendata1/raw/physionet.org/files")
CHALLENGE_BASE = RAW_ROOT / "challenge-2021/1.0.3/training"
LABEL_XLSX = Path("/home/irteam/local-node-d/tykim/ecg-fm-benchmarking/Label mappings 2021.xlsx")
PTBXL_META = Path("/home/irteam/ddn-opendata1/raw/ptbxl_metadata")
BENCHMARK_DIR = Path("/home/irteam/local-node-d/tykim/benchmark")
OUT_DIR = BENCHMARK_DIR / "labels"

MIN_CNT = 10  # 논문 동일


# ═══════════════════════════════════════════════════════════════
# map_and_filter_labels — 논문 그대로 재현
# ═══════════════════════════════════════════════════════════════
def map_and_filter_labels(df, min_cnt, lbl_cols):
    """
    원본 ecg_utils.map_and_filter_labels() 재현.

    각 lbl_col에 대해:
      1. 모든 라벨 카운트
      2. min_cnt 이상인 라벨만 남기고 {col}_filtered 컬럼 생성
      3. {col}_filtered를 numeric으로 인코딩한 {col}_filtered_numeric 컬럼 생성

    Returns:
        df:        확장된 DataFrame
        lbl_itos:  {col_filtered: [label1, label2, ...]} — 라벨 순서
    """
    lbl_itos = {}
    for col in lbl_cols:
        # 전체 라벨 flatten
        all_labels = [item for sublist in df[col] for item in sublist]
        unique, cnt = np.unique(all_labels, return_counts=True)
        selected = set(unique[cnt >= min_cnt])

        # filtered 컬럼
        df[col + "_filtered"] = df[col].apply(lambda x: [y for y in x if y in selected])

        # numeric 매핑 (정렬된 순서)
        lbl_list = sorted(selected)
        lbl_itos[col + "_filtered"] = lbl_list
        lbl_stoi = {s: i for i, s in enumerate(lbl_list)}
        df[col + "_filtered_numeric"] = df[col + "_filtered"].apply(
            lambda x: [lbl_stoi[y] for y in x]
        )
    return df, lbl_itos


# ═══════════════════════════════════════════════════════════════
# PhysioNet SNOMED 데이터셋 (논문 prepare_data_*() 재현)
# ═══════════════════════════════════════════════════════════════
SNOMED_DATASETS = {
    "chapman":      {"wfdb_dir": CHALLENGE_BASE / "chapman_shaoxing", "sheet": "Chapman"},
    "cpsc2018":     {"wfdb_dir": CHALLENGE_BASE / "cpsc_2018",        "sheet": "CPSC"},
    "cpsc_extra":   {"wfdb_dir": CHALLENGE_BASE / "cpsc_2018_extra",  "sheet": "CPSC-Extra"},
    "georgia":      {"wfdb_dir": CHALLENGE_BASE / "georgia",          "sheet": "G12EC"},
    "ningbo":       {"wfdb_dir": CHALLENGE_BASE / "ningbo",           "sheet": "Ningbo",
                     "extra_map": {"106068003": "ARH"}},  # 논문 ningbo prepare의 수동 추가
    "ptb":          {"wfdb_dir": CHALLENGE_BASE / "ptb",              "sheet": "PTB"},
    "stpetersburg": {"wfdb_dir": CHALLENGE_BASE / "st_petersburg_incart", "sheet": "INCART"},
}


def load_snomed_mapping(sheet_name: str, extra_map: dict = None) -> dict:
    """Label Mappings xlsx 로드."""
    df = pd.read_excel(LABEL_XLSX, sheet_name=sheet_name, dtype={"SNOMED code": str})
    df = df.dropna(subset=["SNOMED code"])
    mapping = {}
    for _, row in df.iterrows():
        code = str(row["SNOMED code"]).strip()
        diag = str(row["Diagnosis in the dataset"]).strip()
        if code not in mapping:   # 논문의 경우 중복 시 첫 번째 유지
            mapping[code] = diag
    if extra_map:
        mapping.update(extra_map)
    return mapping


def parse_wfdb_dx(hea_path: str) -> list:
    """WFDB .hea에서 # Dx: SNOMED 코드 리스트 파싱 (논문 동일)."""
    try:
        rec = wfdb.rdheader(hea_path)
        for c in (rec.comments or []):
            cl = c.strip()
            if cl.lower().startswith("dx:"):
                # 논문: codes = [str(int(x)) for x in arrs[1].split(',')]
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
    논문 prepare_data_ningbo/cpsc2018/chapman 등 재현.

    - df["label"] = [SNOMED diagnosis names (from xlsx mapping)]
    - map_and_filter_labels(min_cnt=10)
    - df["label_filtered"], df["label_filtered_numeric"] 생성
    """
    cfg = SNOMED_DATASETS[dataset_name]
    wfdb_dir = cfg["wfdb_dir"]
    snomed_map = load_snomed_mapping(cfg["sheet"], cfg.get("extra_map"))

    logging.info(f"  SNOMED mapping: {len(snomed_map)} entries ({cfg['sheet']})")

    # file_name.csv로 h5 filepath 매핑
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
        # 논문: labels = [dx_mapping_snomed[code] for code in codes] (매핑 실패는 스킵)
        labels = [snomed_map[c] for c in codes if c in snomed_map]
        h5_fp = orig_to_h5fp.get(rec_name)
        if h5_fp:
            records.append({"filepath": h5_fp, "label": labels})

    logging.info(f"  Mapped records: {len(records)}")
    df = pd.DataFrame(records)

    # 논문 동일: map_and_filter_labels
    df, lbl_itos = map_and_filter_labels(df, min_cnt=MIN_CNT, lbl_cols=["label"])
    labels_itos = lbl_itos["label_filtered"]
    logging.info(f"  Labels (≥{MIN_CNT}): {len(labels_itos)}")

    return df, labels_itos


# ═══════════════════════════════════════════════════════════════
# PTB-XL — 논문 prepare_data_ptb_xl() 재현
# ═══════════════════════════════════════════════════════════════
def build_ptbxl_dataset():
    """
    논문 prepare_data_ptb_xl() 그대로 재현.

    서브태스크:
      - label_all         (모든 SCP 코드)
      - label_diag        (scp_statements의 diagnostic>0)
      - label_form        (scp_statements의 form>0)
      - label_rhythm      (scp_statements의 rhythm>0)
      - label_diag_subclass    (diag → diagnostic_subclass)
      - label_diag_superclass  (diag → diagnostic_class)
    """
    logging.info("  Loading ptbxl_database.csv + scp_statements.csv")

    ptbxl_db_path = PTBXL_META / "ptbxl_database.csv"
    scp_path = PTBXL_META / "scp_statements.csv"

    df = pd.read_csv(ptbxl_db_path, index_col="ecg_id")
    # scp_codes는 dict 문자열
    df["scp_codes"] = df["scp_codes"].apply(lambda x: eval(x.replace("nan", "np.nan")))

    scp = pd.read_csv(scp_path)
    scp = scp.set_index(scp.columns[0])

    # 논문 동일
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

    # 논문 min_cnt=10 필터링
    df, lbl_itos = map_and_filter_labels(
        df, min_cnt=MIN_CNT,
        lbl_cols=["label_all", "label_diag", "label_form", "label_rhythm",
                  "label_diag_subclass", "label_diag_superclass"],
    )

    # 원본 strat_fold도 가져옴
    # ptbxl_database의 filename_hr: records500/00000/00001_hr
    # H5의 original_filename: HR00001  (convert_to_h5.py 처리 결과)
    fn_csv = H5_ROOT / "physionet/v2.0/file_name.csv"
    fn_df = pd.read_csv(fn_csv)
    fn_df = fn_df[fn_df["dataset"] == "ptbxl"]
    orig_to_h5fp = dict(zip(fn_df["original_filename"].astype(str),
                            fn_df["h5_filepath"].astype(str)))

    # ecg_id → HR{padded} 매핑
    def ecg_id_to_hr(ecg_id):
        return f"HR{int(ecg_id):05d}"

    df["filepath"] = df.index.to_series().apply(lambda x: orig_to_h5fp.get(ecg_id_to_hr(x), None))
    df_mapped = df[df["filepath"].notna()].copy()
    logging.info(f"  PTB-XL records mapped to H5: {len(df_mapped)}/{len(df)}")

    return df_mapped, lbl_itos


# ═══════════════════════════════════════════════════════════════
# ZZU — 원본 prepare_data_zzu_pecg() 유사 (AHA/CHN 코드 기반)
# ═══════════════════════════════════════════════════════════════
def build_zzu_dataset():
    """
    ZZU: raw AttributesDictionary.csv의 AHA_code + CHN_code → label 리스트.
    논문의 prepare_data_zzu_pecg와 동일.
    """
    attr_csv = Path("/home/irteam/ddn-opendata1/raw/ZZU-pECG/AttributesDictionary.csv")
    df_attr = pd.read_csv(attr_csv)

    # Filename과 H5 filepath 매핑
    h5_table = pd.read_csv(H5_ROOT / "ZZU-pECG/v2.0/ecg_table.csv")
    # H5의 pid가 attr의 Patient_ID, rid가 attr 내 인덱스
    # 가장 쉬운 건 attr에 H5 filepath 직접 붙이기
    # attr의 Filename (예: "P00001_E01") → H5 file에는 어떻게?
    # convert_to_h5에서 prefix=zzu, pid=Patient_ID, rid=attr row index 순서로 저장됨
    # file_name.csv를 보자
    fn_csv = H5_ROOT / "ZZU-pECG/v2.0/file_name.csv"
    fn_df = pd.read_csv(fn_csv)
    # original_filename → h5_filepath
    orig_to_h5 = dict(zip(fn_df["original_filename"].astype(str),
                          fn_df["h5_filepath"].astype(str)))

    # AHA_code와 CHN_code를 합쳐서 label 리스트로
    def parse_codes(row):
        labels = []
        for col in ["AHA_code", "CHN_code"]:
            val = row.get(col)
            if pd.isna(val): continue
            for part in str(val).split(";"):
                p = part.strip().strip("'")
                if p:
                    labels.append(p)
        return labels

    df_attr["label"] = df_attr.apply(parse_codes, axis=1)

    # filepath 매핑
    df_attr["filepath"] = df_attr["Filename"].apply(
        lambda x: orig_to_h5.get(str(x).split("/")[-1], None)
    )
    df = df_attr[df_attr["filepath"].notna()].copy()
    logging.info(f"  ZZU records: {len(df)}/{len(df_attr)}")

    df, lbl_itos = map_and_filter_labels(df, min_cnt=MIN_CNT, lbl_cols=["label"])
    return df, lbl_itos["label_filtered"]


# ═══════════════════════════════════════════════════════════════
# CODE-15% — 원본 CSV 유지 (이미 논문과 동일)
# ═══════════════════════════════════════════════════════════════
def copy_code15_labels():
    """code15는 변경 없이 기존 라벨 CSV 복사"""
    import shutil
    src = OUT_DIR / "code15_bench_labels.csv"
    dst = OUT_DIR / "code15_paper_labels.csv"
    if src.exists():
        shutil.copy(src, dst)
        logging.info(f"  Copied: {dst.name}")
        return True
    return False


# ═══════════════════════════════════════════════════════════════
# DataFrame → labels CSV 변환
# ═══════════════════════════════════════════════════════════════
def save_label_csv(dataset: str, df: pd.DataFrame, labels_itos, suffix=""):
    """
    df['label_filtered_numeric'] 리스트를 binary 컬럼으로 확장하여 CSV 저장.

    Returns: 저장 경로
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"{dataset}{suffix}_paper_labels.csv"

    # label 컬럼명 sanitize
    label_cols = []
    for lbl in labels_itos:
        col = str(lbl).replace(" ", "_").replace(",", "").replace("-", "_") \
                      .replace("(", "").replace(")", "").replace("'", "") \
                      .replace("/", "_")
        label_cols.append(col)

    # binary 컬럼 만들기
    out_df = df[["filepath"]].copy()
    for j, col in enumerate(label_cols):
        out_df[col] = df["label_filtered_numeric" if "label_filtered_numeric" in df.columns else f"label_filtered_numeric"].apply(
            lambda x: j in x if isinstance(x, list) else False
        )
    out_df.to_csv(out_csv, index=False)

    # JSON 라벨 정의 저장
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
    """PTB-XL 6개 서브태스크 각각 저장"""
    for task in ["label_all", "label_diag", "label_form", "label_rhythm",
                 "label_diag_subclass", "label_diag_superclass"]:
        labels_list = lbl_itos[task + "_filtered"]

        # task별로 label_filtered_numeric 컬럼을 binary로
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

        # 태스크 이름 매핑
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
# 메인
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
            df, labels_itos = build_zzu_dataset()
            save_label_csv("zzu", df, labels_itos)
        elif ds == "code15":
            copy_code15_labels()

    logging.info("\n완료!")


if __name__ == "__main__":
    main()
