"""
벤치마크 라벨 생성 스크립트
=============================
ecg-fm-benchmarking 논문의 벤치마크 태스크에 맞는 라벨 CSV를 생성합니다.
WFDB .hea 파일에서 SNOMED 코드를 직접 파싱하고, Label Mappings xlsx로 진단명 매핑.

생성되는 라벨 CSV:
  physionet/{dataset}_bench_labels.csv  — SNOMED multi-label (데이터셋별)
  ptbxl_bench_labels_{task}.csv         — PTB-XL 서브태스크 (super/sub/all/diag/form/rhythm)
  zzu_bench_labels.csv                  — ZZU AHA 기반
  code15_bench_labels.csv               — CODE-15% 6-class
  cpsc2021_bench_labels.csv             — CPSC2021 AF 3-class

실행:
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
# 경로
# ═══════════════════════════════════════════════════════════════
H5_ROOT = Path("/home/irteam/ddn-opendata1/h5")
RAW_ROOT = Path("/home/irteam/ddn-opendata1/raw/physionet.org/files")
CHALLENGE_BASE = RAW_ROOT / "challenge-2021/1.0.3/training"
LABEL_XLSX = Path("/home/irteam/local-node-d/tykim/ecg-fm-benchmarking/Label mappings 2021.xlsx")
BENCHMARK_DIR = Path("/home/irteam/local-node-d/tykim/benchmark")

MIN_CNT = 10  # 최소 양성 수

# 데이터셋별 WFDB 경로 + xlsx 시트명
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
# SNOMED 라벨 추출 (physionet 공통)
# ═══════════════════════════════════════════════════════════════
def load_snomed_mapping(sheet_name: str) -> dict:
    """Label Mappings xlsx에서 SNOMED code → diagnosis name 매핑 로드"""
    df = pd.read_excel(LABEL_XLSX, sheet_name=sheet_name, dtype={"SNOMED code": str})
    df = df.dropna(subset=["SNOMED code"])
    mapping = {}
    for _, row in df.iterrows():
        code = str(row["SNOMED code"]).strip()
        diag = str(row["Diagnosis in the dataset"]).strip()
        mapping[code] = diag
    return mapping


def parse_wfdb_dx(hea_path: str) -> list:
    """WFDB .hea 파일에서 # Dx: SNOMED 코드 목록 추출"""
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
    physionet 데이터셋의 SNOMED 기반 multi-label CSV 생성.

    Returns: (DataFrame, label_cols, lbl_itos)
    """
    cfg = SNOMED_DATASETS[dataset_name]
    wfdb_dir = cfg["wfdb_dir"]
    snomed_map = load_snomed_mapping(cfg["sheet"])

    logging.info(f"  SNOMED mapping: {len(snomed_map)}개 ({cfg['sheet']})")

    # file_name.csv로 h5 filepath → original_filename 매핑
    fn_csv = H5_ROOT / "physionet/v2.0/file_name.csv"
    fn_df = pd.read_csv(fn_csv)
    fn_df = fn_df[fn_df["dataset"] == dataset_name]
    orig_to_h5fp = dict(zip(fn_df["original_filename"].astype(str),
                            fn_df["h5_filepath"].astype(str)))

    # 모든 .hea 파일에서 SNOMED 코드 파싱
    hea_files = sorted(glob.glob(str(wfdb_dir / "g*" / "*.hea")))
    logging.info(f"  .hea 파일: {len(hea_files)}개")

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

    logging.info(f"  맵핑된 레코드: {len(records)}개")

    # 빈도 기반 라벨 선정
    diag_freq = Counter()
    for r in records:
        for d in r["diags"]:
            diag_freq[d] += 1

    selected = [d for d, cnt in diag_freq.most_common() if cnt >= min_cnt]
    logging.info(f"  라벨 (≥{min_cnt}건): {len(selected)}개")

    # DataFrame 구축
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

    # lbl_itos 정보
    lbl_itos = {col: diag for diag, col in diag_to_col.items()}

    return df, label_cols, lbl_itos


# ═══════════════════════════════════════════════════════════════
# PTB-XL 서브태스크 (SCP 코드 기반)
# ═══════════════════════════════════════════════════════════════
def build_ptbxl_subtask_labels(min_cnt: int = MIN_CNT):
    """
    PTB-XL의 WFDB 헤더에서 SNOMED 코드 기반 라벨을 만들되,
    PTB-XL 전용 서브태스크도 SNOMED 매핑에서 유도합니다.

    반환: dict of task_name → (DataFrame, label_cols)
    """
    # 먼저 기본 SNOMED 라벨 생성 (ptbxl 전체)
    df_all, all_cols, all_itos = build_snomed_labels("ptbxl", min_cnt=min_cnt)

    # PTB-XL 서브태스크 정의 (SNOMED 진단명 기반 분류)
    SUPERCLASS_MAP = {
        "NORM": ["sinus rhythm", "normal ecg"],
        "MI":   ["myocardial infarction", "anterior myocardial infarction",
                 "lateral myocardial infarction", "inferior myocardial infarction",
                 "anteroseptal myocardial infarction", "anterolateral myocardial infarction"],
        "STTC": ["st depression", "st elevation", "t wave abnormal", "t wave inversion",
                 "nonspecific st t changes"],
        "CD":   ["right bundle branch block", "left bundle branch block",
                 "1st degree av block", "left anterior fascicular block",
                 "incomplete right bundle branch block", "left posterior fascicular block",
                 "complete right bundle branch block"],
        "HYP":  ["left ventricular hypertrophy", "right ventricular hypertrophy",
                 "left atrial enlargement", "right atrial enlargement"],
    }

    RHYTHM_KEYWORDS = ["sinus", "rhythm", "tachycardia", "bradycardia",
                       "fibrillation", "flutter", "atrial", "junctional"]
    FORM_KEYWORDS = ["wave", "st ", "qrs", "qt ", "axis", "voltage",
                     "hypertrophy", "enlargement"]

    # 서브태스크별 라벨 생성
    tasks = {}

    # 1. all — 전체 SNOMED 라벨
    tasks["ptbxl_all"] = (df_all.copy(), all_cols)

    # 2. super — 5 superclass
    super_cols = list(SUPERCLASS_MAP.keys())
    df_super = df_all[["filepath"]].copy()
    for sclass, keywords in SUPERCLASS_MAP.items():
        # all_itos에서 키워드가 포함된 컬럼 찾기
        matching = []
        for col in all_cols:
            diag_name = all_itos.get(col, col).lower()
            for kw in keywords:
                if kw in diag_name:
                    matching.append(col)
                    break
        df_super[sclass] = df_all[matching].any(axis=1) if matching else False
    tasks["ptbxl_super"] = (df_super, super_cols)

    # 3. rhythm — 리듬 관련 라벨만
    rhythm_cols = []
    for col in all_cols:
        diag_name = all_itos.get(col, col).lower()
        if any(kw in diag_name for kw in RHYTHM_KEYWORDS):
            rhythm_cols.append(col)
    if rhythm_cols:
        tasks["ptbxl_rhythm"] = (df_all[["filepath"] + rhythm_cols].copy(), rhythm_cols)

    # 4. form — 형태 관련 라벨만
    form_cols = []
    for col in all_cols:
        diag_name = all_itos.get(col, col).lower()
        if any(kw in diag_name for kw in FORM_KEYWORDS):
            form_cols.append(col)
    if form_cols:
        tasks["ptbxl_form"] = (df_all[["filepath"] + form_cols].copy(), form_cols)

    # 5. diag — 진단 라벨 (rhythm/form 제외)
    diag_cols = [c for c in all_cols if c not in rhythm_cols and c not in form_cols]
    if diag_cols:
        tasks["ptbxl_diag"] = (df_all[["filepath"] + diag_cols].copy(), diag_cols)

    # 6. sub — diag의 서브클래스 (diag와 동일하게 사용)
    tasks["ptbxl_sub"] = tasks.get("ptbxl_diag", (df_all[["filepath"]].copy(), []))

    return tasks


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="벤치마크 라벨 생성")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dataset", type=str, default=None,
                        help="특정 데이터셋 (chapman, ptbxl, etc)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    out_dir = BENCHMARK_DIR / "labels"
    os.makedirs(out_dir, exist_ok=True)

    if args.all:
        targets = list(SNOMED_DATASETS.keys()) + ["zzu", "code15", "cpsc2021"]
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
            logging.info(f"  저장: {csv_path.name} ({len(df):,}행, {len(label_cols)} 라벨)")

            # lbl_itos JSON
            json_path = out_dir / f"{ds}_bench_labels.json"
            with open(json_path, "w") as f:
                json.dump({"dataset": ds, "n_labels": len(label_cols),
                           "labels": lbl_itos}, f, indent=2, ensure_ascii=False)

            # PTB-XL 서브태스크
            if ds == "ptbxl":
                logging.info("  PTB-XL 서브태스크 생성...")
                ptbxl_tasks = build_ptbxl_subtask_labels(min_cnt=MIN_CNT)
                for task_name, (task_df, task_cols) in ptbxl_tasks.items():
                    csv_p = out_dir / f"{task_name}_bench_labels.csv"
                    task_df.to_csv(csv_p, index=False)
                    logging.info(f"    {task_name}: {csv_p.name} ({len(task_df):,}행, {len(task_cols)} 라벨)")
                    json_p = out_dir / f"{task_name}_bench_labels.json"
                    with open(json_p, "w") as f:
                        json.dump({"dataset": task_name, "n_labels": len(task_cols),
                                   "labels": task_cols}, f, indent=2, ensure_ascii=False)

        elif ds == "zzu":
            # 기존 라벨 CSV 복사
            src = H5_ROOT / "ZZU-pECG/v2.0/zzu_labels.csv"
            if src.exists():
                import shutil
                dst = out_dir / "zzu_bench_labels.csv"
                shutil.copy(src, dst)
                df = pd.read_csv(dst, nrows=0)
                key = {"filepath","dataset","pid","rid","oid"}
                n = len([c for c in df.columns if c not in key])
                logging.info(f"  저장: {dst.name} ({n} 라벨)")

        elif ds == "code15":
            src = H5_ROOT / "code15/v2.0/code15_labels.csv"
            if src.exists():
                import shutil
                dst = out_dir / "code15_bench_labels.csv"
                shutil.copy(src, dst)
                logging.info(f"  저장: {dst.name} (6 라벨)")

        elif ds == "cpsc2021":
            src = H5_ROOT / "cpsc2021/v2.0/cpsc2021_labels.csv"
            if src.exists():
                import shutil
                dst = out_dir / "cpsc2021_bench_labels.csv"
                shutil.copy(src, dst)
                logging.info(f"  저장: {dst.name} (3 라벨)")

    logging.info(f"\n완료! 라벨 디렉토리: {out_dir}")


if __name__ == "__main__":
    main()
