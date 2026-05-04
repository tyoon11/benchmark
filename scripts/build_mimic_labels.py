"""
MIMIC-IV-ECG 벤치마크 라벨 생성 스크립트
=============================================
원본 ecg-fm-benchmarking/mimic_preprocessing.py 및
ecg-fm-benchmarking/code/clinical_ts/utils/mimic_ecg_preprocessing.py
1:1 재현.

Paper Table의 11개 MIMIC 태스크:

  ✅ Cardiac discharge diagnoses     (records_w_diag_icd10.csv, ICD-10 chapter IX)
  ✅ Non-cardiac discharge diagnoses (records_w_diag_icd10.csv, 그 외)
  ✅ Sex (binary)                    (records_w_diag_icd10.csv)
  ✅ Age (regression)                (records_w_diag_icd10.csv)
  ✅ ECG features (regression, 7)    (machine_measurements.csv)
  ✅ Clinical deterioration (6)      (mds_ed.csv)
  ✅ Mortality (7-horizon)           (mds_ed.csv)
  ✅ ICU admission (2)               (mds_ed.csv)
  ✅ Biometrics (3)                  (omr.csv.gz + chartevents.csv.gz)
  ✅ Vital signs (6)                 (vitalsign.csv.gz + chartevents.csv.gz)
  ✅ Lab values (18)                 (labevents.csv.gz + d_labitems.csv.gz + chartevents.csv.gz)

필요한 raw 파일 (모두 PhysioNet credentialed):
  /raw/physionet.org/files/
    ├── mimic-iv-ecg/1.0/machine_measurements.csv
    ├── mimic-iv-ecg-ext-icd-labels/1.0.1/records_w_diag_icd10.csv
    ├── mimic-iv-ed/2.2/ed/{vitalsign,edstays}.csv.gz
    ├── mimiciv/3.1/hosp/{omr,labevents,d_labitems,admissions}.csv.gz
    ├── mimiciv/3.1/icu/{chartevents,d_items,icustays}.csv.gz
    └── multimodal-emergency-benchmark/1.0.0/mds_ed.csv

출력 (모두 labels/mimic_<task>_paper_labels.{csv,json}):
  cardiac, noncardiac, sex, age, ecg_features
  deterioration, mortality, icu_admission
  biometrics, vitals, labvalues

실행:
  python scripts/build_mimic_labels.py --all
  python scripts/build_mimic_labels.py --task biometrics

캐시:
  labels/_cache/chartevents_filtered.csv 가 ~30GB chartevents 청크 필터 결과를
  보존 (3개 task 공유). 재실행 시 자동 재사용.
"""
import argparse
import json
import logging
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════
# 경로
# ═══════════════════════════════════════════════════════════════
H5_ROOT = Path("/home/irteam/ddn-opendata1/h5/mimic4/v2.0")
RAW_BASE = Path("/home/irteam/ddn-opendata1/raw/physionet.org/files")
ICD_CSV = RAW_BASE / "mimic-iv-ecg-ext-icd-labels/1.0.1/records_w_diag_icd10.csv"
MM_CSV = RAW_BASE / "mimic-iv-ecg/1.0/machine_measurements.csv"
ADM_CSV = RAW_BASE / "mimiciv/3.1/hosp/admissions.csv.gz"
ED_STAYS_CSV = RAW_BASE / "mimic-iv-ed/2.2/ed/edstays.csv.gz"
MDS_ED_CSV = RAW_BASE / "multimodal-emergency-benchmark/1.0.0/mds_ed.csv"

# Biometrics / Vitals / Labs (paper Table 8-10)
OMR_CSV = RAW_BASE / "mimiciv/3.1/hosp/omr.csv.gz"
VITAL_CSV = RAW_BASE / "mimic-iv-ed/2.2/ed/vitalsign.csv.gz"
LABEVENTS_CSV = RAW_BASE / "mimiciv/3.1/hosp/labevents.csv.gz"
D_LABITEMS_CSV = RAW_BASE / "mimiciv/3.1/hosp/d_labitems.csv.gz"
CHARTEVENTS_CSV = RAW_BASE / "mimiciv/3.1/icu/chartevents.csv.gz"
D_ITEMS_CSV = RAW_BASE / "mimiciv/3.1/icu/d_items.csv.gz"

# 캐시 (chartevents 1회 필터 후 재사용)
CACHE_DIR = Path("/home/irteam/local-node-d/tykim/benchmark/labels/_cache")
CHARTEVENTS_FILTERED = CACHE_DIR / "chartevents_filtered.csv"

# 라벨 정의 (paper / mimic_preprocessing.py 동일)
BIOMETRIC_COLS = ["Height (Inches)", "Weight (Lbs)", "BMI (kg/m2)"]
VITAL_COLS = ["dbp", "heartrate", "o2sat", "resprate", "sbp", "temperature"]
LAB_COLS = [
    "PT", "Albumin", "Anion Gap", "Bicarbonate", "Bilirubin, Total",
    "Calcium, Total", "Creatinine", "Ferritin", "Urea Nitrogen",
    "Hematocrit", "Hemoglobin", "Lymphocytes", "MCHC", "RDW",
    "Red Blood Cells", "RDW-SD", "Creatine Kinase (CK)", "NTproBNP",
]

# d_labitems itemid 화이트리스트 (mimic_preprocessing.py:126-128 그대로)
LAB_ITEMIDS = [
    50963, 51006, 52647, 50811, 51222, 51640, 50912, 52546, 50924, 50912,
    52546, 51221, 51480, 51638, 51639, 52028, 50862, 53085, 51006, 52647,
    52172, 50811, 51222, 51640, 50868, 52500, 51277, 50882, 50885, 53089,
    51221, 51480, 51638, 51639, 52028, 51237, 51675, 51279, 51274, 52921,
    50910, 51249, 50893, 51244,
]

# chartevents에서 추출할 라벨 (mimic_preprocessing.py:208-215)
CHARTEVENTS_EXTRACT_LABELS = [
    "Height (cm)", "Height", "Daily Weight",
    "Admission Weight (lbs.)", "Admission Weight (Kg)",
    "Temperature Celsius", "Temperature Fahrenheit",
    "Heart Rate", "Respiratory Rate",
    "PAR-Oxygen saturation", "O2 saturation pulseoxymetry",
    "Albumin", "Anion Gap", "Total Bilirubin",
    "Creatinine (serum)", "Hematocrit (serum)", "Hemoglobin",
]

# MDS-ED 컬럼 정의 (paper Table와 동일한 분류)
MDS_DETERIORATION_COLS = [
    "deterioration_severe_hypoxemia",
    "deterioration_ecmo",
    "deterioration_vasopressors",
    "deterioration_inotropes",
    "deterioration_mechanical_ventilation",
    "deterioration_cardiac_arrest",
]
MDS_MORTALITY_COLS = [
    "deterioration_mortality_1d",
    "deterioration_mortality_7d",
    "deterioration_mortality_28d",
    "deterioration_mortality_90d",
    "deterioration_mortality_180d",
    "deterioration_mortality_365d",
    "deterioration_mortality_stay",
]
MDS_ICU_COLS = [
    "deterioration_icu_24h",
    "deterioration_icu_stay",
]

OUT_DIR = Path("/home/irteam/local-node-d/tykim/benchmark/labels")

# 원본 paper 설정 (mimic_preprocessing.py)
FINETUNE_DATASET = "mimic_ed_all_edfirst_all_2000_5A"
MIN_CNT = 2000      # 진단 라벨 최소 양성 수
DIGITS = 5          # ICD-10 truncate digits
PROPAGATE_ALL = True  # 5A 모드 — 조상 노드 모두 추가


# ═══════════════════════════════════════════════════════════════
# H5 매핑
# ═══════════════════════════════════════════════════════════════
def load_h5_mapping():
    """study_id → h5_filepath 매핑 (799,929개)."""
    fn = pd.read_csv(H5_ROOT / "file_name.csv")
    fn["original_record_name"] = fn["original_record_name"].astype(int)
    return dict(zip(fn["original_record_name"], fn["h5_filepath"]))


# ═══════════════════════════════════════════════════════════════
# ICD 진단 — 원본 prepare_mimic_ecg() 재현
# ═══════════════════════════════════════════════════════════════
def prepare_consistency_mapping(codes_unique, codes_unique_all, propagate_all=False):
    res = {}
    for c in codes_unique:
        if propagate_all:
            res[c] = [c[:i] for i in range(3, len(c) + 1)]
        else:
            res[c] = list(np.intersect1d([c[:i] for i in range(3, len(c) + 1)], codes_unique_all))
    return res


def get_chapter_prefix(icd_code):
    """ICD-10 chapter 매핑 (단순화 — 알파벳 prefix 기준).

    Paper에서 'cardiac chapter IX' = 순환계통 = 'I'로 시작하는 코드.
    icd10 패키지 의존을 피하기 위해 첫 글자 prefix 매핑 사용.

    ICD-10 chapter:
      I00-I99 = Chapter IX (순환계통, cardiac)
      나머지 = non-cardiac
    """
    if not icd_code or not isinstance(icd_code, str) or len(icd_code) == 0:
        return "unknown"
    return icd_code[0].upper()


def parse_diag_lists(df, cols):
    """문자열로 저장된 list를 실제 list로 파싱."""
    for c in cols:
        df[c] = df[c].apply(lambda x: eval(x) if isinstance(x, str) else [])
    return df


def prepare_diagnostic_labels(df_diags, label_col="all_diag_all",
                               min_cnt=MIN_CNT, digits=DIGITS,
                               propagate_all=PROPAGATE_ALL):
    """원본 prepare_mimic_ecg()의 라벨 추출 부분 재현.

    1. ICD-10 코드를 digits 자릿수로 truncate
    2. trailing X 제거
    3. propagate_all: 모든 조상 코드 (3~len) 추가
    4. min_cnt 이상인 코드만 유지

    Returns:
        df: label_train 컬럼이 추가된 DataFrame
        lbl_itos: 살아남은 라벨 리스트 (정렬된 순서)
    """
    df = df_diags.copy()
    df["label_train"] = df[label_col].apply(
        lambda x: list({y.strip()[:digits].rstrip("X") for y in x})
    )

    # propagate ancestors
    flat = [c for sub in df["label_train"] for c in sub]
    cons_map = prepare_consistency_mapping(np.unique(flat), np.unique(flat), propagate_all)
    df["label_train"] = df["label_train"].apply(
        lambda x: list({a for c in x for a in cons_map.get(c, [c])})
    )

    # filter min_cnt
    flat = [c for sub in df["label_train"] for c in sub]
    codes, counts = np.unique(flat, return_counts=True)
    idxs = np.argsort(counts)[::-1]
    codes = codes[idxs]
    counts = counts[idxs]
    codes_kept = codes[counts >= min_cnt]

    lbl_itos = sorted(codes_kept.tolist())
    df["label_train"] = df["label_train"].apply(lambda x: [v for v in x if v in set(lbl_itos)])
    return df, lbl_itos


# ═══════════════════════════════════════════════════════════════
# Output writer
# ═══════════════════════════════════════════════════════════════
def sanitize(s):
    """라벨 → CSV 컬럼명."""
    return (str(s).replace(" ", "_").replace(",", "")
            .replace("-", "_").replace("(", "").replace(")", "")
            .replace("'", "").replace("/", "_").replace(":", "_")
            .replace(".", "_"))


def save_multilabel_csv(name, df, label_col, lbl_itos, source_desc):
    """multi-label binary CSV 저장."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"mimic_{name}_paper_labels.csv"
    out_json = OUT_DIR / f"mimic_{name}_paper_labels.json"

    label_cols = [sanitize(l) for l in lbl_itos]
    label_set = set(lbl_itos)

    out_df = df[["filepath"]].copy()
    for j, col in enumerate(label_cols):
        target = lbl_itos[j]
        out_df[col] = df[label_col].apply(lambda x: target in x if isinstance(x, list) else False)
    out_df.to_csv(out_csv, index=False)

    with open(out_json, "w") as f:
        json.dump({
            "dataset": f"mimic_{name}",
            "source": source_desc,
            "n_labels": len(lbl_itos),
            "labels": {col: lbl for col, lbl in zip(label_cols, lbl_itos)},
        }, f, indent=2, ensure_ascii=False)
    logging.info(f"  Saved: {out_csv.name} ({len(out_df):,} rows, {len(label_cols)} labels)")


def save_binary_csv(name, df, label_col, label_name, source_desc):
    """단일 binary 컬럼 CSV 저장."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"mimic_{name}_paper_labels.csv"
    out_json = OUT_DIR / f"mimic_{name}_paper_labels.json"
    out_df = df[["filepath"]].copy()
    out_df[label_name] = df[label_col].astype(bool)
    out_df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump({
            "dataset": f"mimic_{name}",
            "source": source_desc,
            "task_type": "binary",
            "n_labels": 1,
            "labels": {label_name: label_name},
        }, f, indent=2, ensure_ascii=False)
    logging.info(f"  Saved: {out_csv.name} ({len(out_df):,} rows, 1 label, "
                 f"{out_df[label_name].sum():,} positive)")


def save_multilabel_numeric_csv(name, df, label_cols, source_desc, label_descriptions=None):
    """다변량 binary multi-label CSV 저장 (NaN 보존, 0/1 numeric)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"mimic_{name}_paper_labels.csv"
    out_json = OUT_DIR / f"mimic_{name}_paper_labels.json"
    out_df = df[["filepath"] + label_cols].copy()
    out_df.to_csv(out_csv, index=False)
    descs = label_descriptions or {c: c for c in label_cols}
    with open(out_json, "w") as f:
        json.dump({
            "dataset": f"mimic_{name}",
            "source": source_desc,
            "task_type": "multi-label-binary",
            "n_labels": len(label_cols),
            "labels": descs,
        }, f, indent=2, ensure_ascii=False)
    pos_counts = {c: int(out_df[c].fillna(0).astype(bool).sum()) for c in label_cols}
    logging.info(f"  Saved: {out_csv.name} ({len(out_df):,} rows, {len(label_cols)} labels, "
                 f"positive per col: {pos_counts})")


def save_regression_csv(name, df, label_cols, source_desc):
    """다변량 regression CSV 저장 (값은 raw, NaN 보존)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"mimic_{name}_paper_labels.csv"
    out_json = OUT_DIR / f"mimic_{name}_paper_labels.json"
    out_df = df[["filepath"] + label_cols].copy()
    out_df.to_csv(out_csv, index=False)
    with open(out_json, "w") as f:
        json.dump({
            "dataset": f"mimic_{name}",
            "source": source_desc,
            "task_type": "regression",
            "n_labels": len(label_cols),
            "labels": {c: c for c in label_cols},
        }, f, indent=2, ensure_ascii=False)
    logging.info(f"  Saved: {out_csv.name} ({len(out_df):,} rows, "
                 f"{len(label_cols)} regression targets)")


# ═══════════════════════════════════════════════════════════════
# Tasks
# ═══════════════════════════════════════════════════════════════
def build_diagnostic_tasks(study_to_h5):
    """Cardiac / Non-cardiac discharge diagnoses (multi-label)."""
    logging.info("\n=== Diagnostic (cardiac / non-cardiac) ===")
    df = pd.read_csv(ICD_CSV, low_memory=False)
    df = parse_diag_lists(df, ["all_diag_all", "ed_diag_ed", "ed_diag_hosp",
                                "hosp_diag_hosp", "all_diag_hosp"])
    logging.info(f"  ICD csv 총 {len(df):,} rows")

    # 원본 finetune_dataset='mimic_ed_all_edfirst_all_2000_5A':
    #   subsettrain=ed (ED ECGs only), labelsettrain=all
    #   subsettest=edfirst (first ECG per ED stay), labelsettest=all
    # 라벨 추출은 ED 진단 전체에 대해 수행 (paper와 동일).
    df_ed = df[df["ecg_taken_in_ed"] == True].copy()
    logging.info(f"  ECG taken in ED: {len(df_ed):,}")

    df_lbl, lbl_itos = prepare_diagnostic_labels(df_ed, label_col="all_diag_all")
    logging.info(f"  Labels (≥{MIN_CNT}): {len(lbl_itos)}")

    # cardiac (chapter IX) vs non-cardiac (other chapters)
    cardiac = [c for c in lbl_itos if get_chapter_prefix(c) == "I"]
    noncardiac = [c for c in lbl_itos if get_chapter_prefix(c) != "I"]
    logging.info(f"  Cardiac (chapter IX, 'I' prefix): {len(cardiac)}")
    logging.info(f"  Non-cardiac:                       {len(noncardiac)}")

    # 통계가 없는 ECG (label_train empty) 제외 — 원본 has_statements_train==True
    df_lbl["has_label"] = df_lbl["label_train"].apply(lambda x: len(x) > 0)

    # filepath 매핑
    df_lbl["filepath"] = df_lbl["study_id"].apply(
        lambda x: study_to_h5.get(int(x)) if pd.notna(x) else None
    )
    df_lbl = df_lbl[df_lbl["filepath"].notna()].copy()
    logging.info(f"  H5 매핑 후: {len(df_lbl):,}")

    # cardiac 출력
    df_card = df_lbl.copy()
    df_card["label_card"] = df_card["label_train"].apply(
        lambda x: [c for c in x if c in set(cardiac)]
    )
    df_card_filt = df_card[df_card["label_card"].apply(len) > 0].copy()
    logging.info(f"  Cardiac task rows (≥1 cardiac code): {len(df_card_filt):,}")
    save_multilabel_csv("cardiac", df_card_filt, "label_card", cardiac,
                        source_desc=f"records_w_diag_icd10.csv → ICD-10 chapter IX (I-prefix), "
                                    f"truncate {DIGITS} digits, propagate ancestors, min_cnt={MIN_CNT}")

    # non-cardiac 출력
    df_nc = df_lbl.copy()
    df_nc["label_nc"] = df_nc["label_train"].apply(
        lambda x: [c for c in x if c in set(noncardiac)]
    )
    df_nc_filt = df_nc[df_nc["label_nc"].apply(len) > 0].copy()
    logging.info(f"  Non-cardiac task rows (≥1 non-cardiac code): {len(df_nc_filt):,}")
    save_multilabel_csv("noncardiac", df_nc_filt, "label_nc", noncardiac,
                        source_desc=f"records_w_diag_icd10.csv → ICD-10 non-circulatory chapters, "
                                    f"truncate {DIGITS} digits, propagate ancestors, min_cnt={MIN_CNT}")


def build_sex_age_tasks(study_to_h5):
    """Sex (binary), Age (regression) — records_w_diag_icd10.csv."""
    logging.info("\n=== Sex / Age (patient characteristics) ===")
    df = pd.read_csv(ICD_CSV, low_memory=False,
                     usecols=["study_id", "gender", "age"])
    df["filepath"] = df["study_id"].apply(
        lambda x: study_to_h5.get(int(x)) if pd.notna(x) else None
    )
    df = df[df["filepath"].notna()].copy()

    # sex
    df_sex = df[df["gender"].isin(["M", "F"])].copy()
    df_sex["sex"] = (df_sex["gender"] == "M").astype(int)
    save_binary_csv("sex", df_sex, "sex", "is_male",
                    source_desc="records_w_diag_icd10.csv (gender=M→1, F→0)")

    # age
    df_age = df[df["age"].notna()].copy()
    save_regression_csv("age", df_age, ["age"],
                        source_desc="records_w_diag_icd10.csv (age in years at ECG time)")


def build_ecg_features_task(study_to_h5):
    """ECG features (regression, 7) — machine_measurements.csv."""
    logging.info("\n=== ECG features (machine_measurements) ===")
    df = pd.read_csv(MM_CSV, low_memory=False)
    logging.info(f"  machine_measurements 총 {len(df):,}")

    # 원본 outlier 처리 (mimic_preprocessing.py:92-100)
    for col in ["qrs_axis", "t_axis", "p_axis"]:
        df.loc[(df[col] < -360) | (df[col] > 360), col] = np.nan
    for col in ["p_onset", "p_end", "qrs_onset", "qrs_end", "t_end", "rr_interval"]:
        df.loc[(df[col] < 0) | (df[col] > 5000), col] = np.nan

    # 원본 파생 (RR/PR/QRS/QT/QTc 계산 — mimic_preprocessing.py:101-109)
    df = df.rename(columns={"rr_interval": "RR", "p_axis": "P_wave_axis",
                             "qrs_axis": "QRS_axis", "t_axis": "T_wave_axis"})
    df["PR"] = df["qrs_onset"] - df["p_onset"]
    df["QRS"] = df["qrs_end"] - df["qrs_onset"]
    df["QT"] = df["t_end"] - df["qrs_onset"]
    df["QTc"] = np.where(df["RR"] != 0, df["QT"] / np.sqrt(df["RR"] / 1000), np.nan)

    # paper의 7개 feature
    feat_cols = ["RR", "QRS", "QT", "QTc", "P_wave_axis", "QRS_axis", "T_wave_axis"]

    df["filepath"] = df["study_id"].apply(
        lambda x: study_to_h5.get(int(x)) if pd.notna(x) else None
    )
    df = df[df["filepath"].notna()].copy()

    # 모든 feature가 NaN인 행 제거
    df = df.dropna(subset=feat_cols, how="all").copy()
    logging.info(f"  H5 매핑 + 최소 1개 feature 보유: {len(df):,}")

    save_regression_csv("ecg_features", df, feat_cols,
                        source_desc="machine_measurements.csv (RR/QRS/QT/QTc/P_axis/QRS_axis/T_axis), "
                                    "outlier 제거 후 raw 값")


def _load_mds_ed_with_filepath(study_to_h5, value_cols):
    """MDS-ED CSV 로드 + study_id → h5 filepath 매핑.

    원본 mimic_preprocessing.py:
      - -999. 을 np.nan으로 변경 (line 75)
      - general_data, general_strat_fold, general_subject_id 사용
    여기서는 general_study_id로 H5 매핑.
    """
    df = pd.read_csv(MDS_ED_CSV, low_memory=False)
    keep = ["general_study_id", "general_subject_id", "general_strat_fold"] + value_cols
    df = df[keep].copy()
    for c in value_cols:
        df[c] = df[c].replace(-999., np.nan)
    df["filepath"] = df["general_study_id"].apply(
        lambda x: study_to_h5.get(int(x)) if pd.notna(x) else None
    )
    df = df[df["filepath"].notna()].copy()
    return df


def build_deterioration_task(study_to_h5):
    """Clinical deterioration — paper Table 5,577 × 6 outputs.

    원본 mimic_preprocessing.py 라인 67-84 재현.
    MDS-ED의 6개 deterioration 이벤트 (mortality/ICU 제외).
    """
    logging.info("\n=== Clinical deterioration (MDS-ED 6 outputs) ===")
    df = _load_mds_ed_with_filepath(study_to_h5, MDS_DETERIORATION_COLS)
    logging.info(f"  H5 매핑 후: {len(df):,}")

    descs = {
        c: c.replace("deterioration_", "").replace("_", " ")
        for c in MDS_DETERIORATION_COLS
    }
    save_multilabel_numeric_csv(
        "deterioration", df, MDS_DETERIORATION_COLS,
        source_desc="multimodal-emergency-benchmark/1.0.0/mds_ed.csv 의 6 deterioration columns "
                    "(severe_hypoxemia, ecmo, vasopressors, inotropes, mechanical_ventilation, cardiac_arrest), "
                    "원본 mimic_preprocessing.py:67-84 재현.",
        label_descriptions=descs,
    )


def build_mortality_task(study_to_h5):
    """Mortality — paper Table 17,639 × 7 outputs (multi-horizon).

    원본은 MDS-ED의 7개 mortality column.
    이전 1-class 단순화 버전을 paper 정확본으로 교체.
    """
    logging.info("\n=== Mortality (MDS-ED 7-horizon) ===")
    df = _load_mds_ed_with_filepath(study_to_h5, MDS_MORTALITY_COLS)
    logging.info(f"  H5 매핑 후: {len(df):,}")
    descs = {
        c: c.replace("deterioration_mortality_", "mortality_") for c in MDS_MORTALITY_COLS
    }
    save_multilabel_numeric_csv(
        "mortality", df, MDS_MORTALITY_COLS,
        source_desc="multimodal-emergency-benchmark/1.0.0/mds_ed.csv 의 7 mortality horizons "
                    "(1d/7d/28d/90d/180d/365d/stay).",
        label_descriptions=descs,
    )


def build_icu_admission_task(study_to_h5):
    """ICU admission — paper Table 18,690 × 2 outputs.

    원본은 MDS-ED의 2개 ICU column (icu_24h, icu_stay).
    이전 hospital admission proxy 버전을 paper 정확본으로 교체.
    """
    logging.info("\n=== ICU admission (MDS-ED 2 outputs) ===")
    df = _load_mds_ed_with_filepath(study_to_h5, MDS_ICU_COLS)
    logging.info(f"  H5 매핑 후: {len(df):,}")
    descs = {c: c.replace("deterioration_", "") for c in MDS_ICU_COLS}
    save_multilabel_numeric_csv(
        "icu_admission", df, MDS_ICU_COLS,
        source_desc="multimodal-emergency-benchmark/1.0.0/mds_ed.csv 의 2 ICU columns "
                    "(icu_24h: 24시간 내 ICU 입실, icu_stay: 동일 stay 내 ICU 입실).",
        label_descriptions=descs,
    )


def _try_read_edstays():  # 사용 중단: ICU admission이 MDS-ED로 이동
    return None
def _legacy_try_read_edstays():
    """edstays.csv.gz는 사용자 raw에서 truncated일 수 있어 부분 읽기 시도."""
    try:
        return pd.read_csv(ED_STAYS_CSV, usecols=["stay_id", "disposition"])
    except (EOFError, OSError) as e:
        logging.warning(f"  edstays.csv.gz 부분 읽기 시도 (gzip 손상 가능): {e}")
        # 부분 읽기 시도 — gzip 손상 시 chunked로 읽으면 일부라도 회수 가능
        chunks = []
        try:
            for chunk in pd.read_csv(ED_STAYS_CSV, usecols=["stay_id", "disposition"],
                                     chunksize=50_000):
                chunks.append(chunk)
        except (EOFError, OSError):
            pass
        if chunks:
            partial = pd.concat(chunks, ignore_index=True)
            logging.warning(f"  edstays partial: {len(partial):,} rows 회수")
            return partial
        return pd.DataFrame(columns=["stay_id", "disposition"])


# build_icu_admission_task은 위 MDS-ED 기반 정확본으로 이동.


# ═══════════════════════════════════════════════════════════════
# chartevents 필터링 (Biometrics / Vitals / Labs 공통 전처리)
# 원본 mimic_preprocessing.py:158-273 재현. ~30GB 청크 스트리밍.
# ═══════════════════════════════════════════════════════════════
def _prepare_chartevents_filtered(subject_ids):
    """chartevents.csv.gz를 청크로 읽어 필터·캐시.

    원본 mimic_preprocessing.py:158-205:
      1. d_items로 itemid→label 매핑
      2. chunksize=1M으로 chartevents 스트리밍
      3. subject_id가 df에 있는 row만 유지
      4. label != 'Safety Measures'
      5. 라벨별 총 count >= 1000인 라벨만 유지
      6. CACHE_DIR/chartevents_filtered.csv 에 누적 저장

    재실행 시 캐시가 있으면 건너뜀.
    """
    if CHARTEVENTS_FILTERED.exists():
        logging.info(f"  chartevents 필터 캐시 사용: {CHARTEVENTS_FILTERED}")
        return
    if not CHARTEVENTS_CSV.exists() or not D_ITEMS_CSV.exists():
        logging.warning(f"  chartevents.csv.gz 또는 d_items.csv.gz 없음 — 보강 skip")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    d_items = pd.read_csv(D_ITEMS_CSV, compression="gzip", low_memory=False)
    d_items_subset = d_items[["itemid", "label"]]
    chunksize = 1_000_000
    min_label_count = 1000

    # PASS 1: 라벨별 총 count
    logging.info("  chartevents PASS 1 — 라벨 카운트 (1M chunks)")
    label_counts = {}
    for chunk in pd.read_csv(CHARTEVENTS_CSV, compression="gzip",
                              low_memory=True, chunksize=chunksize):
        chunk = chunk[chunk["subject_id"].isin(subject_ids)]
        if chunk.empty:
            continue
        chunk = chunk.merge(d_items_subset, on="itemid", how="left")
        chunk = chunk[chunk["label"] != "Safety Measures"]
        for label, count in chunk["label"].value_counts().items():
            label_counts[label] = label_counts.get(label, 0) + count
    labels_to_keep = {l for l, c in label_counts.items() if c >= min_label_count}
    logging.info(f"  유지될 라벨 수: {len(labels_to_keep)}")

    # PASS 2: 필터 후 디스크 누적
    logging.info("  chartevents PASS 2 — 필터 후 디스크 저장")
    if CHARTEVENTS_FILTERED.exists():
        CHARTEVENTS_FILTERED.unlink()
    for chunk in pd.read_csv(CHARTEVENTS_CSV, compression="gzip",
                              low_memory=True, chunksize=chunksize):
        chunk = chunk[chunk["subject_id"].isin(subject_ids)]
        if chunk.empty:
            continue
        chunk = chunk.merge(d_items_subset, on="itemid", how="left")
        chunk = chunk[(chunk["label"] != "Safety Measures") &
                       (chunk["label"].isin(labels_to_keep))]
        if not chunk.empty:
            chunk.to_csv(CHARTEVENTS_FILTERED, mode="a",
                         header=not CHARTEVENTS_FILTERED.exists(), index=False)
    logging.info(f"  chartevents 캐시 저장: {CHARTEVENTS_FILTERED}")


def _load_chartevents_extract():
    """필터된 chartevents에서 to_extract 라벨만 로드 + 단위 변환.

    원본 mimic_preprocessing.py:218-273 재현.
    """
    if not CHARTEVENTS_FILTERED.exists():
        return None
    chunksize = 1_000_000
    dfs = []
    for chunk in pd.read_csv(CHARTEVENTS_FILTERED, chunksize=chunksize):
        dfs.append(chunk[chunk["label"].isin(CHARTEVENTS_EXTRACT_LABELS)])
    if not dfs:
        return pd.DataFrame()
    fdf = pd.concat(dfs, ignore_index=True)

    # 단위 변환 (원본 lines 231-244)
    mask = fdf["label"] == "Admission Weight (Kg)"
    fdf.loc[mask, "valuenum"] = fdf.loc[mask, "valuenum"] * 2.20462
    fdf.loc[mask, "label"] = "Weight (lbs)"
    fdf.loc[mask, "valueuom"] = "lbs"

    mask = fdf["label"] == "Height (cm)"
    fdf.loc[mask, "valuenum"] = fdf.loc[mask, "valuenum"] * 0.393701
    fdf.loc[mask, "label"] = "Height (Inches)"
    fdf.loc[mask, "valueuom"] = "Inch"

    mask = fdf["label"] == "Daily Weight"
    fdf.loc[mask, "valuenum"] = fdf.loc[mask, "valuenum"] * 2.20462
    fdf.loc[mask, "label"] = "Weight (lbs)"
    fdf.loc[mask, "valueuom"] = "lbs"

    fdf = fdf[fdf["label"] != "PAR-Oxygen saturation"]
    labels_to_clean = ["Albumin", "Total Bilirubin", "Hematocrit (serum)",
                       "Creatinine (serum)", "Weight (lbs)"]
    fdf = fdf[~((fdf["label"].isin(labels_to_clean)) & (fdf["valueuom"].isna()))]
    fdf = fdf.reset_index(drop=True)

    rename_map = {
        "Height": "Height (Inches)",
        "Temperature Fahrenheit": "temperature",
        "Heart Rate": "heartrate",
        "Respiratory Rate": "resprate",
        "O2 saturation pulseoxymetry": "o2sat",
    }
    fdf["label"] = fdf["label"].replace(rename_map)
    fdf = fdf[fdf["label"] != "Admission Weight (lbs.)"]

    mask = fdf["label"] == "Temperature Celsius"
    fdf.loc[mask, "valuenum"] = (fdf.loc[mask, "valuenum"] * 9 / 5) + 32
    fdf.loc[mask, "label"] = "temperature"
    fdf.loc[mask, "valueuom"] = "°F"

    return fdf.reset_index(drop=True)


def _load_ecg_metadata():
    """records_w_diag_icd10.csv → study_id, subject_id, ecg_time DataFrame."""
    df = pd.read_csv(ICD_CSV, low_memory=False,
                     usecols=["study_id", "subject_id", "ecg_time"])
    df["ecg_time"] = pd.to_datetime(df["ecg_time"])
    return df


def _quantile_filter(df, group_col, value_col, lo=0.01, hi=0.99):
    """원본 mimic_preprocessing.py 의 1-99% quantile filter."""
    q_lo = df.groupby(group_col)[value_col].transform(lambda x: x.quantile(lo))
    q_hi = df.groupby(group_col)[value_col].transform(lambda x: x.quantile(hi))
    return df[(df[value_col] >= q_lo) & (df[value_col] <= q_hi)]


# ═══════════════════════════════════════════════════════════════
# Biometrics (3) — omr.csv.gz + chartevents 보강 + 30일 윈도우
# 원본 mimic_preprocessing.py:115-117, 280-306, 378-386 재현.
# ═══════════════════════════════════════════════════════════════
def build_biometrics_task(study_to_h5):
    logging.info("\n=== Biometrics (Height / Weight / BMI) ===")
    if not OMR_CSV.exists():
        logging.warning(f"  omr.csv.gz 없음 ({OMR_CSV}) — 건너뜀")
        return

    df_ecg = _load_ecg_metadata()
    subject_ids = set(df_ecg["subject_id"].unique())

    omr = pd.read_csv(OMR_CSV)
    omr = omr[omr["result_name"].isin(BIOMETRIC_COLS)]
    omr = omr.dropna(subset=["result_value"])
    omr["chartdate"] = pd.to_datetime(omr["chartdate"])

    # chartevents 보강 (Weight (lbs), Height (Inches))
    _prepare_chartevents_filtered(subject_ids)
    fdf = _load_chartevents_extract()
    if fdf is not None and not fdf.empty:
        new_rows = []
        for label in ["Weight (lbs)", "Height (Inches)"]:
            sub = fdf[fdf["label"] == label]
            for _, r in sub.iterrows():
                new_rows.append({"subject_id": r["subject_id"],
                                  "chartdate": r["storetime"],
                                  "seq_num": 0,
                                  "result_value": r["valuenum"],
                                  "result_name": label})
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            omr = pd.concat([omr, new_df], ignore_index=True)
        # Weight (lbs) → "Weight (Lbs)" (omr 표준)
        omr["result_name"] = omr["result_name"].replace({"Weight (lbs)": "Weight (Lbs)"})
        logging.info(f"  chartevents 보강 후 omr rows: {len(omr):,}")

    omr["result_value"] = pd.to_numeric(omr["result_value"], errors="coerce")
    omr = _quantile_filter(omr, "result_name", "result_value")
    omr["chartdate"] = pd.to_datetime(omr["chartdate"])

    # ECG-time 기준 closest 30일 내 매칭 (mimic_preprocessing.py:378-386)
    omr_subset = omr[omr["result_name"].isin(BIOMETRIC_COLS)]
    merged = df_ecg[["subject_id", "study_id", "ecg_time"]].merge(
        omr_subset, on="subject_id", how="left"
    )
    merged["time_diff"] = (merged["chartdate"] - merged["ecg_time"]).abs().dt.days
    merged = merged[merged["time_diff"] <= 30]
    closest_idx = merged.groupby(
        ["subject_id", "ecg_time", "result_name"]
    )["time_diff"].idxmin()
    closest = merged.loc[closest_idx]
    wide = closest.pivot_table(
        index=["study_id", "subject_id", "ecg_time"],
        columns="result_name", values="result_value"
    ).reset_index()

    wide["filepath"] = wide["study_id"].apply(
        lambda x: study_to_h5.get(int(x)) if pd.notna(x) else None
    )
    wide = wide[wide["filepath"].notna()].copy()

    # 컬럼이 있는지 확인 (없으면 NaN으로)
    for c in BIOMETRIC_COLS:
        if c not in wide.columns:
            wide[c] = np.nan
    save_regression_csv(
        "biometrics", wide, BIOMETRIC_COLS,
        source_desc="omr.csv.gz (Height/Weight/BMI) + chartevents 보강 (Weight/Height), "
                    "ECG-time 기준 closest 30일 매칭. mimic_preprocessing.py:115-117,280-306,378-386 재현.",
    )


# ═══════════════════════════════════════════════════════════════
# Vital signs (6) — vitalsign.csv.gz + chartevents 보강 + 1시간 윈도우
# 원본 mimic_preprocessing.py:120-122, 307-339, 388-396 재현.
# ═══════════════════════════════════════════════════════════════
def build_vitals_task(study_to_h5):
    logging.info("\n=== Vital signs (temp / HR / RR / SpO2 / SBP / DBP) ===")
    if not VITAL_CSV.exists():
        logging.warning(f"  vitalsign.csv.gz 없음 ({VITAL_CSV}) — 건너뜀")
        return

    df_ecg = _load_ecg_metadata()
    subject_ids = set(df_ecg["subject_id"].unique())

    vital = pd.read_csv(VITAL_CSV)
    vital = vital[["subject_id", "stay_id", "charttime",
                    "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]]
    vital["charttime"] = pd.to_datetime(vital["charttime"])
    vital_long = vital.melt(
        id_vars=["subject_id", "stay_id", "charttime"],
        value_vars=VITAL_COLS,
        var_name="result_name", value_name="result_value",
    ).sort_values(["subject_id", "charttime", "result_name"]).reset_index(drop=True)

    # chartevents 보강 (temperature, heartrate, resprate, o2sat)
    _prepare_chartevents_filtered(subject_ids)
    fdf = _load_chartevents_extract()
    if fdf is not None and not fdf.empty:
        new_rows = []
        for label in ["temperature", "heartrate", "resprate", "o2sat"]:
            sub = fdf[fdf["label"] == label]
            for _, r in sub.iterrows():
                new_rows.append({"subject_id": r["subject_id"],
                                  "stay_id": 0,
                                  "charttime": r["storetime"],
                                  "result_name": label,
                                  "result_value": r["valuenum"]})
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            vital_long = pd.concat([vital_long, new_df], ignore_index=True)
        logging.info(f"  chartevents 보강 후 vital rows: {len(vital_long):,}")

    vital_long = vital_long.dropna(subset=["result_value"]).reset_index(drop=True)
    vital_long["result_value"] = pd.to_numeric(vital_long["result_value"], errors="coerce")
    vital_long = _quantile_filter(vital_long, "result_name", "result_value")
    vital_long["charttime"] = pd.to_datetime(vital_long["charttime"])

    # ECG-time 기준 closest 1시간 내 매칭 (mimic_preprocessing.py:388-396)
    merged = df_ecg[["subject_id", "study_id", "ecg_time"]].merge(
        vital_long[["subject_id", "charttime", "result_name", "result_value"]],
        on="subject_id", how="left"
    )
    merged["time_diff"] = (merged["charttime"] - merged["ecg_time"]).abs().dt.total_seconds() / 3600
    merged = merged[merged["time_diff"] <= 1]
    closest_idx = merged.groupby(
        ["subject_id", "ecg_time", "result_name"]
    )["time_diff"].idxmin()
    closest = merged.loc[closest_idx]
    wide = closest.pivot_table(
        index=["study_id", "subject_id", "ecg_time"],
        columns="result_name", values="result_value"
    ).reset_index()

    wide["filepath"] = wide["study_id"].apply(
        lambda x: study_to_h5.get(int(x)) if pd.notna(x) else None
    )
    wide = wide[wide["filepath"].notna()].copy()

    for c in VITAL_COLS:
        if c not in wide.columns:
            wide[c] = np.nan
    save_regression_csv(
        "vitals", wide, VITAL_COLS,
        source_desc="vitalsign.csv.gz (HR/RR/BP/Temp/SpO2) + chartevents 보강, "
                    "ECG-time 기준 closest 1시간 매칭. "
                    "mimic_preprocessing.py:120-122,307-339,388-396 재현.",
    )


# ═══════════════════════════════════════════════════════════════
# Lab values (18) — labevents + d_labitems + chartevents 보강
# 원본 mimic_preprocessing.py:124-154, 344-372, 398-407 재현.
# ═══════════════════════════════════════════════════════════════
def build_labvalues_task(study_to_h5):
    logging.info("\n=== Lab values (18 targets) ===")
    if not LABEVENTS_CSV.exists() or not D_LABITEMS_CSV.exists():
        logging.warning(f"  labevents.csv.gz 또는 d_labitems.csv.gz 없음 — 건너뜀")
        return

    df_ecg = _load_ecg_metadata()
    subject_ids = set(df_ecg["subject_id"].unique())

    # 1. labitems 화이트리스트
    dflabitems = pd.read_csv(D_LABITEMS_CSV)
    dflabitems = dflabitems[dflabitems["itemid"].isin(LAB_ITEMIDS)]

    # 2. labevents (청크로 읽기 — 30M+ rows)
    logging.info("  labevents 청크 읽기")
    keep_itemids = set(dflabitems["itemid"].unique())
    chunks = []
    for chunk in pd.read_csv(LABEVENTS_CSV, compression="gzip",
                              chunksize=2_000_000, low_memory=True):
        chunk = chunk[chunk["itemid"].isin(keep_itemids)]
        chunk = chunk[chunk["valuenum"].notna()]
        chunk = chunk[chunk["subject_id"].isin(subject_ids)]
        if not chunk.empty:
            chunks.append(chunk)
    dflabevents = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    if dflabevents.empty:
        logging.warning("  labevents 매칭 0건 — skip")
        return
    dflabevents = dflabevents.merge(dflabitems[["itemid", "label"]], on="itemid", how="left")

    # 3. (label, itemid) 가장 흔한 pair만 (mimic_preprocessing.py:135-138)
    pair_counts = dflabevents.groupby(["label", "itemid"]).size().reset_index(name="count")
    most_common = pair_counts.loc[pair_counts.groupby("label")["count"].idxmax(),
                                   ["label", "itemid"]]
    dflabevents = dflabevents[
        dflabevents.set_index(["label", "itemid"]).index.isin(
            most_common.set_index(["label", "itemid"]).index
        )
    ]

    # 4. 1-99% outlier filter per label
    dflabevents = _quantile_filter(dflabevents, "label", "valuenum")

    # 5. 가장 흔한 valueuom
    uom_counts = dflabevents.groupby(["itemid", "valueuom"]).size().reset_index(name="count")
    most_common_uom = uom_counts.loc[uom_counts.groupby("itemid")["count"].idxmax(),
                                       ["itemid", "valueuom"]]
    dflabevents = dflabevents.merge(most_common_uom, on=["itemid", "valueuom"], how="inner")
    dflabevents["storetime"] = pd.to_datetime(dflabevents["storetime"])
    dflabevents = dflabevents[["subject_id", "storetime", "valuenum", "label", "valueuom"]]

    # chartevents 보강 (Albumin/Bilirubin/Hematocrit/Creatinine/Hemoglobin)
    _prepare_chartevents_filtered(subject_ids)
    fdf = _load_chartevents_extract()
    if fdf is not None and not fdf.empty:
        new_rows = []
        for label in ["Creatinine (serum)", "Hemoglobin", "Hematocrit (serum)",
                      "Total Bilirubin", "Albumin"]:
            sub = fdf[fdf["label"] == label]
            for _, r in sub.iterrows():
                new_rows.append({"subject_id": r["subject_id"],
                                  "storetime": r["storetime"],
                                  "valuenum": r["valuenum"],
                                  "label": label})
        if new_rows:
            new_df = pd.DataFrame(new_rows)
            dflabevents = pd.concat([dflabevents, new_df], ignore_index=True)
        # 라벨명 통일 (mimic_preprocessing.py:359-365)
        dflabevents["label"] = dflabevents["label"].replace({
            "Creatinine (serum)": "Creatinine",
            "Hematocrit (serum)": "Hematocrit",
            "Total Bilirubin": "Bilirubin, Total",
        })
        logging.info(f"  chartevents 보강 후 labevents rows: {len(dflabevents):,}")

    # 보강 후 다시 quantile filter
    dflabevents["valuenum"] = pd.to_numeric(dflabevents["valuenum"], errors="coerce")
    dflabevents = _quantile_filter(dflabevents, "label", "valuenum")
    dflabevents["storetime"] = pd.to_datetime(dflabevents["storetime"])

    # ECG-time 기준 closest 1시간 매칭 (mimic_preprocessing.py:398-407)
    labs_subset = dflabevents[dflabevents["label"].isin(LAB_COLS)]
    merged = df_ecg[["subject_id", "study_id", "ecg_time"]].merge(
        labs_subset, on="subject_id", how="left"
    )
    merged["time_diff"] = (merged["storetime"] - merged["ecg_time"]).abs().dt.total_seconds() / 3600
    merged = merged[merged["time_diff"] <= 1]
    closest_idx = merged.groupby(["subject_id", "ecg_time", "label"])["time_diff"].idxmin()
    closest = merged.loc[closest_idx]
    wide = closest.pivot_table(
        index=["study_id", "subject_id", "ecg_time"],
        columns="label", values="valuenum"
    ).reset_index()

    wide["filepath"] = wide["study_id"].apply(
        lambda x: study_to_h5.get(int(x)) if pd.notna(x) else None
    )
    wide = wide[wide["filepath"].notna()].copy()

    for c in LAB_COLS:
        if c not in wide.columns:
            wide[c] = np.nan
    save_regression_csv(
        "labvalues", wide, LAB_COLS,
        source_desc="labevents.csv.gz + d_labitems.csv.gz (18 lab labels) + "
                    "chartevents 보강 (Creatinine/Hemoglobin/Hematocrit/Bilirubin/Albumin), "
                    "ECG-time 기준 closest 1시간 매칭. "
                    "mimic_preprocessing.py:124-154,344-372,398-407 재현.",
    )


# ═══════════════════════════════════════════════════════════════
# 메인
# ═══════════════════════════════════════════════════════════════
ALL_TASKS = ["diagnostic", "sex", "age", "ecg_features",
             "deterioration", "mortality", "icu_admission",
             "biometrics", "vitals", "labvalues"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="전 태스크 생성")
    parser.add_argument("--task", choices=ALL_TASKS, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    if not args.all and args.task is None:
        parser.print_help()
        return

    targets = ALL_TASKS if args.all else [args.task]

    logging.info("Loading H5 study_id → filepath mapping…")
    study_to_h5 = load_h5_mapping()
    logging.info(f"  매핑 entries: {len(study_to_h5):,}")

    for t in targets:
        if t == "diagnostic":
            build_diagnostic_tasks(study_to_h5)
        elif t == "sex" or t == "age":
            # sex/age는 한 번에 빌드
            if t == "sex":
                build_sex_age_tasks(study_to_h5)
        elif t == "ecg_features":
            build_ecg_features_task(study_to_h5)
        elif t == "deterioration":
            build_deterioration_task(study_to_h5)
        elif t == "mortality":
            build_mortality_task(study_to_h5)
        elif t == "icu_admission":
            build_icu_admission_task(study_to_h5)
        elif t == "biometrics":
            build_biometrics_task(study_to_h5)
        elif t == "vitals":
            build_vitals_task(study_to_h5)
        elif t == "labvalues":
            build_labvalues_task(study_to_h5)

    logging.info("\n완료!")


if __name__ == "__main__":
    main()
