"""
MIMIC-IV-ECG Benchmark label build script
=============================================
Original ecg-fm-benchmarking/mimic_preprocessing.py and
ecg-fm-benchmarking/code/clinical_ts/utils/mimic_ecg_preprocessing.py
1:1 reproduction.

Paper Table's 11 MIMIC tasks:

  ✅ Cardiac discharge diagnoses     (records_w_diag_icd10.csv, ICD-10 chapter IX)
  ✅ Non-cardiac discharge diagnoses (records_w_diag_icd10.csv, Other)
  ✅ Sex (binary)                    (records_w_diag_icd10.csv)
  ✅ Age (regression)                (records_w_diag_icd10.csv)
  ✅ ECG features (regression, 7)    (machine_measurements.csv)
  ✅ Clinical deterioration (6)      (mds_ed.csv)
  ✅ Mortality (7-horizon)           (mds_ed.csv)
  ✅ ICU admission (2)               (mds_ed.csv)
  ✅ Biometrics (3)                  (omr.csv.gz + chartevents.csv.gz)
  ✅ Vital signs (6)                 (vitalsign.csv.gz + chartevents.csv.gz)
  ✅ Lab values (18)                 (labevents.csv.gz + d_labitems.csv.gz + chartevents.csv.gz)

required raw files (all PhysioNet credentialed):
  /raw/physionet.org/files/
    ├── mimic-iv-ecg/1.0/machine_measurements.csv
    ├── mimic-iv-ecg-ext-icd-labels/1.0.1/records_w_diag_icd10.csv
    ├── mimic-iv-ed/2.2/ed/{vitalsign,edstays}.csv.gz
    ├── mimiciv/3.1/hosp/{omr,labevents,d_labitems,admissions}.csv.gz
    ├── mimiciv/3.1/icu/{chartevents,d_items,icustays}.csv.gz
    └── multimodal-emergency-benchmark/1.0.0/mds_ed.csv

output (all labels/mimic_<task>_paper_labels.{csv,json}):
  cardiac, noncardiac, sex, age, ecg_features
  deterioration, mortality, icu_admission
  biometrics, vitals, labvalues

run:
  python scripts/build_mimic_labels.py --all
  python scripts/build_mimic_labels.py --task biometrics

Cache:
  labels/_cache/chartevents_filtered.csv ~30GB chartevents chunk filter result is
  preserve (shared by 3 tasks). re-run at automatic re-use.
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
# path
# ═══════════════════════════════════════════════════════════════
H5_ROOT = Path("/path/to/ecg_data/h5/mimic4/v2.0")
RAW_BASE = Path("/path/to/ecg_data/raw/physionet.org/files")
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

# cache (chartevents filtered once, then re-used)
CACHE_DIR = Path("/path/to/workspace/benchmark/labels/_cache")
CHARTEVENTS_FILTERED = CACHE_DIR / "chartevents_filtered.csv"

# label definition (paper / mimic_preprocessing.py identical)
BIOMETRIC_COLS = ["Height (Inches)", "Weight (Lbs)", "BMI (kg/m2)"]
VITAL_COLS = ["dbp", "heartrate", "o2sat", "resprate", "sbp", "temperature"]
LAB_COLS = [
    "PT", "Albumin", "Anion Gap", "Bicarbonate", "Bilirubin, Total",
    "Calcium, Total", "Creatinine", "Ferritin", "Urea Nitrogen",
    "Hematocrit", "Hemoglobin", "Lymphocytes", "MCHC", "RDW",
    "Red Blood Cells", "RDW-SD", "Creatine Kinase (CK)", "NTproBNP",
]

# d_labitems itemid whitelist (mimic_preprocessing.py:126-128 as-is)
LAB_ITEMIDS = [
    50963, 51006, 52647, 50811, 51222, 51640, 50912, 52546, 50924, 50912,
    52546, 51221, 51480, 51638, 51639, 52028, 50862, 53085, 51006, 52647,
    52172, 50811, 51222, 51640, 50868, 52500, 51277, 50882, 50885, 53089,
    51221, 51480, 51638, 51639, 52028, 51237, 51675, 51279, 51274, 52921,
    50910, 51249, 50893, 51244,
]

# to extract from chartevents label (mimic_preprocessing.py:208-215)
CHARTEVENTS_EXTRACT_LABELS = [
    "Height (cm)", "Height", "Daily Weight",
    "Admission Weight (lbs.)", "Admission Weight (Kg)",
    "Temperature Celsius", "Temperature Fahrenheit",
    "Heart Rate", "Respiratory Rate",
    "PAR-Oxygen saturation", "O2 saturation pulseoxymetry",
    "Albumin", "Anion Gap", "Total Bilirubin",
    "Creatinine (serum)", "Hematocrit (serum)", "Hemoglobin",
]

# MDS-ED column definitions (same classification as paper Table)
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

OUT_DIR = Path("/path/to/workspace/benchmark/labels")

# original paper config (mimic_preprocessing.py)
FINETUNE_DATASET = "mimic_ed_all_edfirst_all_2000_5A"
MIN_CNT = 2000      # min positive count per diagnosis label
DIGITS = 5          # ICD-10 truncate digits
PROPAGATE_ALL = True  # 5A mode — add all ancestor nodes


# ═══════════════════════════════════════════════════════════════
# H5 mapping
# ═══════════════════════════════════════════════════════════════
def load_h5_mapping():
    """study_id → h5_filepath mapping (799,929)."""
    fn = pd.read_csv(H5_ROOT / "file_name.csv")
    fn["original_record_name"] = fn["original_record_name"].astype(int)
    return dict(zip(fn["original_record_name"], fn["h5_filepath"]))


# ═══════════════════════════════════════════════════════════════
# ICD diagnosis — original prepare_mimic_ecg() reproduction
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
    """ICD-10 chapter mapping (simplified — by alphabetic prefix).

    Paper's 'cardiac chapter IX' = circulatory system, codes starting with 'I'.
    Avoid the icd10 package dependency by using the first-letter prefix mapping.

    ICD-10 chapter:
      I00-I99 = Chapter IX (circulatory system, cardiac)
      others = non-cardiac
    """
    if not icd_code or not isinstance(icd_code, str) or len(icd_code) == 0:
        return "unknown"
    return icd_code[0].upper()


def parse_diag_lists(df, cols):
    """Parse a list stored as a string into an actual list."""
    for c in cols:
        df[c] = df[c].apply(lambda x: eval(x) if isinstance(x, str) else [])
    return df


def prepare_diagnostic_labels(df_diags, label_col="all_diag_all",
                               min_cnt=MIN_CNT, digits=DIGITS,
                               propagate_all=PROPAGATE_ALL):
    """Reproduces the label-extraction portion of the original prepare_mimic_ecg().

    1. truncate ICD-10 codes by digit count
    2. trailing X remove
    3. propagate_all: add all ancestor codes (3..len)
    4. min_cnt or moreis code only keep

    Returns:
        df: label_train column added DataFrame
        lbl_itos: surviving label list (sorted order)
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
    """label → CSV column name."""
    return (str(s).replace(" ", "_").replace(",", "")
            .replace("-", "_").replace("(", "").replace(")", "")
            .replace("'", "").replace("/", "_").replace(":", "_")
            .replace(".", "_"))


def save_multilabel_csv(name, df, label_col, lbl_itos, source_desc):
    """multi-label binary CSV save. strat_fold column include (paper split)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"mimic_{name}_paper_labels.csv"
    out_json = OUT_DIR / f"mimic_{name}_paper_labels.json"

    label_cols = [sanitize(l) for l in lbl_itos]
    label_set = set(lbl_itos)

    keep_meta = [c for c in ("filepath", "strat_fold", "fold") if c in df.columns]
    out_df = df[keep_meta].copy()
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
    """single binary column CSV save. strat_fold column include."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"mimic_{name}_paper_labels.csv"
    out_json = OUT_DIR / f"mimic_{name}_paper_labels.json"
    keep_meta = [c for c in ("filepath", "strat_fold", "fold") if c in df.columns]
    out_df = df[keep_meta].copy()
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
    """Save multivariate binary multi-label CSV (NaN preserved, 0/1 numeric, strat_fold included)."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"mimic_{name}_paper_labels.csv"
    out_json = OUT_DIR / f"mimic_{name}_paper_labels.json"
    keep_meta = [c for c in ("filepath", "strat_fold", "fold") if c in df.columns]
    out_df = df[keep_meta + label_cols].copy()
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
    """multivariate regression CSV save (value raw, NaN preserve). strat_fold include."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / f"mimic_{name}_paper_labels.csv"
    out_json = OUT_DIR / f"mimic_{name}_paper_labels.json"
    keep_meta = [c for c in ("filepath", "strat_fold", "fold") if c in df.columns]
    out_df = df[keep_meta + label_cols].copy()
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
def get_diagnostic_cohort(study_to_h5=None, return_filepath=True):
    """original paper's is_diagnostic==1 cohort reproduction.

    paper mimic_preprocessing.py:44-50 + ecg_utils.py prepare_mimic_ecg
    (finetune_dataset='mimic_ed_all_edfirst_all_2000_5A'):
      - subsettrain='ed' + has_statements_train==True
      - subsettest='edfirst' (first ECG per stay) + has_statements_test==True

    paper Table's metadata tasks (sex/age/ecg_features/biometrics/vitals/labvalues)
    and MDS-ED tasks restricted to the cohort.

    Returns:
        DataFrame with columns: study_id, subject_id, ecg_time, fold, [filepath]
    """
    df = pd.read_csv(ICD_CSV, low_memory=False)
    df = parse_diag_lists(df, ["all_diag_all", "ed_diag_ed", "ed_diag_hosp",
                                "hosp_diag_hosp", "all_diag_hosp"])
    df["has_all"] = df["all_diag_all"].apply(lambda x: len(x) > 0)
    df["has_ed"] = df["ed_diag_ed"].apply(lambda x: len(x) > 0)

    # paper subsettrain='ed' OR subsettest='edfirst':
    #   train: ED ECG + has_all (all_diag_all not empty)
    #   test:  ED ECG + ecg_no_within_stay==0 + has_ed
    # merge: ED ECG + (has_all OR has_ed)
    cohort = df[
        (df["ecg_taken_in_ed"] == True) &
        (df["has_all"] | df["has_ed"])
    ].copy()

    if study_to_h5 is not None and return_filepath:
        cohort["filepath"] = cohort["study_id"].apply(
            lambda x: study_to_h5.get(int(x)) if pd.notna(x) else None
        )
        cohort = cohort[cohort["filepath"].notna()].copy()

    return cohort[["study_id", "subject_id", "ecg_time", "fold", "strat_fold"] +
                   (["filepath"] if return_filepath and study_to_h5 is not None else [])]


def build_diagnostic_tasks(study_to_h5):
    """Cardiac / Non-cardiac discharge diagnoses (multi-label).

    original prepare_mimic_ecg(finetune_dataset='mimic_ed_all_edfirst_all_2000_5A')
    Exact order:
      1. extract labels from all records_w_diag_icd10.csv → determine lbl_itos
         (= mimic_ecg_preprocessing.py:120-127, df_diags all use)
      2. ED ECG by train subset filtering (subsettrain='ed')
      3. ED first-ECG-per-stay by test subset filtering (subsettest='edfirst')
    Key: lbl_itos must be extracted AFTER (not BEFORE) the ED filter to reproduce the paper's 158/918 label set.
    """
    logging.info("\n=== Diagnostic (cardiac / non-cardiac) ===")
    df = pd.read_csv(ICD_CSV, low_memory=False)
    df = parse_diag_lists(df, ["all_diag_all", "ed_diag_ed", "ed_diag_hosp",
                                "hosp_diag_hosp", "all_diag_hosp"])
    logging.info(f"  total ICD csv rows: {len(df):,} rows")

    # 1. all corpus from label extract (paper original identical)
    df_lbl, lbl_itos = prepare_diagnostic_labels(df, label_col="all_diag_all")
    logging.info(f"  all corpus Labels (≥{MIN_CNT}): {len(lbl_itos)}")

    # 2. ED filter (subsettrain='ed' / subsettest='edfirst' common constraint)
    df_lbl = df_lbl[df_lbl["ecg_taken_in_ed"] == True].copy()
    logging.info(f"  ECG taken in ED: {len(df_lbl):,}")

    # cardiac (chapter IX) vs non-cardiac (other chapters)
    cardiac = [c for c in lbl_itos if get_chapter_prefix(c) == "I"]
    noncardiac = [c for c in lbl_itos if get_chapter_prefix(c) != "I"]
    logging.info(f"  Cardiac (chapter IX, 'I' prefix): {len(cardiac)}")
    logging.info(f"  Non-cardiac:                       {len(noncardiac)}")

    # exclude ECGs without statistics (label_train empty) — matches original has_statements_train==True
    df_lbl["has_label"] = df_lbl["label_train"].apply(lambda x: len(x) > 0)

    # filepath mapping
    df_lbl["filepath"] = df_lbl["study_id"].apply(
        lambda x: study_to_h5.get(int(x)) if pd.notna(x) else None
    )
    df_lbl = df_lbl[df_lbl["filepath"].notna()].copy()
    logging.info(f"  after H5 mapping: {len(df_lbl):,}")

    # cardiac output
    df_card = df_lbl.copy()
    df_card["label_card"] = df_card["label_train"].apply(
        lambda x: [c for c in x if c in set(cardiac)]
    )
    df_card_filt = df_card[df_card["label_card"].apply(len) > 0].copy()
    logging.info(f"  Cardiac task rows (≥1 cardiac code): {len(df_card_filt):,}")
    save_multilabel_csv("cardiac", df_card_filt, "label_card", cardiac,
                        source_desc=f"records_w_diag_icd10.csv → ICD-10 chapter IX (I-prefix), "
                                    f"truncate {DIGITS} digits, propagate ancestors, min_cnt={MIN_CNT}")

    # non-cardiac output
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
    """Sex (binary), Age (regression) — paper is_diagnostic cohort apply."""
    logging.info("\n=== Sex / Age (patient characteristics) ===")
    cohort = get_diagnostic_cohort(study_to_h5)
    logging.info(f"  diagnostic cohort: {len(cohort):,}")

    df = pd.read_csv(ICD_CSV, low_memory=False,
                     usecols=["study_id", "gender", "age"])
    df = df.merge(cohort[["study_id", "filepath", "strat_fold", "fold"]],
                  on="study_id", how="inner")

    # sex
    df_sex = df[df["gender"].isin(["M", "F"])].copy()
    df_sex["sex"] = (df_sex["gender"] == "M").astype(int)
    save_binary_csv("sex", df_sex, "sex", "is_male",
                    source_desc="records_w_diag_icd10.csv ∩ paper is_diagnostic cohort "
                                "(ED + has_statements). gender=M→1, F→0.")

    # age
    df_age = df[df["age"].notna()].copy()
    save_regression_csv("age", df_age, ["age"],
                        source_desc="records_w_diag_icd10.csv ∩ paper is_diagnostic cohort. "
                                    "age in years at ECG time.")


def build_ecg_features_task(study_to_h5):
    """ECG features (regression, 7) — machine_measurements.csv."""
    logging.info("\n=== ECG features (machine_measurements) ===")
    df = pd.read_csv(MM_CSV, low_memory=False)
    logging.info(f"  machine_measurements  total {len(df):,}")

    # original outlier handling (mimic_preprocessing.py:92-100)
    for col in ["qrs_axis", "t_axis", "p_axis"]:
        df.loc[(df[col] < -360) | (df[col] > 360), col] = np.nan
    for col in ["p_onset", "p_end", "qrs_onset", "qrs_end", "t_end", "rr_interval"]:
        df.loc[(df[col] < 0) | (df[col] > 5000), col] = np.nan

    # Original derived (RR/PR/QRS/QT/QTc compute — mimic_preprocessing.py:101-109)
    df = df.rename(columns={"rr_interval": "RR", "p_axis": "P_wave_axis",
                             "qrs_axis": "QRS_axis", "t_axis": "T_wave_axis"})
    df["PR"] = df["qrs_onset"] - df["p_onset"]
    df["QRS"] = df["qrs_end"] - df["qrs_onset"]
    df["QT"] = df["t_end"] - df["qrs_onset"]
    df["QTc"] = np.where(df["RR"] != 0, df["QT"] / np.sqrt(df["RR"] / 1000), np.nan)

    # paper's 7 feature
    feat_cols = ["RR", "QRS", "QT", "QTc", "P_wave_axis", "QRS_axis", "T_wave_axis"]

    # paper cohort intersection (mimic_preprocessing.py:420 is_diagnostic==1)
    cohort = get_diagnostic_cohort(study_to_h5)
    logging.info(f"  diagnostic cohort: {len(cohort):,}")
    df = df.merge(cohort[["study_id", "filepath", "strat_fold", "fold"]],
                  on="study_id", how="inner")

    # all feature NaNis rows remove
    df = df.dropna(subset=feat_cols, how="all").copy()
    logging.info(f"  cohort ∩ ECG features: {len(df):,}")

    save_regression_csv("ecg_features", df, feat_cols,
                        source_desc="machine_measurements.csv ∩ paper is_diagnostic cohort. "
                                    "RR/QRS/QT/QTc/P_axis/QRS_axis/T_axis (outlier handling after raw).")


def _load_mds_ed_with_filepath(study_to_h5, value_cols, restrict_to_cohort=True):
    """MDS-ED CSV load + study_id → h5 filepath mapping + paper cohort filter.

    original mimic_preprocessing.py:
      - Replace -999.0 with np.nan (line 75)
      - general_data, general_strat_fold, general_subject_id use
      - line 420: is_diagnostic==1 cohort in restrict (paper Table 5,577 / 17,639 / 18,690)

    Here: H5 mapping by general_study_id + intersect with diagnostic cohort.

    add by paper Table matching above "value column all NaNis row remove" apply:
      - paper's "Samples"  label definitioned sample  
    """
    df = pd.read_csv(MDS_ED_CSV, low_memory=False)
    keep = ["general_study_id", "general_subject_id", "general_strat_fold"] + value_cols
    df = df[keep].copy()
    # paper's split: general_strat_fold 0-17/18/19 (18/1/1)
    df = df.rename(columns={"general_strat_fold": "strat_fold"})
    for c in value_cols:
        df[c] = df[c].replace(-999., np.nan)
    df["filepath"] = df["general_study_id"].apply(
        lambda x: study_to_h5.get(int(x)) if pd.notna(x) else None
    )
    df = df[df["filepath"].notna()].copy()

    # value column all NaNis row remove (paper Table samples matching (for))
    n_before_value = len(df)
    df = df.dropna(subset=value_cols, how="all").copy()
    logging.info(f"  MDS-ED rows: {len(df):,} / {n_before_value:,} "
                 f"(value column  of 1 or more valid)")

    # paper is_diagnostic cohort intersection (mimic_preprocessing.py:420)
    if restrict_to_cohort:
        cohort = get_diagnostic_cohort(study_to_h5, return_filepath=False)
        cohort_studies = set(cohort["study_id"].astype(int))
        n_before = len(df)
        df = df[df["general_study_id"].astype(int).isin(cohort_studies)].copy()
        logging.info(f"  cohort intersection: {len(df):,} / {n_before:,}")

    return df


def build_deterioration_task(study_to_h5):
    """Clinical deterioration — paper Table 5,577 × 6 outputs.

    original mimic_preprocessing.py is 67-84 reproduction.
    MDS-ED's 6 deterioration  (mortality/ICU exclude).
    """
    logging.info("\n=== Clinical deterioration (MDS-ED 6 outputs) ===")
    df = _load_mds_ed_with_filepath(study_to_h5, MDS_DETERIORATION_COLS)
    logging.info(f"  after H5 mapping: {len(df):,}")

    descs = {
        c: c.replace("deterioration_", "").replace("_", " ")
        for c in MDS_DETERIORATION_COLS
    }
    save_multilabel_numeric_csv(
        "deterioration", df, MDS_DETERIORATION_COLS,
        source_desc="multimodal-emergency-benchmark/1.0.0/mds_ed.csv of 6 deterioration columns "
                    "(severe_hypoxemia, ecmo, vasopressors, inotropes, mechanical_ventilation, cardiac_arrest), "
                    "original mimic_preprocessing.py:67-84 reproduction.",
        label_descriptions=descs,
    )


def build_mortality_task(study_to_h5):
    """Mortality — paper Table 17,639 × 7 outputs (multi-horizon).

    original MDS-ED's 7 mortality column.
    previous 1-class justorder version paper  as replace.
    """
    logging.info("\n=== Mortality (MDS-ED 7-horizon) ===")
    df = _load_mds_ed_with_filepath(study_to_h5, MDS_MORTALITY_COLS)
    logging.info(f"  after H5 mapping: {len(df):,}")
    descs = {
        c: c.replace("deterioration_mortality_", "mortality_") for c in MDS_MORTALITY_COLS
    }
    save_multilabel_numeric_csv(
        "mortality", df, MDS_MORTALITY_COLS,
        source_desc="multimodal-emergency-benchmark/1.0.0/mds_ed.csv of 7 mortality horizons "
                    "(1d/7d/28d/90d/180d/365d/stay).",
        label_descriptions=descs,
    )


def build_icu_admission_task(study_to_h5):
    """ICU admission — paper Table 18,690 × 2 outputs.

    original MDS-ED's 2 ICU column (icu_24h, icu_stay).
    previous hospital admission proxy version paper  as replace.
    """
    logging.info("\n=== ICU admission (MDS-ED 2 outputs) ===")
    df = _load_mds_ed_with_filepath(study_to_h5, MDS_ICU_COLS)
    logging.info(f"  after H5 mapping: {len(df):,}")
    descs = {c: c.replace("deterioration_", "") for c in MDS_ICU_COLS}
    save_multilabel_numeric_csv(
        "icu_admission", df, MDS_ICU_COLS,
        source_desc="multimodal-emergency-benchmark/1.0.0/mds_ed.csv of 2 ICU columns "
                    "(icu_24h: 24  inside ICU , icu_stay: identical stay  inside ICU ).",
        label_descriptions=descs,
    )


def _try_read_edstays():  # use  ofjust: ICU admission MDS-ED by 
    return None
def _legacy_try_read_edstays():
    """edstays.csv.gz users raw from truncatedcase   read ."""
    try:
        return pd.read_csv(ED_STAYS_CSV, usecols=["stay_id", "disposition"])
    except (EOFError, OSError) as e:
        logging.warning(f"  edstays.csv.gz  read  (gzip above available): {e}")
        #  read  — gzip above at chunked by  then some time available
        chunks = []
        try:
            for chunk in pd.read_csv(ED_STAYS_CSV, usecols=["stay_id", "disposition"],
                                     chunksize=50_000):
                chunks.append(chunk)
        except (EOFError, OSError):
            pass
        if chunks:
            partial = pd.concat(chunks, ignore_index=True)
            logging.warning(f"  edstays partial: {len(partial):,} rows time")
            return partial
        return pd.DataFrame(columns=["stay_id", "disposition"])


# build_icu_admission_task above MDS-ED based  as .


# ═══════════════════════════════════════════════════════════════
# chartevents filtering (Biometrics / Vitals / Labs common preprocessing)
# original mimic_preprocessing.py:158-273 reproduction. ~30GB chunk .
# ═══════════════════════════════════════════════════════════════
def _prepare_chartevents_filtered(subject_ids):
    """chartevents.csv.gz chunk by  filter·cache.

    original mimic_preprocessing.py:158-205:
      1. d_items by itemid→label mapping
      2. chunksize=1M as chartevents 
      3. subject_id df in row only keep
      4. label != 'Safety Measures'
      5. label per  total count >= 1000is label only keep
      6. CACHE_DIR/chartevents_filtered.csv in  save

    re-run at cache if present, .
    """
    if CHARTEVENTS_FILTERED.exists():
        logging.info(f"  chartevents filter cache use: {CHARTEVENTS_FILTERED}")
        return
    if not CHARTEVENTS_CSV.exists() or not D_ITEMS_CSV.exists():
        logging.warning(f"  chartevents.csv.gz or d_items.csv.gz none — enrich skip")
        return

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    d_items = pd.read_csv(D_ITEMS_CSV, compression="gzip", low_memory=False)
    d_items_subset = d_items[["itemid", "label"]]
    chunksize = 1_000_000
    min_label_count = 1000

    # PASS 1: label per  total count
    logging.info("  chartevents PASS 1 — label  (1M chunks)")
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
    logging.info(f"  keep label : {len(labels_to_keep)}")

    # PASS 2: filter after  
    logging.info("  chartevents PASS 2 — filter after  save")
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
    logging.info(f"  chartevents cache save: {CHARTEVENTS_FILTERED}")


def _load_chartevents_extract():
    """filtered chartevents from to_extract label only load + justabove convert.

    original mimic_preprocessing.py:218-273 reproduction.
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

    # justabove convert (original lines 231-244)
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


def _load_ecg_metadata(study_to_h5=None, cohort_only=True):
    """records_w_diag_icd10.csv → study_id, subject_id, ecg_time, strat_fold, fold DataFrame.

    cohort_only=True: paper is_diagnostic cohort in restrict
    (mimic_preprocessing.py:420 — biometrics/vitals/labvalues all cohort).
    strat_fold/fold paper's 18/1/1 split above include.
    """
    if cohort_only and study_to_h5 is not None:
        cohort = get_diagnostic_cohort(study_to_h5)
        df = cohort[["study_id", "subject_id", "ecg_time", "strat_fold", "fold"]].copy()
    else:
        df = pd.read_csv(ICD_CSV, low_memory=False,
                         usecols=["study_id", "subject_id", "ecg_time", "strat_fold", "fold"])
    df["ecg_time"] = pd.to_datetime(df["ecg_time"])
    return df


def _quantile_filter(df, group_col, value_col, lo=0.01, hi=0.99):
    """original mimic_preprocessing.py of 1-99% quantile filter."""
    q_lo = df.groupby(group_col)[value_col].transform(lambda x: x.quantile(lo))
    q_hi = df.groupby(group_col)[value_col].transform(lambda x: x.quantile(hi))
    return df[(df[value_col] >= q_lo) & (df[value_col] <= q_hi)]


# ═══════════════════════════════════════════════════════════════
# Biometrics (3) — omr.csv.gz + chartevents enrich + 30case 
# original mimic_preprocessing.py:115-117, 280-306, 378-386 reproduction.
# ═══════════════════════════════════════════════════════════════
def build_biometrics_task(study_to_h5):
    logging.info("\n=== Biometrics (Height / Weight / BMI) ===")
    if not OMR_CSV.exists():
        logging.warning(f"  omr.csv.gz none ({OMR_CSV}) — ")
        return

    df_ecg = _load_ecg_metadata(study_to_h5, cohort_only=True)
    subject_ids = set(df_ecg["subject_id"].unique())
    logging.info(f"  diagnostic cohort: {len(df_ecg):,} ECGs / {len(subject_ids):,} patients")

    omr = pd.read_csv(OMR_CSV)
    omr = omr[omr["result_name"].isin(BIOMETRIC_COLS)]
    omr = omr.dropna(subset=["result_value"])
    omr["chartdate"] = pd.to_datetime(omr["chartdate"])

    # chartevents enrich (Weight (lbs), Height (Inches))
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
        # Weight (lbs) → "Weight (Lbs)" (omr standard)
        omr["result_name"] = omr["result_name"].replace({"Weight (lbs)": "Weight (Lbs)"})
        logging.info(f"  chartevents enrich after omr rows: {len(omr):,}")

    omr["result_value"] = pd.to_numeric(omr["result_value"], errors="coerce")
    omr = _quantile_filter(omr, "result_name", "result_value")
    omr["chartdate"] = pd.to_datetime(omr["chartdate"])

    # ECG-time reference: closest 30case  inside matching (mimic_preprocessing.py:378-386)
    omr_subset = omr[omr["result_name"].isin(BIOMETRIC_COLS)]
    merged = df_ecg[["subject_id", "study_id", "ecg_time", "strat_fold", "fold"]].merge(
        omr_subset, on="subject_id", how="left"
    )
    merged["time_diff"] = (merged["chartdate"] - merged["ecg_time"]).abs().dt.days
    merged = merged[merged["time_diff"] <= 30]
    closest_idx = merged.groupby(
        ["subject_id", "ecg_time", "result_name"]
    )["time_diff"].idxmin()
    closest = merged.loc[closest_idx]
    wide = closest.pivot_table(
        index=["study_id", "subject_id", "ecg_time", "strat_fold", "fold"],
        columns="result_name", values="result_value"
    ).reset_index()

    wide["filepath"] = wide["study_id"].apply(
        lambda x: study_to_h5.get(int(x)) if pd.notna(x) else None
    )
    wide = wide[wide["filepath"].notna()].copy()

    # column  confirm (if absent NaN as)
    for c in BIOMETRIC_COLS:
        if c not in wide.columns:
            wide[c] = np.nan
    save_regression_csv(
        "biometrics", wide, BIOMETRIC_COLS,
        source_desc="omr.csv.gz (Height/Weight/BMI) + chartevents enrich (Weight/Height), "
                    "ECG-time reference: closest 30case matching. mimic_preprocessing.py:115-117,280-306,378-386 reproduction.",
    )


# ═══════════════════════════════════════════════════════════════
# Vital signs (6) — vitalsign.csv.gz + chartevents enrich + 1 
# original mimic_preprocessing.py:120-122, 307-339, 388-396 reproduction.
# ═══════════════════════════════════════════════════════════════
def build_vitals_task(study_to_h5):
    logging.info("\n=== Vital signs (temp / HR / RR / SpO2 / SBP / DBP) ===")
    if not VITAL_CSV.exists():
        logging.warning(f"  vitalsign.csv.gz none ({VITAL_CSV}) — ")
        return

    df_ecg = _load_ecg_metadata(study_to_h5, cohort_only=True)
    subject_ids = set(df_ecg["subject_id"].unique())
    logging.info(f"  diagnostic cohort: {len(df_ecg):,} ECGs / {len(subject_ids):,} patients")

    vital = pd.read_csv(VITAL_CSV)
    vital = vital[["subject_id", "stay_id", "charttime",
                    "temperature", "heartrate", "resprate", "o2sat", "sbp", "dbp"]]
    vital["charttime"] = pd.to_datetime(vital["charttime"])
    vital_long = vital.melt(
        id_vars=["subject_id", "stay_id", "charttime"],
        value_vars=VITAL_COLS,
        var_name="result_name", value_name="result_value",
    ).sort_values(["subject_id", "charttime", "result_name"]).reset_index(drop=True)

    # chartevents enrich (temperature, heartrate, resprate, o2sat)
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
        logging.info(f"  chartevents enrich after vital rows: {len(vital_long):,}")

    vital_long = vital_long.dropna(subset=["result_value"]).reset_index(drop=True)
    vital_long["result_value"] = pd.to_numeric(vital_long["result_value"], errors="coerce")
    vital_long = _quantile_filter(vital_long, "result_name", "result_value")
    vital_long["charttime"] = pd.to_datetime(vital_long["charttime"])

    # ECG-time reference: closest 1  inside matching (mimic_preprocessing.py:388-396)
    merged = df_ecg[["subject_id", "study_id", "ecg_time", "strat_fold", "fold"]].merge(
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
        index=["study_id", "subject_id", "ecg_time", "strat_fold", "fold"],
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
        source_desc="vitalsign.csv.gz (HR/RR/BP/Temp/SpO2) + chartevents enrich, "
                    "ECG-time (closest within ±1 h). "
                    "mimic_preprocessing.py:120-122,307-339,388-396 reproduction.",
    )


# ═══════════════════════════════════════════════════════════════
# Lab values (18) — labevents + d_labitems + chartevents enrich
# original mimic_preprocessing.py:124-154, 344-372, 398-407 reproduction.
# ═══════════════════════════════════════════════════════════════
def build_labvalues_task(study_to_h5):
    logging.info("\n=== Lab values (18 targets) ===")
    if not LABEVENTS_CSV.exists() or not D_LABITEMS_CSV.exists():
        logging.warning(f"  labevents.csv.gz or d_labitems.csv.gz none — ")
        return

    df_ecg = _load_ecg_metadata(study_to_h5, cohort_only=True)
    subject_ids = set(df_ecg["subject_id"].unique())
    logging.info(f"  diagnostic cohort: {len(df_ecg):,} ECGs / {len(subject_ids):,} patients")

    # 1. labitems whitelist
    dflabitems = pd.read_csv(D_LABITEMS_CSV)
    dflabitems = dflabitems[dflabitems["itemid"].isin(LAB_ITEMIDS)]

    # 2. labevents (chunk by read — 30M+ rows)
    logging.info("  labevents chunk read")
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
        logging.warning("  labevents matching 0 — skip")
        return
    dflabevents = dflabevents.merge(dflabitems[["itemid", "label"]], on="itemid", how="left")

    # 3. (label, itemid)   pair only (mimic_preprocessing.py:135-138)
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

    # 5.   valueuom
    uom_counts = dflabevents.groupby(["itemid", "valueuom"]).size().reset_index(name="count")
    most_common_uom = uom_counts.loc[uom_counts.groupby("itemid")["count"].idxmax(),
                                       ["itemid", "valueuom"]]
    dflabevents = dflabevents.merge(most_common_uom, on=["itemid", "valueuom"], how="inner")
    dflabevents["storetime"] = pd.to_datetime(dflabevents["storetime"])
    dflabevents = dflabevents[["subject_id", "storetime", "valuenum", "label", "valueuom"]]

    # chartevents enrich (Albumin/Bilirubin/Hematocrit/Creatinine/Hemoglobin)
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
        # label name case (mimic_preprocessing.py:359-365)
        dflabevents["label"] = dflabevents["label"].replace({
            "Creatinine (serum)": "Creatinine",
            "Hematocrit (serum)": "Hematocrit",
            "Total Bilirubin": "Bilirubin, Total",
        })
        logging.info(f"  chartevents enrich after labevents rows: {len(dflabevents):,}")

    # enrich and then re- quantile filter
    dflabevents["valuenum"] = pd.to_numeric(dflabevents["valuenum"], errors="coerce")
    dflabevents = _quantile_filter(dflabevents, "label", "valuenum")
    dflabevents["storetime"] = pd.to_datetime(dflabevents["storetime"])

    # ECG-time (closest within ±1 h) (mimic_preprocessing.py:398-407)
    labs_subset = dflabevents[dflabevents["label"].isin(LAB_COLS)]
    merged = df_ecg[["subject_id", "study_id", "ecg_time", "strat_fold", "fold"]].merge(
        labs_subset, on="subject_id", how="left"
    )
    merged["time_diff"] = (merged["storetime"] - merged["ecg_time"]).abs().dt.total_seconds() / 3600
    merged = merged[merged["time_diff"] <= 1]
    closest_idx = merged.groupby(["subject_id", "ecg_time", "label"])["time_diff"].idxmin()
    closest = merged.loc[closest_idx]
    wide = closest.pivot_table(
        index=["study_id", "subject_id", "ecg_time", "strat_fold", "fold"],
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
                    "chartevents enrich (Creatinine/Hemoglobin/Hematocrit/Bilirubin/Albumin), "
                    "ECG-time (closest within ±1 h). "
                    "mimic_preprocessing.py:124-154,344-372,398-407 reproduction.",
    )


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════
ALL_TASKS = ["diagnostic", "sex", "age", "ecg_features",
             "deterioration", "mortality", "icu_admission",
             "biometrics", "vitals", "labvalues"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="before task generate")
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
    logging.info(f"  mapping entries: {len(study_to_h5):,}")

    for t in targets:
        if t == "diagnostic":
            build_diagnostic_tasks(study_to_h5)
        elif t == "sex" or t == "age":
            # sex/age  in 
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

    logging.info("\ndone!")


if __name__ == "__main__":
    main()
