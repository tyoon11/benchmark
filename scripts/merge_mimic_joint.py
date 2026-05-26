"""
Merge per-task MIMIC label CSVs into a single joint label file.
================================================================
Paper (`mimic_preprocessing.py`) trains MIMIC as a single multi-task model
with output = `label_diagnostic + label_deterioration + label_sex + label_metadata`
(158 + 6 + 1 + 35 = 200 dims), composite loss = BCE(cls) + L1(reg),
NaN-masked per position. Cohort = is_diagnostic==1 (the cardiac/diag set).

This script reproduces that joint label file by left-merging the existing
per-task CSVs onto the cardiac (is_diagnostic==1) cohort. Z-normalization of
the 35 regression columns happens at train time in run.py using train fold
stats — not here.

Output: labels/mimic_paper_labels.csv  (~116k rows × 203 cols)
        labels/mimic_paper_labels.json  (column-group schema)

Usage:
    python scripts/merge_mimic_joint.py
"""
import json
import logging
from pathlib import Path

import pandas as pd

LABEL_DIR = Path(__file__).resolve().parent.parent / "labels"

# Paper metadata_cols (35) — ordering matches mimic_preprocessing.py:413-417
REG_COLS = [
    "age",
    "Height (Inches)", "Weight (Lbs)", "BMI (kg/m2)",
    "RR", "QRS", "QT", "QTc", "P_wave_axis", "QRS_axis", "T_wave_axis",
    "PT", "Albumin", "Anion Gap", "Bicarbonate", "Bilirubin, Total",
    "Calcium, Total", "Creatinine", "Ferritin", "Urea Nitrogen",
    "Hematocrit", "Hemoglobin", "Lymphocytes", "MCHC", "RDW",
    "Red Blood Cells", "RDW-SD", "Creatine Kinase (CK)", "NTproBNP",
    "dbp", "heartrate", "o2sat", "resprate", "sbp", "temperature",
]
DET_COLS = [
    "deterioration_severe_hypoxemia",
    "deterioration_ecmo",
    "deterioration_vasopressors",
    "deterioration_inotropes",
    "deterioration_mechanical_ventilation",
    "deterioration_cardiac_arrest",
]
SEX_COL = "is_male"


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")

    # 1. cardiac (158 diag cols) → cohort base
    card_path = LABEL_DIR / "mimic_cardiac_paper_labels.csv"
    if not card_path.exists():
        raise FileNotFoundError(
            f"{card_path} not found. Run build_mimic_labels.py --all first.")
    card = pd.read_csv(card_path, low_memory=False)
    diag_cols = [c for c in card.columns if c not in ("filepath", "strat_fold", "fold")]
    logging.info(f"cohort (cardiac/is_diagnostic): {len(card):,} rows × {len(diag_cols)} diag cols")
    out = card.copy()

    # 2. deterioration (6) — left-merge, NaN where ECG not in MDS-ED
    det_path = LABEL_DIR / "mimic_deterioration_paper_labels.csv"
    det = pd.read_csv(det_path, low_memory=False)[["filepath"] + DET_COLS]
    out = out.merge(det, on="filepath", how="left")
    logging.info(f"+ deterioration: matched {out[DET_COLS[0]].notna().sum():,}/{len(out):,}")

    # 3. sex (1)
    sex = pd.read_csv(LABEL_DIR / "mimic_sex_paper_labels.csv",
                      low_memory=False)[["filepath", SEX_COL]]
    out = out.merge(sex, on="filepath", how="left")
    logging.info(f"+ sex: matched {out[SEX_COL].notna().sum():,}/{len(out):,}")

    # 4. regression sources — age / biometrics / ecg_features / vitals / labvalues
    reg_sources = [
        ("mimic_age_paper_labels.csv", ["age"]),
        ("mimic_biometrics_paper_labels.csv",
            ["Height (Inches)", "Weight (Lbs)", "BMI (kg/m2)"]),
        ("mimic_ecg_features_paper_labels.csv",
            ["RR", "QRS", "QT", "QTc", "P_wave_axis", "QRS_axis", "T_wave_axis"]),
        ("mimic_labvalues_paper_labels.csv",
            ["PT", "Albumin", "Anion Gap", "Bicarbonate", "Bilirubin, Total",
             "Calcium, Total", "Creatinine", "Ferritin", "Urea Nitrogen",
             "Hematocrit", "Hemoglobin", "Lymphocytes", "MCHC", "RDW",
             "Red Blood Cells", "RDW-SD", "Creatine Kinase (CK)", "NTproBNP"]),
        ("mimic_vitals_paper_labels.csv",
            ["dbp", "heartrate", "o2sat", "resprate", "sbp", "temperature"]),
    ]
    for fname, cols in reg_sources:
        src = pd.read_csv(LABEL_DIR / fname, low_memory=False)[["filepath"] + cols]
        out = out.merge(src, on="filepath", how="left")
        matched = out[cols[0]].notna().sum()
        logging.info(f"+ {fname}: matched {matched:,}/{len(out):,}")

    # 5. reorder columns: meta + cls (diag 158, det 6, sex 1) + reg (35)
    cls_cols = diag_cols + DET_COLS + [SEX_COL]
    meta_cols = [c for c in ("filepath", "strat_fold", "fold") if c in out.columns]
    out = out[meta_cols + cls_cols + REG_COLS]

    # 6. cast cls to numeric (1/0/NaN). booleans → int, NaN preserved
    for c in cls_cols:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("Float32")
    for c in REG_COLS:
        out[c] = pd.to_numeric(out[c], errors="coerce").astype("float32")

    out_path = LABEL_DIR / "mimic_paper_labels.csv"
    out.to_csv(out_path, index=False)
    logging.info(f"\nSaved: {out_path.name}")
    logging.info(f"  rows: {len(out):,}")
    logging.info(f"  cls (binary, NaN-maskable): {len(cls_cols)} "
                 f"(diag {len(diag_cols)} + det {len(DET_COLS)} + sex 1)")
    logging.info(f"  reg (z-normed at train time): {len(REG_COLS)}")
    logging.info(f"  fold split: strat_fold 0..17 train, 18 val, 19 test")

    # Paper-style MIMIC sub-task groups (Table 3/4/7/8): same joint model,
    # metrics reported per sub-group. Each group's "kind" tells the trainer
    # how to slice the 200-dim output (binary AUROC vs regression MAE).
    schema = {
        "dataset": "mimic",
        "source": "joint MIMIC task (paper reproduction). Merged from per-task label CSVs. "
                  "Cohort = is_diagnostic==1 (cardiac).",
        "task_type": "classification_and_regression",
        "n_cls": len(cls_cols),
        "n_reg": len(REG_COLS),
        "cls_cols": cls_cols,
        "reg_cols": REG_COLS,
        "report_groups": {
            "cardiac":       {"kind": "cls", "cols": diag_cols},
            "deterioration": {"kind": "cls", "cols": DET_COLS},
            "sex":           {"kind": "cls", "cols": [SEX_COL]},
            "age":           {"kind": "reg", "cols": ["age"]},
            "biometrics":    {"kind": "reg", "cols": ["Height (Inches)", "Weight (Lbs)",
                                                       "BMI (kg/m2)"]},
            "ecg_features":  {"kind": "reg", "cols": ["RR", "QRS", "QT", "QTc",
                                                       "P_wave_axis", "QRS_axis", "T_wave_axis"]},
            "labvalues":     {"kind": "reg", "cols": ["PT", "Albumin", "Anion Gap", "Bicarbonate",
                                                       "Bilirubin, Total", "Calcium, Total",
                                                       "Creatinine", "Ferritin", "Urea Nitrogen",
                                                       "Hematocrit", "Hemoglobin", "Lymphocytes",
                                                       "MCHC", "RDW", "Red Blood Cells", "RDW-SD",
                                                       "Creatine Kinase (CK)", "NTproBNP"]},
            "vitals":        {"kind": "reg", "cols": ["dbp", "heartrate", "o2sat", "resprate",
                                                       "sbp", "temperature"]},
        },
    }
    with open(LABEL_DIR / "mimic_paper_labels.json", "w") as f:
        json.dump(schema, f, indent=2, ensure_ascii=False)
    logging.info(f"  schema: mimic_paper_labels.json")


if __name__ == "__main__":
    main()
