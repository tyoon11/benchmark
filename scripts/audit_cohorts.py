"""
Cohort / label-vocabulary audit
===============================
Compares what this benchmark actually loads against the original
``ecg-fm-benchmarking`` setup, so residual gaps are visible rather than
silently folded into the reported numbers.

Checks per task:
  * record count in the H5 table vs. the published source-dataset size
  * label-vocabulary size vs. ``num_classes`` in ``main_lite_ecg.py``
  * effect of the ``min_data_length`` cohort filter
  * fold layout (train / val / test sizes under ``fold < max-1`` split)
  * lead order stored in the table and the permutation that will be applied

Run:
    python scripts/audit_cohorts.py
    python scripts/audit_cohorts.py --tasks ptbxl_super chapman
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

from run import load_config  # noqa: E402
from src.dataset import NON_LABEL_COLS  # noqa: E402
from src.leads import (STANDARD_12, build_lead_permutation,  # noqa: E402
                       parse_channel_names)
from src.signal_utils import load_record_lengths  # noqa: E402

# Published sizes of the source datasets (PhysioNet/CinC 2021 + friends).
PUBLISHED_RECORDS = {
    "chapman": 10646, "ningbo": 34905, "cpsc2018": 6877, "cpsc_extra": 3453,
    "georgia": 10344, "ptb": 516, "ptbxl": 21837,
}

DEFAULT_TASKS = [
    "ptb", "ningbo", "cpsc2018", "cpsc_extra", "georgia", "chapman",
    "code15_diag", "ptbxl_all", "ptbxl_super", "ptbxl_sub", "sph_diag", "zzu_pecg",
]


def audit_task(task: str, scan_lengths: bool) -> dict:
    cfg = load_config(task)
    data = cfg.get("data", {})
    tcfg = cfg.get("task", {})

    row = {"task": task, "status": "ok"}

    table_csv = data.get("table_csv")
    if not table_csv or not Path(table_csv).exists():
        row["status"] = f"table missing ({table_csv})"
        return row

    table = pd.read_csv(table_csv, low_memory=False)
    row["h5_records"] = len(table)

    stem = Path(table_csv).stem.replace("_table", "")
    published = PUBLISHED_RECORDS.get(stem)
    if published:
        row["published"] = published
        row["records_delta"] = len(table) - published

    # labels
    label_csv = data.get("label_csv")
    if label_csv and Path(label_csv).exists():
        header = pd.read_csv(label_csv, nrows=0)
        label_cols = data.get("label_cols") or [c for c in header.columns
                                                if c not in NON_LABEL_COLS]
        row["n_labels"] = len(label_cols)
        labels = pd.read_csv(label_csv, usecols=["filepath"], low_memory=False)
        row["labelled_records"] = len(set(labels["filepath"]) & set(table["filepath"]))
    row["num_classes_cfg"] = tcfg.get("num_classes")
    row["num_classes_original"] = tcfg.get("expected_num_classes")

    # lead order
    if "channel_name" in table.columns and len(table):
        src_leads = parse_channel_names(table["channel_name"].iloc[0])
        perm = build_lead_permutation(src_leads, STANDARD_12)
        row["lead_order"] = "|".join(src_leads) if src_leads else "?"
        row["lead_perm"] = "none" if perm is None else ",".join(map(str, perm.tolist()))

    # fold layout
    fold_col = cfg.get("fold", {}).get("col", "strat_fold")
    if fold_col in table.columns:
        folds = table[fold_col]
        mx = int(folds.max())
        row["folds"] = f"{int(folds.min())}..{mx}"
        row["train/val/test"] = (f"{int((folds < mx - 1).sum())}/"
                                 f"{int((folds == mx - 1).sum())}/"
                                 f"{int((folds == mx).sum())}")

    # cohort filter
    min_len = data.get("min_data_length")
    row["min_data_length"] = min_len
    if scan_lengths:
        lengths = load_record_lengths(data["h5_root"], table_csv,
                                      table["filepath"].tolist())
        total = lengths.groupby("filepath")["length"].sum()
        row["median_len"] = int(total.median())
        row["multi_segment"] = int((lengths.groupby("filepath").size() > 1).sum())
        if min_len:
            row["dropped_by_filter"] = int((total < int(min_len)).sum())
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", nargs="*", default=DEFAULT_TASKS)
    ap.add_argument("--scan-lengths", action="store_true",
                    help="also scan H5 headers for record lengths (slow the first time, "
                         "then cached under labels/_cache/lengths)")
    args = ap.parse_args()

    rows = []
    for task in args.tasks:
        try:
            rows.append(audit_task(task, args.scan_lengths))
        except Exception as exc:  # keep auditing the remaining tasks
            rows.append({"task": task, "status": f"ERROR: {type(exc).__name__}: {exc}"})

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)
    print(df.to_string(index=False))

    print("\nNotes")
    print("-----")
    mismatched = [r for r in rows
                  if r.get("num_classes_original") is not None
                  and r.get("n_labels") is not None
                  and r["n_labels"] != r["num_classes_original"]]
    if mismatched:
        print("Label vocabulary differs from the original for: "
              + ", ".join(r["task"] for r in mismatched))
        print("  Cause: the H5 store holds fewer records than the source dataset, which")
        print("  moves borderline labels across the min_cnt=10 cut in build_labels_paper.py.")
        print("  Fix requires re-ingesting the missing records into H5; until then the")
        print("  absolute numbers for these tasks are not comparable to the paper.")
    else:
        print("Label vocabularies match the original for every audited task.")

    short = [r for r in rows if r.get("records_delta", 0) < 0]
    if short:
        print("\nRecord-count shortfall vs. the published datasets:")
        for r in short:
            print(f"  {r['task']:12s} {r['h5_records']:>7,} / {r['published']:>7,} "
                  f"({r['records_delta']:+,})")


if __name__ == "__main__":
    main()
