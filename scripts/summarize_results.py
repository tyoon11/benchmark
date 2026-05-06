#!/usr/bin/env python3
"""results/{TS}/ 의 모든 run에서 test/val metric을 읽어 CSV로 정리.

- binary task (sex 등): test_AUROC, test_AUPRC, test_F1
- regression task (age, biometrics, ecg_features, labvalues, vitals): test_MAE, test_R2
- test가 없으면 val로 fallback
"""
import argparse
import csv
import re
from pathlib import Path

DIR_RE = re.compile(r'^(?P<model>.+?)_(?P<task>mimic_[a-z_]+?)_(?P<mode>linear_probe|attention_probe|finetune_linear|finetune_attention)$')

# task → task_type
TASK_TYPE = {
    "mimic_sex": "binary",
    "mimic_cardiac": "multi_label_binary",
    "mimic_noncardiac": "multi_label_binary",
    "mimic_deterioration": "multi_label_binary",
    "mimic_mortality": "multi_label_binary",
    "mimic_icu_admission": "multi_label_binary",
    "mimic_age": "regression",
    "mimic_biometrics": "regression",
    "mimic_ecg_features": "regression",
    "mimic_labvalues": "regression",
    "mimic_vitals": "regression",
}


def read_metrics(path: Path) -> dict:
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            out[k.strip()] = float(v.strip())
        except ValueError:
            pass
    return out


def fmt(x):
    if isinstance(x, float):
        return f"{x:.4f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir", type=Path)
    ap.add_argument("-o", "--output", type=Path, default=None)
    args = ap.parse_args()

    if not args.results_dir.is_dir():
        raise SystemExit(f"not a dir: {args.results_dir}")

    out_path = args.output or args.results_dir / "summary.csv"

    rows = []
    for d in sorted(args.results_dir.iterdir()):
        if not d.is_dir():
            continue
        m = DIR_RE.match(d.name)
        if not m:
            continue
        model = m.group("model")
        task = m.group("task")
        mode = m.group("mode")
        ttype = TASK_TYPE.get(task, "?")

        test_m = read_metrics(d / "test_metrics.txt")
        val_m = read_metrics(d / "val_metrics.txt")
        # prefer test, fallback to val
        m_dict = test_m if test_m else val_m
        split = "test" if test_m else ("val" if val_m else "none")

        if ttype == "regression":
            primary = m_dict.get("mae_macro", "")
            primary_name = "MAE"
            secondary = m_dict.get("r2_macro", "")
            secondary_name = "R2"
            third = m_dict.get("rmse_macro", "")
            third_name = "RMSE"
        else:  # binary or multi_label_binary
            primary = m_dict.get("auroc_macro", "")
            primary_name = "AUROC"
            secondary = m_dict.get("auprc_macro", "")
            secondary_name = "AUPRC"
            third = m_dict.get("f1_macro", "")
            third_name = "F1"

        rows.append({
            "model": model,
            "task": task,
            "mode": mode,
            "task_type": ttype,
            "split": split,
            f"{primary_name}": fmt(primary) if primary != "" else "",
            f"{secondary_name}": fmt(secondary) if secondary != "" else "",
            f"{third_name}": fmt(third) if third != "" else "",
            "AUROC": fmt(m_dict.get("auroc_macro")) if m_dict.get("auroc_macro") is not None else "",
            "AUPRC": fmt(m_dict.get("auprc_macro")) if m_dict.get("auprc_macro") is not None else "",
            "F1": fmt(m_dict.get("f1_macro")) if m_dict.get("f1_macro") is not None else "",
            "MAE": fmt(m_dict.get("mae_macro")) if m_dict.get("mae_macro") is not None else "",
            "R2": fmt(m_dict.get("r2_macro")) if m_dict.get("r2_macro") is not None else "",
            "RMSE": fmt(m_dict.get("rmse_macro")) if m_dict.get("rmse_macro") is not None else "",
        })

    if not rows:
        raise SystemExit(f"no runs found in {args.results_dir}")

    # dedupe duplicate columns from key collision (primary == AUROC for binary etc.)
    fieldnames = ["model", "task", "mode", "task_type", "split", "AUROC", "AUPRC", "F1", "MAE", "R2", "RMSE"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows → {out_path}")

    # summary
    n_test = sum(1 for r in rows if r["split"] == "test")
    n_val = sum(1 for r in rows if r["split"] == "val")
    by_mode = {}
    for r in rows:
        if r["split"] != "none":
            by_mode.setdefault(r["mode"], []).append(r)
    print(f"\n[summary]")
    print(f"  total run dirs:   {len(rows)}")
    print(f"  with test_metrics: {n_test}")
    print(f"  with val only:    {n_val}")
    for mode, rs in sorted(by_mode.items()):
        print(f"  {mode}: {len(rs)} done")


if __name__ == "__main__":
    main()
