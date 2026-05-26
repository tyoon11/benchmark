#!/usr/bin/env python3
"""results/{TS}/ of all run from test/val metric  CSV by .

- binary task (sex etc.): test_AUROC, test_AUPRC, test_F1
- regression task (age, biometrics, ecg_features, labvalues, vitals): test_MAE, test_R2
- test if absent, val by fallback
"""
import argparse
import csv
import re
from pathlib import Path

DIR_RE = re.compile(r'^(?P<model>.+?)_(?P<task>mimic(?:_[a-z_]+)?)_(?P<mode>linear_probe|attention_probe|finetune_linear|finetune_attention)$')

# task → task_type
# Paper-faithful MIMIC: single joint task (classification_and_regression).
# Per-sub-task metrics are reported as <group>_auroc_macro / <group>_mae_macro
# inside the same test_metrics.txt.
TASK_TYPE = {
    "mimic": "classification_and_regression",
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

        if ttype == "classification_and_regression":
            # Paper joint MIMIC: emit one row per sub-group sliced from the same joint metric file.
            sub_groups = [
                ("mimic_cardiac",       "cls", "cardiac"),
                ("mimic_noncardiac",    "cls", "noncardiac"),
                ("mimic_deterioration", "cls", "deterioration"),
                ("mimic_mortality",     "cls", "mortality"),
                ("mimic_icu",           "cls", "icu"),
                ("mimic_sex",           "cls", "sex"),
                ("mimic_age",           "reg", "age"),
                ("mimic_biometrics",    "reg", "biometrics"),
                ("mimic_ecg_features",  "reg", "ecg_features"),
                ("mimic_labvalues",     "reg", "labvalues"),
                ("mimic_vitals",        "reg", "vitals"),
            ]
            for sub_task, kind, prefix in sub_groups:
                if kind == "cls":
                    rows.append({
                        "model": model, "task": sub_task, "mode": mode,
                        "task_type": "multi_label_binary", "split": split,
                        "AUROC": fmt(m_dict.get(f"{prefix}_auroc_macro")) if m_dict.get(f"{prefix}_auroc_macro") is not None else "",
                        "AUPRC": fmt(m_dict.get(f"{prefix}_auprc_macro")) if m_dict.get(f"{prefix}_auprc_macro") is not None else "",
                        "F1":    fmt(m_dict.get(f"{prefix}_f1_macro")) if m_dict.get(f"{prefix}_f1_macro") is not None else "",
                        "MAE": "", "R2": "", "RMSE": "",
                    })
                else:
                    rows.append({
                        "model": model, "task": sub_task, "mode": mode,
                        "task_type": "regression", "split": split,
                        "AUROC": "", "AUPRC": "", "F1": "",
                        "MAE":  fmt(m_dict.get(f"{prefix}_mae_macro")) if m_dict.get(f"{prefix}_mae_macro") is not None else "",
                        "R2":   fmt(m_dict.get(f"{prefix}_r2_macro")) if m_dict.get(f"{prefix}_r2_macro") is not None else "",
                        "RMSE": fmt(m_dict.get(f"{prefix}_rmse_macro")) if m_dict.get(f"{prefix}_rmse_macro") is not None else "",
                    })
            continue

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
