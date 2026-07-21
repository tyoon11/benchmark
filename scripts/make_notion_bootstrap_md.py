"""
Assemble a Notion-pasteable bootstrap_results.md from the per-mode marked
summary CSVs produced by make_summary_table.py (which uses the PAIRED bootstrap
rank, i.e. the original-paper convention: rank-1 = tied with best because the
95% paired-bootstrap CI of the pairwise difference contains 0).

Re-marks cells to the requested visual convention:
    **x**  (single best point, paper bold)        → **<u>x</u>**   (bold + underline = best)
    __x__  (rank-1 tied group, paper underline)    → **x**          (bold = CI ∋ best)
    x      (else)                                   → x

Underline uses <u></u> (Notion renders inline HTML on paste; markdown has no
native underline).

Usage:
  python scripts/make_notion_bootstrap_md.py --root <RESULT_ROOT> [--out bootstrap_results.md]
"""
import argparse
import csv
import re
from pathlib import Path

MODE_ORDER = ["linear_probe", "attention_probe", "finetune_linear"]
MODE_DISPLAY = {
    "linear_probe":    "Linear Probe",
    "attention_probe": "Attention Probe",
    "finetune_linear": "Fine-tune (Linear)",
}

_BEST = re.compile(r"^\*\*(.+)\*\*$")   # **x** → best
_TIED = re.compile(r"^__(.+)__$")        # __x__ → tied (rank-1, paired CI ∋ best)


def remark(cell: str) -> str:
    cell = (cell or "").strip()
    m = _BEST.match(cell)
    if m:
        return f"**<u>{m.group(1)}</u>**"   # best → bold + underline
    m = _TIED.match(cell)
    if m:
        return f"**{m.group(1)}**"          # tied → bold
    return cell


def table_from_marked(csv_path: Path) -> str:
    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return ""
    header = rows[0]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows[1:]:
        cells = list(r) + [""] * (len(header) - len(r))
        out = [cells[0], cells[1]] + [remark(c) for c in cells[2:]]
        lines.append("| " + " | ".join(out) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="bootstrap_results.md")
    args = ap.parse_args()
    root = Path(args.root)
    pair_dir = root / "pairwise"

    doc = [
        "# ECG Foundation-Model Benchmark — Bootstrap Results",
        "",
        "Metric: macro-AUROC (↑ higher better) for classification, z-norm MAE "
        "(↓ lower better) for regression. 95% CIs from 1000-iteration empirical bootstrap; "
        "ties from paired bootstrap of the pairwise difference (original-paper method).",
        "",
        "**Legend:** **<u>bold + underline</u>** = best (point estimate) · "
        "**bold** = tied with best (95% paired-bootstrap CI of the difference contains 0) · "
        "plain = significantly below best.",
        "",
    ]
    modes = [m for m in MODE_ORDER if (pair_dir / f"summary_{m}_marked.csv").exists()]
    for p in sorted(pair_dir.glob("summary_*_marked.csv")):
        m = p.name[len("summary_"):-len("_marked.csv")]
        if m not in modes:
            modes.append(m)
    if not modes:
        raise SystemExit(f"No summary_*_marked.csv under {pair_dir} — run make_summary_table.py first")

    for mode in modes:
        doc.append(f"## {MODE_DISPLAY.get(mode, mode)}")
        doc.append("")
        doc.append(table_from_marked(pair_dir / f"summary_{mode}_marked.csv"))
        doc.append("")

    out_path = root / args.out
    out_path.write_text("\n".join(doc) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
