"""
Compare two benchmark runs
==========================
Builds a model x task table of the primary metric for one run and, when a
baseline run is given, the per-cell delta against it. Written for the
2026-07-31 parity change, where every baseline had to be re-run because the
HEEDB lead order was reaching the pretrained encoders permuted.

Reads test_metrics.txt (per-record / aggregated metrics) plus the inline
bootstrap CI the trainer writes there. Nothing is recomputed.

Run:
    python scripts/compare_runs.py --root results/20260731_110124
    python scripts/compare_runs.py --root results/20260731_110124 \
        --baseline results/20260511_172035 --mode linear_probe --markdown
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODELS = ["ecg_founder", "ecg_jepa", "st_mem", "merl", "ecgfm_ked",
          "hubert_ecg", "ecg_fm", "cpc", "moryecg_a5"]
TASKS = ["ptbxl_super", "ptbxl_all", "ptbxl_sub", "ptbxl_diag", "ptbxl_form",
         "ptbxl_rhythm", "chapman", "chapman_rhythm", "cpsc2018", "cpsc_extra",
         "georgia", "ningbo", "sph_diag", "zzu_pecg", "ptb"]


def read_metrics(run_dir: Path) -> dict:
    f = run_dir / "test_metrics.txt"
    if not f.exists():
        return {}
    out = {}
    for line in f.read_text().splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            try:
                out[k.strip()] = float(v.strip())
            except ValueError:
                pass
    return out


def cell(m: dict, key: str, with_ci: bool) -> str:
    v = m.get(key)
    if v is None:
        return "-"
    lo, hi = m.get(key + "_low"), m.get(key + "_high")
    if with_ci and lo is not None and hi is not None:
        return "%.3f [%.3f,%.3f]" % (v, lo, hi)
    return "%.3f" % v


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="results/<timestamp> to report")
    ap.add_argument("--baseline", default=None, help="results/<timestamp> to diff against")
    ap.add_argument("--mode", default="linear_probe",
                    choices=["linear_probe", "attention_probe",
                             "finetune_linear", "finetune_attention"])
    ap.add_argument("--metric", default="auroc_macro")
    ap.add_argument("--models", nargs="*", default=MODELS)
    ap.add_argument("--tasks", nargs="*", default=TASKS)
    ap.add_argument("--ci", action="store_true", help="include the bootstrap CI per cell")
    ap.add_argument("--markdown", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    base = Path(args.baseline) if args.baseline else None

    grid, deltas = {}, {}
    for model in args.models:
        for task in args.tasks:
            name = "%s_%s_%s" % (model, task, args.mode)
            m = read_metrics(root / name)
            grid[(model, task)] = cell(m, args.metric, args.ci)
            if base is not None:
                b = read_metrics(base / name)
                if args.metric in m and args.metric in b:
                    deltas[(model, task)] = m[args.metric] - b[args.metric]

    width = max(24 if args.ci else 8, max([len(m) for m in args.models] + [8]) + 2)

    def render(values, title):
        lines = []
        if args.markdown:
            lines.append("\n**" + title + "**\n")
            lines.append("| task | " + " | ".join(args.models) + " |")
            lines.append("|---" * (len(args.models) + 1) + "|")
            for t in args.tasks:
                lines.append("| " + t + " | "
                             + " | ".join(values.get((m, t), "-") for m in args.models) + " |")
        else:
            lines.append("\n=== " + title + " ===")
            lines.append("task".ljust(16) + "".join(m.rjust(width) for m in args.models))
            for t in args.tasks:
                lines.append(t.ljust(16)
                             + "".join(values.get((m, t), "-").rjust(width) for m in args.models))
        return "\n".join(lines)

    print(render(grid, "%s / %s - %s" % (args.metric, args.mode, root.name)))

    if base is not None:
        dstr = {k: "%+.3f" % v for k, v in deltas.items()}
        print(render(dstr, "delta vs %s (positive = better)" % base.name))

        print("\n=== per-model summary ===")
        print("model".ljust(14) + "n".rjust(4) + "mean".rjust(9)
              + "min".rjust(9) + "max".rjust(9) + "regressions".rjust(13))
        for model in args.models:
            ds = [deltas[(model, t)] for t in args.tasks if (model, t) in deltas]
            if not ds:
                print(model.ljust(14) + "0".rjust(4) + "-".rjust(9) * 3 + "-".rjust(13))
                continue
            neg = sum(1 for d in ds if d < 0)
            print(model.ljust(14) + str(len(ds)).rjust(4)
                  + ("%+.3f" % (sum(ds) / len(ds))).rjust(9)
                  + ("%+.3f" % min(ds)).rjust(9) + ("%+.3f" % max(ds)).rjust(9)
                  + str(neg).rjust(13))

        all_d = list(deltas.values())
        if all_d:
            neg = sum(1 for d in all_d if d < 0)
            print("\noverall: n=%d  mean=%+.3f  regressions=%d (%.1f%%)"
                  % (len(all_d), sum(all_d) / len(all_d), neg, 100.0 * neg / len(all_d)))

    missing = [m + "/" + t for m in args.models for t in args.tasks
               if grid.get((m, t)) == "-"]
    if missing:
        print("\nnot yet complete (%d): %s%s"
              % (len(missing), ", ".join(missing[:12]), " ..." if len(missing) > 12 else ""))


if __name__ == "__main__":
    main()
