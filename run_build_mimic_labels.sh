#!/usr/bin/env bash
# MIMIC 11 paper task label parallel generate
# ─────────────────────────────────────────────────────────────────
# Stage 1: chartevents- task 6 concurrent run (~5 )
# Stage 2: biometrics single run (chartevents cache generate, ~30)
# Stage 3: vitals + labvalues concurrent run (cache re-use, ~5)
#
# use:
#   ./run_build_mimic_labels.sh
#   ./run_build_mimic_labels.sh --skip-stage1   # already  stage skip
#   PARALLEL_JOBS=4 ./run_build_mimic_labels.sh # stage1 concurrent run 
#
# :
#   conda env by pandas/numpy/wfdb etc. use available.
#   raw file location build_mimic_labels.py just path from confirm.

set -euo pipefail

# ── config ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILDER="$SCRIPT_DIR/scripts/build_mimic_labels.py"
LOG_DIR="$SCRIPT_DIR/labels/_logs"
mkdir -p "$LOG_DIR"

PARALLEL_JOBS="${PARALLEL_JOBS:-6}"
SKIP_STAGE1="${SKIP_STAGE1:-false}"
SKIP_STAGE2="${SKIP_STAGE2:-false}"
SKIP_STAGE3="${SKIP_STAGE3:-false}"

# option parsing
while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-stage1) SKIP_STAGE1=true; shift ;;
    --skip-stage2) SKIP_STAGE2=true; shift ;;
    --skip-stage3) SKIP_STAGE3=true; shift ;;
    -j|--jobs)     PARALLEL_JOBS="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,17p' "$0"; exit 0 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Auto-activate the project conda env (see environment.yaml: name=ecg-fm)
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "ecg-fm" ]]; then
    conda activate ecg-fm 2>/dev/null || echo "⚠ conda env 'ecg-fm'  failure — keep current env"
  fi
fi
echo "[$(date '+%H:%M:%S')] python: $(which python)"
python -c "import pandas, numpy" || { echo "❌ pandas/numpy none"; exit 1; }

# ── helper ────────────────────────────────────────────────────────
ts() { date '+%Y-%m-%d %H:%M:%S'; }
run_task() {
  local task="$1"
  local log="$LOG_DIR/build_${task}.log"
  echo "[$(ts)] [$task] start  → $log"
  if python "$BUILDER" --task "$task" >"$log" 2>&1; then
    echo "[$(ts)] [$task] ✅ done"
  else
    echo "[$(ts)] [$task] ❌ FAILED (see $log)"
    return 1
  fi
}
export -f run_task ts
export BUILDER LOG_DIR

# ── Stage 1: chartevents  task parallel ──────────────────────────
STAGE1_TASKS=(diagnostic sex ecg_features deterioration mortality icu_admission)
if [[ "$SKIP_STAGE1" != "true" ]]; then
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo " Stage 1: ${#STAGE1_TASKS[@]} task parallel run (jobs=$PARALLEL_JOBS)"
  echo "════════════════════════════════════════════════════════════════"
  STAGE1_START=$(date +%s)

  # GNU parallel if present, use, if absent, xargs -P
  if command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${STAGE1_TASKS[@]}" | parallel -j "$PARALLEL_JOBS" --halt now,fail=1 run_task {}
  else
    printf '%s\n' "${STAGE1_TASKS[@]}" | xargs -n1 -P"$PARALLEL_JOBS" -I{} bash -c 'run_task "$@"' _ {}
  fi

  echo "[$(ts)] Stage 1 done ($(( $(date +%s) - STAGE1_START ))s)"
else
  echo "[$(ts)] Stage 1 skip"
fi

# ── Stage 2: biometrics single (chartevents cache generate) ────────────
if [[ "$SKIP_STAGE2" != "true" ]]; then
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo " Stage 2: biometrics single (chartevents cache generate,  )"
  echo "════════════════════════════════════════════════════════════════"
  STAGE2_START=$(date +%s)
  run_task biometrics
  echo "[$(ts)] Stage 2 done ($(( $(date +%s) - STAGE2_START ))s)"
else
  echo "[$(ts)] Stage 2 skip"
fi

# ── Stage 3: vitals + labvalues concurrent ────────────────────────────
STAGE3_TASKS=(vitals labvalues)
if [[ "$SKIP_STAGE3" != "true" ]]; then
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo " Stage 3: ${#STAGE3_TASKS[@]} task parallel (chartevents cache re-use)"
  echo "════════════════════════════════════════════════════════════════"
  STAGE3_START=$(date +%s)
  if command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${STAGE3_TASKS[@]}" | parallel -j 2 --halt now,fail=1 run_task {}
  else
    printf '%s\n' "${STAGE3_TASKS[@]}" | xargs -n1 -P2 -I{} bash -c 'run_task "$@"' _ {}
  fi
  echo "[$(ts)] Stage 3 done ($(( $(date +%s) - STAGE3_START ))s)"
else
  echo "[$(ts)] Stage 3 skip"
fi

# ── summary ────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " label file summary"
echo "════════════════════════════════════════════════════════════════"
for f in cardiac noncardiac sex age ecg_features deterioration mortality \
         icu_admission biometrics vitals labvalues; do
  csv="$SCRIPT_DIR/labels/mimic_${f}_paper_labels.csv"
  if [[ -f "$csv" ]]; then
    rows=$(wc -l <"$csv")
    size=$(du -h "$csv" | cut -f1)
    printf "  ✅ %-15s %8s  %10s rows  (%s)\n" "$f" "$size" "$rows" "$(basename $csv)"
  else
    printf "  ❌ %-15s NOT GENERATED\n" "$f"
  fi
done

echo ""
echo "[$(ts)] all done. per log: $LOG_DIR/"
