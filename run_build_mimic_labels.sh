#!/usr/bin/env bash
# MIMIC 11개 paper task 라벨 병렬 생성
# ─────────────────────────────────────────────────────────────────
# Stage 1: chartevents-독립 task 6개 동시 실행 (~5분 내)
# Stage 2: biometrics 단독 실행 (chartevents 캐시 생성, ~30분)
# Stage 3: vitals + labvalues 동시 실행 (캐시 재사용, ~5분)
#
# 사용:
#   ./run_build_mimic_labels.sh
#   ./run_build_mimic_labels.sh --skip-stage1   # 이미 끝낸 stage skip
#   PARALLEL_JOBS=4 ./run_build_mimic_labels.sh # stage1 동시 실행 수
#
# 환경:
#   conda env로 pandas/numpy/wfdb 등 사용 가능해야 함.
#   raw 파일 위치는 build_mimic_labels.py 상단 경로에서 확인.

set -euo pipefail

# ── 설정 ────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILDER="$SCRIPT_DIR/scripts/build_mimic_labels.py"
LOG_DIR="$SCRIPT_DIR/labels/_logs"
mkdir -p "$LOG_DIR"

PARALLEL_JOBS="${PARALLEL_JOBS:-6}"
SKIP_STAGE1="${SKIP_STAGE1:-false}"
SKIP_STAGE2="${SKIP_STAGE2:-false}"
SKIP_STAGE3="${SKIP_STAGE3:-false}"

# 옵션 파싱
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

# Conda 자동 활성화 — tykim 환경 사용 (project convention)
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if [[ "${CONDA_DEFAULT_ENV:-}" != "tykim" ]]; then
    conda activate tykim 2>/dev/null || echo "⚠ conda env 'tykim' 활성화 실패 — 현재 env 유지"
  fi
fi
echo "[$(date '+%H:%M:%S')] python: $(which python)"
python -c "import pandas, numpy" || { echo "❌ pandas/numpy 없음"; exit 1; }

# ── 헬퍼 ────────────────────────────────────────────────────────
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

# ── Stage 1: chartevents 독립 task 병렬 ──────────────────────────
STAGE1_TASKS=(diagnostic sex ecg_features deterioration mortality icu_admission)
if [[ "$SKIP_STAGE1" != "true" ]]; then
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo " Stage 1: ${#STAGE1_TASKS[@]}개 task 병렬 실행 (jobs=$PARALLEL_JOBS)"
  echo "════════════════════════════════════════════════════════════════"
  STAGE1_START=$(date +%s)

  # GNU parallel 있으면 사용, 없으면 xargs -P
  if command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${STAGE1_TASKS[@]}" | parallel -j "$PARALLEL_JOBS" --halt now,fail=1 run_task {}
  else
    printf '%s\n' "${STAGE1_TASKS[@]}" | xargs -n1 -P"$PARALLEL_JOBS" -I{} bash -c 'run_task "$@"' _ {}
  fi

  echo "[$(ts)] Stage 1 완료 ($(( $(date +%s) - STAGE1_START ))s)"
else
  echo "[$(ts)] Stage 1 skip"
fi

# ── Stage 2: biometrics 단독 (chartevents 캐시 생성) ────────────
if [[ "$SKIP_STAGE2" != "true" ]]; then
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo " Stage 2: biometrics 단독 (chartevents 캐시 생성, 시간 소요)"
  echo "════════════════════════════════════════════════════════════════"
  STAGE2_START=$(date +%s)
  run_task biometrics
  echo "[$(ts)] Stage 2 완료 ($(( $(date +%s) - STAGE2_START ))s)"
else
  echo "[$(ts)] Stage 2 skip"
fi

# ── Stage 3: vitals + labvalues 동시 ────────────────────────────
STAGE3_TASKS=(vitals labvalues)
if [[ "$SKIP_STAGE3" != "true" ]]; then
  echo ""
  echo "════════════════════════════════════════════════════════════════"
  echo " Stage 3: ${#STAGE3_TASKS[@]}개 task 병렬 (chartevents 캐시 재사용)"
  echo "════════════════════════════════════════════════════════════════"
  STAGE3_START=$(date +%s)
  if command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${STAGE3_TASKS[@]}" | parallel -j 2 --halt now,fail=1 run_task {}
  else
    printf '%s\n' "${STAGE3_TASKS[@]}" | xargs -n1 -P2 -I{} bash -c 'run_task "$@"' _ {}
  fi
  echo "[$(ts)] Stage 3 완료 ($(( $(date +%s) - STAGE3_START ))s)"
else
  echo "[$(ts)] Stage 3 skip"
fi

# ── 요약 ────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo " 라벨 파일 요약"
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
echo "[$(ts)] 전체 완료. 개별 로그: $LOG_DIR/"
