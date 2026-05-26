#!/bin/bash
# =============================================================
# 전체 모델 벤치마크 + 경험적 부트스트랩 한 번에 백그라운드 실행
#
# - GPU 0~6 (7장) 전부 사용
# - 8 모델 × 17 태스크 × 3 모드 (linear_probe, attention_probe, finetune_linear)
# - 벤치마크 끝나면 자동으로 부트스트랩 4-stage 파이프라인 실행
#
# 사용법:
#   bash run_all_background.sh                       # 새 timestamp로 시작
#   bash run_all_background.sh 20260511_120000       # 기존 timestamp resume
# =============================================================

set -e
SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

RESUME_TS=${1:-}
if [ -n "$RESUME_TS" ]; then
    TIMESTAMP="$RESUME_TS"
else
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
fi
RESULT_DIR="$SCRIPT_DIR/results/$TIMESTAMP"
mkdir -p "$RESULT_DIR"
RUN_LOG="$RESULT_DIR/run_all.log"

# 모든 GPU 사용
ALL_GPUS="0 1 2 3 4 5 6"
ALL_GPUS_CSV="0,1,2,3,4,5,6"

nohup bash -c "
set -e
cd '$SCRIPT_DIR'

echo '======================================================================'
echo ' Stage 1/2: Full Benchmark'
echo '   Timestamp : $TIMESTAMP'
echo '   GPUs      : $ALL_GPUS'
echo '   Started   : '\$(date '+%Y-%m-%d %H:%M:%S')
echo '======================================================================'

GPU_IDS_OVERRIDE='$ALL_GPUS' bash run_full_benchmark.sh all '$TIMESTAMP'

# run_full_benchmark.sh 는 백그라운드로 떨어지지 않고 동기 실행되므로
# 끝나면 바로 부트스트랩으로 진입.

echo ''
echo '======================================================================'
echo ' Stage 2/2: Empirical Bootstrap (4-stage pipeline)'
echo '   GPUs      : $ALL_GPUS_CSV (multi-GPU extract)'
echo '   Started   : '\$(date '+%Y-%m-%d %H:%M:%S')
echo '======================================================================'

WORKERS=\$(nproc) bash run_bootstrap.sh '$RESULT_DIR' '$ALL_GPUS_CSV'

echo ''
echo '======================================================================'
echo ' ALL DONE: '\$(date '+%Y-%m-%d %H:%M:%S')
echo '   Results : $RESULT_DIR'
echo '======================================================================'
" > "$RUN_LOG" 2>&1 &

BG_PID=$!
echo "백그라운드 시작 (PID $BG_PID)"
echo "  Timestamp : $TIMESTAMP"
echo "  Result dir: $RESULT_DIR"
echo "  Run log   : $RUN_LOG"
echo "  Bench log : $RESULT_DIR/benchmark.log"
echo ""
echo "모니터링:"
echo "  tail -f $RUN_LOG"
echo "  tail -f $RESULT_DIR/benchmark.log | grep -E 'Test AUROC|SKIP|Multi-window'"
echo ""
echo "중단:"
echo "  kill $BG_PID  # (자식 프로세스도 함께 정리하려면 pkill -P $BG_PID)"
