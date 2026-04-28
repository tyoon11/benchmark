#!/bin/bash
# =============================================================
# ECG Downstream Benchmark 일괄 실행
#
# 사용법:
#   bash run_benchmark.sh                          # 전체 (linear_probe, GPU 0)
#   bash run_benchmark.sh linear_probe 0           # linear_probe, GPU 0
#   bash run_benchmark.sh finetune_attention 0,1    # finetune + attention, GPU 0,1 (DDP)
#   bash run_benchmark.sh all 0,1,2,3              # 4가지 모드 전부
# =============================================================

set -e

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
ENCODER_CLS="src.encoders.ecg_jepa.ECGJEPAEncoder"
ENCODER_CKPT="/home/irteam/local-node-d/tykim/ecg_jepa/weights/ecg_jepa_heedb_20260402_095909/best.pth"
EPOCHS=50
FINETUNE_EPOCHS=30
FINETUNE_LR="5e-4"

EVAL_MODE=${1:-linear_probe}   # linear_probe | attention_probe | finetune_linear | finetune_attention | all
GPUS=${2:-0}                   # GPU IDs (쉼표 구분, 예: 0,1,2,3)

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"
mkdir -p results

# 태스크 목록 (논문 벤치마크 17개 — run_full_benchmark.sh와 동일)
TASKS=(
    # Adult ECG interpretation
    ptb
    ningbo
    cpsc2018
    cpsc_extra
    georgia
    chapman
    chapman_rhythm
    code15
    ptbxl_all
    ptbxl_super
    ptbxl_diag
    ptbxl_sub
    ptbxl_form
    ptbxl_rhythm
    sph_diag
    # Pediatric ECG interpretation
    zzu_pecg
    # Cardiac structure & function (NumPy loader)
    echonext
)

# ─────────────────────────────────────────────────────────────
# 유틸
# ─────────────────────────────────────────────────────────────
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

count_gpus() {
    echo "$1" | tr ',' '\n' | wc -l
}

run_task() {
    local task=$1
    local mode=$2
    local gpus=$3

    local n_gpus=$(count_gpus "$gpus")
    local epochs=$EPOCHS
    local lr_arg=""

    if [[ "$mode" == finetune_* ]]; then
        epochs=$FINETUNE_EPOCHS
        lr_arg="--lr $FINETUNE_LR"
    fi

    local log_file="results/${task}_${mode}.log"
    log "[$task] $mode (GPU=$gpus, epochs=$epochs)"

    if [ "$n_gpus" -gt 1 ]; then
        # Multi-GPU (DDP)
        CUDA_VISIBLE_DEVICES=$gpus torchrun --nproc_per_node=$n_gpus run.py \
            --task "$task" --eval_mode "$mode" \
            --encoder_cls "$ENCODER_CLS" \
            --encoder_ckpt "$ENCODER_CKPT" \
            --epochs $epochs $lr_arg \
            > "$log_file" 2>&1
    else
        # Single GPU
        CUDA_VISIBLE_DEVICES=$gpus python run.py \
            --task "$task" --eval_mode "$mode" \
            --encoder_cls "$ENCODER_CLS" \
            --encoder_ckpt "$ENCODER_CKPT" \
            --epochs $epochs $lr_arg \
            > "$log_file" 2>&1
    fi

    # 결과 요약
    local auroc=$(grep "Best val AUROC" "$log_file" 2>/dev/null | grep -oP '[\d.]+' | head -1)
    log "[$task] $mode → AUROC=${auroc:-N/A}"
}

# ─────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────
if [ "$EVAL_MODE" = "all" ]; then
    MODES=(linear_probe attention_probe finetune_linear finetune_attention)
else
    MODES=($EVAL_MODE)
fi

log "======================================"
log "ECG Downstream Benchmark"
log "  Encoder: $ENCODER_CLS"
log "  Checkpoint: $ENCODER_CKPT"
log "  Tasks: ${#TASKS[@]}개"
log "  Modes: ${MODES[*]}"
log "  GPUs: $GPUS"
log "======================================"

for mode in "${MODES[@]}"; do
    log ""
    log "══════ $mode ══════"
    for task in "${TASKS[@]}"; do
        run_task "$task" "$mode" "$GPUS"
    done
done

# ─────────────────────────────────────────────────────────────
# 최종 요약
# ─────────────────────────────────────────────────────────────
log ""
log "======================================"
log "최종 결과 요약"
log "======================================"
printf "%-20s" "Task"
for mode in "${MODES[@]}"; do
    printf "  %-20s" "$mode"
done
echo ""
printf "%s\n" "$(printf '─%.0s' {1..80})"

for task in "${TASKS[@]}"; do
    printf "%-20s" "$task"
    for mode in "${MODES[@]}"; do
        local_log="results/${task}_${mode}.log"
        auroc=$(grep "Best val AUROC" "$local_log" 2>/dev/null | grep -oP '[\d.]+' | head -1)
        printf "  %-20s" "${auroc:-—}"
    done
    echo ""
done

log ""
log "로그: results/*.log"
log "완료!"
