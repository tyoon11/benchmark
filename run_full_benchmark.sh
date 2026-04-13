#!/bin/bash
# =============================================================
# 전체 모델 × 전체 태스크 벤치마크
#
# 사용법:
#   bash run_full_benchmark.sh              # 전체 (linear_probe, 7 GPU)
#   bash run_full_benchmark.sh linear_probe # 특정 모드만
#   bash run_full_benchmark.sh all          # 4가지 모드 전부
# =============================================================

set -e

EVAL_MODE=${1:-linear_probe}
GPUS="0,1,2,3,4,5,6"
N_GPUS=7

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"
mkdir -p results

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

# ─────────────────────────────────────────────────────────────
# 모델 정의
# ─────────────────────────────────────────────────────────────
MODEL_NAMES=(
    ecg_founder
    ecg_jepa
    st_mem
    merl
    ecgfm_ked
    hubert_ecg
    cpc
)

MODEL_CLS=(
    "src.encoders.ecg_founder.ECGFounderEncoder"
    "src.encoders.ecg_jepa.ECGJEPAEncoder"
    "src.encoders.st_mem.StMemEncoder"
    "src.encoders.merl.MerlResNetEncoder"
    "src.encoders.ecgfm_ked.EcgFmKEDEncoder"
    "src.encoders.hubert_ecg.HuBERTECGEncoder"
    "src.encoders.cpc.CPCEncoder"
)

MODEL_CKPT=(
    "/home/irteam/ddn-opendata1/model/ECGFMs/ecg_founder/12_lead_ECGFounder.pth"
    "/home/irteam/ddn-opendata1/model/ECGFMs/ecg_jepa/multiblock_epoch100.pth"
    "/home/irteam/ddn-opendata1/model/ECGFMs/st_mem/st_mem_vit_base_full.pth"
    "/home/irteam/ddn-opendata1/model/ECGFMs/merl/res18_best_encoder.pth"
    "/home/irteam/ddn-opendata1/model/ECGFMs/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt"
    "/home/irteam/ddn-opendata1/model/ECGFMs/hubert_ecg/hubert_ecg_base.safetensors"
    "/home/irteam/ddn-opendata1/model/ECGFMs/cpc/last_11597276.ckpt"
)

# ─────────────────────────────────────────────────────────────
# 태스크 정의
# ─────────────────────────────────────────────────────────────
TASKS=(
    ptb
    ningbo
    cpsc2018
    cpsc_extra
    georgia
    chapman
    code15
    ptbxl_all
    ptbxl_super
    ptbxl_diag
    ptbxl_sub
    zzu_pecg
)

# ─────────────────────────────────────────────────────────────
# 설정
# ─────────────────────────────────────────────────────────────
EPOCHS=50
FINETUNE_EPOCHS=30
FINETUNE_LR="5e-4"

if [ "$EVAL_MODE" = "all" ]; then
    MODES=(linear_probe attention_probe finetune_linear finetune_attention)
else
    MODES=($EVAL_MODE)
fi

# ─────────────────────────────────────────────────────────────
# 실행
# ─────────────────────────────────────────────────────────────
log "======================================"
log "Full Benchmark: ${#MODEL_NAMES[@]} models × ${#TASKS[@]} tasks × ${#MODES[@]} modes"
log "  GPUs: $GPUS ($N_GPUS)"
log "  Modes: ${MODES[*]}"
log "======================================"

TOTAL=$((${#MODEL_NAMES[@]} * ${#TASKS[@]} * ${#MODES[@]}))
DONE=0

for mode in "${MODES[@]}"; do
    epochs=$EPOCHS
    lr_arg=""
    if [[ "$mode" == finetune_* ]]; then
        epochs=$FINETUNE_EPOCHS
        lr_arg="--lr $FINETUNE_LR"
    fi

    for i in "${!MODEL_NAMES[@]}"; do
        model_name="${MODEL_NAMES[$i]}"
        encoder_cls="${MODEL_CLS[$i]}"
        encoder_ckpt="${MODEL_CKPT[$i]}"

        log ""
        log "══════ $model_name / $mode ══════"

        for task in "${TASKS[@]}"; do
            DONE=$((DONE + 1))
            log_file="results/${model_name}_${task}_${mode}.log"
            log "[$DONE/$TOTAL] $model_name / $task / $mode"

            # Multi-GPU로 실행
            CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$N_GPUS run.py \
                --task "$task" --eval_mode "$mode" \
                --encoder_cls "$encoder_cls" \
                --encoder_ckpt "$encoder_ckpt" \
                --epochs $epochs $lr_arg \
                > "$log_file" 2>&1 || {
                log "  [FAIL] $model_name / $task / $mode — see $log_file"
                continue
            }

            # 결과 요약
            auroc=$(grep "Best val AUROC" "$log_file" 2>/dev/null | grep -oP '0\.\d+' | head -1)
            test_auroc=$(grep "Test AUROC" "$log_file" 2>/dev/null | grep -oP '0\.\d+' | head -1)
            log "  val_auroc=${auroc:-N/A} test_auroc=${test_auroc:-N/A}"
        done
    done
done

# ─────────────────────────────────────────────────────────────
# 최종 결과 테이블
# ─────────────────────────────────────────────────────────────
log ""
log "======================================"
log "최종 결과 (Test AUROC)"
log "======================================"

for mode in "${MODES[@]}"; do
    log ""
    log "── $mode ──"
    printf "%-20s" "Task"
    for model_name in "${MODEL_NAMES[@]}"; do
        printf "  %-14s" "$model_name"
    done
    echo ""
    printf "%s\n" "$(printf '─%.0s' {1..120})"

    for task in "${TASKS[@]}"; do
        printf "%-20s" "$task"
        for model_name in "${MODEL_NAMES[@]}"; do
            log_file="results/${model_name}_${task}_${mode}.log"
            auroc=$(grep "Test AUROC" "$log_file" 2>/dev/null | grep -oP '0\.\d+' | head -1)
            [ -z "$auroc" ] && auroc=$(grep "Best val AUROC" "$log_file" 2>/dev/null | grep -oP '0\.\d+' | head -1)
            printf "  %-14s" "${auroc:-—}"
        done
        echo ""
    done
done

log ""
log "완료! 로그: results/{model}_{task}_{mode}.log"
