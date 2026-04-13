#!/bin/bash
# =============================================================
# 전체 모델 × 전체 태스크 벤치마크
#
# 모델별로 GPU 1장씩 할당, 병렬 실행.
# 8 모델 → 7 GPU (마지막 모델은 GPU 0 재사용, 순차 대기)
#
# 사용법:
#   bash run_full_benchmark.sh              # linear_probe만
#   bash run_full_benchmark.sh all          # 4가지 모드 전부
#
# 모니터링:
#   tail -f results/{timestamp}/benchmark.log
# =============================================================

EVAL_MODE=${1:-linear_probe}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULT_DIR="results/$TIMESTAMP"
LOG="$RESULT_DIR/benchmark.log"
mkdir -p "$RESULT_DIR"

# ─────────────────────────────────────────────────────────────
# 모델
# ─────────────────────────────────────────────────────────────
MODEL_NAMES=(ecg_founder ecg_jepa st_mem merl ecgfm_ked hubert_ecg ecg_fm cpc)

MODEL_CLS=(
    "src.encoders.ecg_founder.ECGFounderEncoder"
    "src.encoders.ecg_jepa.ECGJEPAEncoder"
    "src.encoders.st_mem.StMemEncoder"
    "src.encoders.merl.MerlResNetEncoder"
    "src.encoders.ecgfm_ked.EcgFmKEDEncoder"
    "src.encoders.hubert_ecg.HuBERTECGEncoder"
    "src.encoders.ecg_fm.ECGFMEncoder"
    "src.encoders.cpc.CPCEncoder"
)

MODEL_CKPT=(
    "/home/irteam/ddn-opendata1/model/ECGFMs/ecg_founder/12_lead_ECGFounder.pth"
    "/home/irteam/ddn-opendata1/model/ECGFMs/ecg_jepa/multiblock_epoch100.pth"
    "/home/irteam/ddn-opendata1/model/ECGFMs/st_mem/st_mem_vit_base_full.pth"
    "/home/irteam/ddn-opendata1/model/ECGFMs/merl/res18_best_encoder.pth"
    "/home/irteam/ddn-opendata1/model/ECGFMs/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt"
    "/home/irteam/ddn-opendata1/model/ECGFMs/hubert_ecg/hubert_ecg_base.safetensors"
    "/home/irteam/ddn-opendata1/model/ECGFMs/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt"
    "/home/irteam/ddn-opendata1/model/ECGFMs/cpc/last_11597276.ckpt"
)

GPU_IDS=(0 1 2 3 4 5 6)
N_GPUS=${#GPU_IDS[@]}

# ─────────────────────────────────────────────────────────────
# 태스크
# ─────────────────────────────────────────────────────────────
TASKS=(ptb ningbo cpsc2018 cpsc_extra georgia chapman code15 ptbxl_all ptbxl_super ptbxl_diag ptbxl_sub zzu_pecg)

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
# 모델 1개의 전체 태스크를 single GPU에서 순차 실행하는 함수
# ─────────────────────────────────────────────────────────────
run_model() {
    local gpu=$1
    local model_name=$2
    local encoder_cls=$3
    local encoder_ckpt=$4

    for mode in "${MODES[@]}"; do
        local epochs=$EPOCHS
        local lr_arg=""
        if [[ "$mode" == finetune_* ]]; then
            epochs=$FINETUNE_EPOCHS
            lr_arg="--lr $FINETUNE_LR"
        fi

        for task in "${TASKS[@]}"; do
            local save_dir="$RESULT_DIR/${model_name}_${task}_${mode}"

            echo ""
            echo "────────────────────────────────────────────────────────────"
            echo " [GPU $gpu] $model_name / $task / $mode  ($(date '+%H:%M:%S'))"
            echo "────────────────────────────────────────────────────────────"

            CUDA_VISIBLE_DEVICES=$gpu python run.py \
                --task "$task" --eval_mode "$mode" \
                --encoder_cls "$encoder_cls" \
                --encoder_ckpt "$encoder_ckpt" \
                --epochs $epochs $lr_arg \
                --save_dir "$save_dir" \
                2>&1

            echo ""
        done
    done

    echo "══════ [GPU $gpu] $model_name 완료 ($(date '+%H:%M:%S')) ══════"
}

# ─────────────────────────────────────────────────────────────
# 메인: 모델별 병렬 실행
# ─────────────────────────────────────────────────────────────
{
echo "======================================================================"
echo " Full Benchmark: ${#MODEL_NAMES[@]} models × ${#TASKS[@]} tasks × ${#MODES[@]} modes"
echo " GPUs: ${GPU_IDS[*]} (single GPU per model, parallel)"
echo " Modes: ${MODES[*]}"
echo " Results: $RESULT_DIR"
echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

PIDS=()

for i in "${!MODEL_NAMES[@]}"; do
    gpu_idx=$((i % N_GPUS))
    gpu=${GPU_IDS[$gpu_idx]}

    echo "Starting ${MODEL_NAMES[$i]} on GPU $gpu"

    run_model "$gpu" "${MODEL_NAMES[$i]}" "${MODEL_CLS[$i]}" "${MODEL_CKPT[$i]}" &
    PIDS+=($!)

    # GPU 수보다 모델이 많으면, GPU가 다 차면 하나 끝날 때까지 대기
    if [ ${#PIDS[@]} -ge $N_GPUS ]; then
        wait "${PIDS[0]}"
        PIDS=("${PIDS[@]:1}")
    fi
done

# 나머지 대기
for pid in "${PIDS[@]}"; do
    wait "$pid"
done

# ─────────────────────────────────────────────────────────────
# 최종 결과 테이블
# ─────────────────────────────────────────────────────────────
echo ""
echo "======================================================================"
echo " RESULTS SUMMARY ($(date '+%Y-%m-%d %H:%M:%S'))"
echo "======================================================================"

for mode in "${MODES[@]}"; do
    echo ""
    echo "── $mode ──"
    printf "%-20s" "Task"
    for model_name in "${MODEL_NAMES[@]}"; do
        printf "  %-14s" "$model_name"
    done
    echo ""
    printf "%s\n" "$(printf '─%.0s' {1..140})"

    for task in "${TASKS[@]}"; do
        printf "%-20s" "$task"
        for model_name in "${MODEL_NAMES[@]}"; do
            metrics_file="$RESULT_DIR/${model_name}_${task}_${mode}/test_metrics.txt"
            val_file="$RESULT_DIR/${model_name}_${task}_${mode}/val_metrics.txt"
            auroc=""
            if [ -f "$metrics_file" ]; then
                auroc=$(grep "auroc_macro" "$metrics_file" | grep -oP '[\d.]+' | head -1)
            elif [ -f "$val_file" ]; then
                auroc=$(grep "auroc_macro" "$val_file" | grep -oP '[\d.]+' | head -1)
            fi
            printf "  %-14s" "${auroc:-—}"
        done
        echo ""
    done
done

echo ""
echo "완료! $(date '+%Y-%m-%d %H:%M:%S')"
echo "결과: $RESULT_DIR"

} > "$LOG" 2>&1

echo "벤치마크 시작!"
echo "  결과: $RESULT_DIR"
echo "  로그: $LOG"
echo "  모니터링: tail -f $LOG"
