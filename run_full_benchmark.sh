#!/bin/bash
# =============================================================
# 전체 모델 × 전체 태스크 벤치마크
#
# 사용법:
#   bash run_full_benchmark.sh              # linear_probe만
#   bash run_full_benchmark.sh all          # 4가지 모드 전부
#
# 로그: results/benchmark.log 하나에 통합
#   tail -f results/benchmark.log
# =============================================================

EVAL_MODE=${1:-linear_probe}
GPUS="0,1,2,3,4,5,6"
N_GPUS=7

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"
mkdir -p results

# LOG는 위에서 설정됨

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

TOTAL=$((${#MODEL_NAMES[@]} * ${#TASKS[@]} * ${#MODES[@]}))
DONE=0
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULT_DIR="results/$TIMESTAMP"
LOG="$RESULT_DIR/benchmark.log"
mkdir -p "$RESULT_DIR"

# ─────────────────────────────────────────────────────────────
# 실행 (모든 출력을 $LOG 하나에 통합)
# ─────────────────────────────────────────────────────────────
{
echo "======================================================================"
echo " Full Benchmark: ${#MODEL_NAMES[@]} models × ${#TASKS[@]} tasks × ${#MODES[@]} modes = $TOTAL runs"
echo " GPUs: $GPUS ($N_GPUS)  |  Modes: ${MODES[*]}"
echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

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

        echo ""
        echo "══════════════════════════════════════════════════════════════════"
        echo " $model_name / $mode"
        echo "══════════════════════════════════════════════════════════════════"

        for task in "${TASKS[@]}"; do
            DONE=$((DONE + 1))
            echo ""
            echo "────────────────────────────────────────────────────────────"
            echo " [$DONE/$TOTAL] $model_name / $task / $mode  ($(date '+%H:%M:%S'))"
            echo "────────────────────────────────────────────────────────────"

            PORT=$((29500 + RANDOM % 1000))
            SAVE_DIR="$RESULT_DIR/${model_name}_${task}_${mode}"
            CUDA_VISIBLE_DEVICES=$GPUS torchrun --nproc_per_node=$N_GPUS \
                --master_port=$PORT run.py \
                --task "$task" --eval_mode "$mode" \
                --encoder_cls "$encoder_cls" \
                --encoder_ckpt "$encoder_ckpt" \
                --epochs $epochs $lr_arg \
                --save_dir "$SAVE_DIR" \
                2>&1 || {
                echo "  [FAIL] $model_name / $task / $mode"
            }
            sleep 3
        done
    done
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
            # 로그에서 해당 구간의 Best/Test AUROC 추출
            auroc=$(grep -A9999 "\[$model_name / $task / $mode\]" "$LOG" 2>/dev/null | grep -m1 "Test AUROC" | grep -oP '0\.\d+' | head -1)
            [ -z "$auroc" ] && auroc=$(grep -A9999 "\[$model_name / $task / $mode\]" "$LOG" 2>/dev/null | grep -m1 "Best val AUROC" | grep -oP '0\.\d+' | head -1)
            printf "  %-14s" "${auroc:-—}"
        done
        echo ""
    done
done

echo ""
echo "완료! $(date '+%Y-%m-%d %H:%M:%S')"

} > "$LOG" 2>&1

echo "벤치마크 시작. 로그: $RESULT_DIR/benchmark.log"
echo "모니터링: tail -f $RESULT_DIR/benchmark.log"
