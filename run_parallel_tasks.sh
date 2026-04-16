#!/bin/bash
# =============================================================
# 한 모델의 남은 태스크들을 여러 GPU로 병렬 실행
#
# 사용법:
#   bash run_parallel_tasks.sh <model> <eval_mode> <timestamp> <gpus> <tasks...>
#
# 예시:
#   bash run_parallel_tasks.sh hubert_ecg finetune_attention 20260414_140124 \
#        "0 1 3 4 5" \
#        ptbxl_all ptbxl_super ptbxl_diag ptbxl_sub zzu_pecg
# =============================================================

set -e

MODEL=$1
EVAL_MODE=$2
TIMESTAMP=$3
GPUS_STR=$4
shift 4
TASKS=("$@")

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

GPUS=($GPUS_STR)
RESULT_DIR="results/$TIMESTAMP"

# 모델별 클래스 + 체크포인트 매핑
declare -A MODEL_CLS
declare -A MODEL_CKPT

MODEL_CLS[ecg_founder]="src.encoders.ecg_founder.ECGFounderEncoder"
MODEL_CLS[ecg_jepa]="src.encoders.ecg_jepa.ECGJEPAEncoder"
MODEL_CLS[st_mem]="src.encoders.st_mem.StMemEncoder"
MODEL_CLS[merl]="src.encoders.merl.MerlResNetEncoder"
MODEL_CLS[ecgfm_ked]="src.encoders.ecgfm_ked.EcgFmKEDEncoder"
MODEL_CLS[hubert_ecg]="src.encoders.hubert_ecg.HuBERTECGEncoder"
MODEL_CLS[ecg_fm]="src.encoders.ecg_fm.ECGFMEncoder"
MODEL_CLS[cpc]="src.encoders.cpc.CPCEncoder"

MODEL_CKPT[ecg_founder]="/home/irteam/ddn-opendata1/model/ECGFMs/ecg_founder/12_lead_ECGFounder.pth"
MODEL_CKPT[ecg_jepa]="/home/irteam/ddn-opendata1/model/ECGFMs/ecg_jepa/multiblock_epoch100.pth"
MODEL_CKPT[st_mem]="/home/irteam/ddn-opendata1/model/ECGFMs/st_mem/st_mem_vit_base_full.pth"
MODEL_CKPT[merl]="/home/irteam/ddn-opendata1/model/ECGFMs/merl/res18_best_encoder.pth"
MODEL_CKPT[ecgfm_ked]="/home/irteam/ddn-opendata1/model/ECGFMs/ecgfm_ked/best_valid_all_increase_with_augment_epoch_3.pt"
MODEL_CKPT[hubert_ecg]="/home/irteam/ddn-opendata1/model/ECGFMs/hubert_ecg/hubert_ecg_base.safetensors"
MODEL_CKPT[ecg_fm]="/home/irteam/ddn-opendata1/model/ECGFMs/ecg_fm/mimic_iv_ecg_physionet_pretrained.pt"
MODEL_CKPT[cpc]="/home/irteam/ddn-opendata1/model/ECGFMs/cpc/last_11597276.ckpt"

# epochs / lr
if [[ "$EVAL_MODE" == finetune_* ]]; then
    EPOCHS=30
    LR_ARG="--lr 5e-4"
else
    EPOCHS=50
    LR_ARG=""
fi

ENCODER_CLS="${MODEL_CLS[$MODEL]}"
ENCODER_CKPT="${MODEL_CKPT[$MODEL]}"

if [ -z "$ENCODER_CLS" ]; then
    echo "[ERROR] 알 수 없는 모델: $MODEL"
    exit 1
fi

echo "============================================="
echo " Model:      $MODEL"
echo " Eval mode:  $EVAL_MODE"
echo " Timestamp:  $TIMESTAMP"
echo " GPUs:       ${GPUS[*]}"
echo " Tasks:      ${TASKS[*]}"
echo "============================================="

# GPU 수와 태스크 수 비교
N_TASKS=${#TASKS[@]}
N_GPUS=${#GPUS[@]}

if [ $N_TASKS -gt $N_GPUS ]; then
    echo "[WARNING] 태스크 수($N_TASKS) > GPU 수($N_GPUS) — 일부 GPU에 여러 태스크 순차 실행"
fi

# 각 GPU에 태스크 분배 (라운드 로빈)
PIDS=()
for i in "${!TASKS[@]}"; do
    task="${TASKS[$i]}"
    gpu_idx=$((i % N_GPUS))
    gpu="${GPUS[$gpu_idx]}"
    save_dir="$RESULT_DIR/${MODEL}_${task}_${EVAL_MODE}"
    log="$RESULT_DIR/parallel_${MODEL}_${task}_${EVAL_MODE}.log"

    # 이미 완료된 태스크 skip
    if [ -f "$save_dir/test_metrics.txt" ]; then
        echo "  [SKIP] $task (이미 완료)"
        continue
    fi

    echo "  [GPU $gpu] $task → $log"

    CUDA_VISIBLE_DEVICES=$gpu nohup python run.py \
        --task "$task" --eval_mode "$EVAL_MODE" \
        --encoder_cls "$ENCODER_CLS" \
        --encoder_ckpt "$ENCODER_CKPT" \
        --epochs $EPOCHS $LR_ARG \
        --save_dir "$save_dir" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "${#PIDS[@]}개 태스크 백그라운드 시작 (PIDS: ${PIDS[*]})"
echo ""
echo "모니터링:"
echo "  tail -f $RESULT_DIR/parallel_${MODEL}_*.log"
echo ""
echo "GPU 사용:"
sleep 3
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits
