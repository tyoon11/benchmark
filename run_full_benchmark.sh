#!/bin/bash
# =============================================================
# 전체 모델 × 전체 태스크 벤치마크
#
# 모델별로 GPU 1장씩 할당, 병렬 실행.
# 8 모델 → 7 GPU (마지막 모델은 GPU 0 재사용, 순차 대기)
#
# 사용법:
#   bash run_full_benchmark.sh              # linear_probe (새 timestamp)
#   bash run_full_benchmark.sh all          # 4가지 모드 전부
#   bash run_full_benchmark.sh all 20260413_183153   # 기존 timestamp에 이어서 실행
#                                                     (이미 완료된 실험은 skip)
#
# 모니터링:
#   tail -f results/{timestamp}/benchmark.log
# =============================================================

EVAL_MODE=${1:-linear_probe}
RESUME_TS=${2:-}    # 두 번째 인자: 기존 timestamp로 resume

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

# CPC(S4/pykeops)가 요구하는 GLIBCXX_3.4.32 심볼이 시스템 libstdc++에는 없음.
# tykim conda env의 libstdc++.so.6.0.34를 preload해서 JIT 컴파일된 nvrtc_jit.so 로드를 성공시킴.
export LD_PRELOAD=/home/irteam/local-node-d/_conda/envs/tykim/lib/libstdc++.so.6

if [ -n "$RESUME_TS" ]; then
    TIMESTAMP="$RESUME_TS"
    echo "Resume mode: 기존 폴더 사용 → results/$TIMESTAMP"
else
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
fi

RESULT_DIR="results/$TIMESTAMP"
LOG="$RESULT_DIR/benchmark.log"
mkdir -p "$RESULT_DIR"

# ─────────────────────────────────────────────────────────────
# 모델 레지스트리 (configs/models.sh에서 관리)
#   새 모델 추가는 configs/models.sh만 편집
#   MODELS_OVERRIDE="ecg_jepa st_mem" 으로 원하는 모델만 실행 가능
# ─────────────────────────────────────────────────────────────
source "$SCRIPT_DIR/configs/models.sh"

if [ -n "$MODELS_OVERRIDE" ]; then
    MODEL_NAMES=($MODELS_OVERRIDE)
else
    MODEL_NAMES=("${MODEL_NAMES_DEFAULT[@]}")
fi

# name array를 cls/ckpt 배열로 전개 (기존 코드 호환)
MODEL_CLS=()
MODEL_CKPT=()
for m in "${MODEL_NAMES[@]}"; do
    if [ -z "${MODEL_CLS_MAP[$m]}" ]; then
        echo "[ERROR] 알 수 없는 모델: $m"
        echo "  사용 가능: ${!MODEL_CLS_MAP[*]}"
        exit 1
    fi
    MODEL_CLS+=("${MODEL_CLS_MAP[$m]}")
    MODEL_CKPT+=("${MODEL_CKPT_MAP[$m]}")
done

# GPU 환경변수로 override 가능: GPU_IDS_OVERRIDE="2 3 4 5 6" bash run_full_benchmark.sh ...
if [ -n "$GPU_IDS_OVERRIDE" ]; then
    GPU_IDS=($GPU_IDS_OVERRIDE)
else
    GPU_IDS=(0 1 2 3 4 5 6)
fi
N_GPUS=${#GPU_IDS[@]}

# ─────────────────────────────────────────────────────────────
# 태스크
# ─────────────────────────────────────────────────────────────
if [ -n "$TASKS_OVERRIDE" ]; then
    TASKS=($TASKS_OVERRIDE)
else
    TASKS=(ptb ningbo cpsc2018 cpsc_extra georgia chapman chapman_rhythm code15 ptbxl_all ptbxl_super ptbxl_diag ptbxl_sub ptbxl_form ptbxl_rhythm zzu_pecg)
fi

# ─────────────────────────────────────────────────────────────
# 설정 (ecg-fm-benchmarking 원본 동일: epochs=100, lr=1e-3, const schedule)
# ─────────────────────────────────────────────────────────────
EPOCHS=100
FINETUNE_EPOCHS=100
FINETUNE_LR="1e-3"

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

            # 이미 완료된 실험은 skip (test_metrics.txt가 존재하면)
            if [ -f "$save_dir/test_metrics.txt" ]; then
                echo ""
                echo "  [SKIP] $model_name / $task / $mode (이미 완료)"
                continue
            fi

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

# PID → GPU 매핑 (associative array)
declare -A PID2GPU
# GPU 사용 상태
declare -A GPU_BUSY
for g in "${GPU_IDS[@]}"; do GPU_BUSY[$g]=""; done

find_free_gpu() {
    for g in "${GPU_IDS[@]}"; do
        if [ -z "${GPU_BUSY[$g]}" ]; then
            echo "$g"
            return
        fi
    done
}

release_finished_gpus() {
    for pid in "${!PID2GPU[@]}"; do
        if ! kill -0 $pid 2>/dev/null; then
            local released_gpu="${PID2GPU[$pid]}"
            GPU_BUSY[$released_gpu]=""
            unset PID2GPU[$pid]
            echo "[$(date '+%H:%M:%S')] GPU $released_gpu 해제 (PID $pid 종료)"
        fi
    done
}

for i in "${!MODEL_NAMES[@]}"; do
    # 모든 GPU 사용 중이면 하나 끝날 때까지 대기
    while [ ${#PID2GPU[@]} -ge $N_GPUS ]; do
        wait -n 2>/dev/null
        release_finished_gpus
    done

    # 비어있는 GPU 찾기
    gpu=$(find_free_gpu)
    GPU_BUSY[$gpu]=1

    echo "[$(date '+%H:%M:%S')] Starting ${MODEL_NAMES[$i]} on GPU $gpu"
    run_model "$gpu" "${MODEL_NAMES[$i]}" "${MODEL_CLS[$i]}" "${MODEL_CKPT[$i]}" &
    bg_pid=$!
    PID2GPU[$bg_pid]=$gpu
done

# 나머지 대기
for pid in "${!PID2GPU[@]}"; do
    wait "$pid" 2>/dev/null
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
