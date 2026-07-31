#!/bin/bash
# =============================================================
# all model × all task benchmark
#
# model per by GPU 1 , parallel run.
# 8 model → 7 GPU ( model GPU 0 re-use, sequential )
#
# Usage:
#   bash run_full_benchmark.sh              # linear_probe (new timestamp)
#   bash run_full_benchmark.sh all          # 3 mode (linear/attention probe + finetune_linear)
#   bash run_full_benchmark.sh all 20260413_183153   # existing timestamp in  run
#                                                     (already doneed  skip)
#
#   MODES_OVERRIDE="linear_probe finetune_attention" bash run_full_benchmark.sh
#       #  of mode  — all of default and    mode only run
#
# :
#   tail -f results/{timestamp}/benchmark.log
# =============================================================

EVAL_MODE=${1:-linear_probe}
RESUME_TS=${2:-}    #   is: existing timestamp by resume

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

# CPC(S4/pykeops)  do GLIBCXX_3.4.32   libstdc++ none.
# rationale: ship a newer libstdc++ to the JIT-compiled nvrtc_jit.so (see README)
# Ship a newer libstdc++ to the JIT-compiled nvrtc_jit.so that CPC's S4/pykeops
# builds (see README). Resolve it from the active env; a missing path in
# LD_PRELOAD makes the loader warn on every single process.
_LIBSTDCXX="$(dirname "$(dirname "$(command -v python)")")/lib/libstdc++.so.6"
if [ -f "$_LIBSTDCXX" ]; then
    export LD_PRELOAD="$_LIBSTDCXX"
else
    echo "[warn] libstdc++.so.6 not found next to $(command -v python) — CPC may fail to JIT" >&2
fi

if [ -n "$RESUME_TS" ]; then
    TIMESTAMP="$RESUME_TS"
    echo "Resume mode: existing directory use → results/$TIMESTAMP"
else
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
fi

RESULT_DIR="results/$TIMESTAMP"
LOG="$RESULT_DIR/benchmark.log"
mkdir -p "$RESULT_DIR"

# ─────────────────────────────────────────────────────────────
# model registry (configs/models.sh from )
#   Add a new model configs/models.sh only 
#   MODELS_OVERRIDE="ecg_jepa st_mem"  as  do model only run available
# ─────────────────────────────────────────────────────────────
source "$SCRIPT_DIR/configs/models.sh"

if [ -n "$MODELS_OVERRIDE" ]; then
    MODEL_NAMES=($MODELS_OVERRIDE)
else
    MODEL_NAMES=("${MODEL_NAMES_DEFAULT[@]}")
fi

# name array cls/ckpt array before (existing code compatibility)
MODEL_CLS=()
MODEL_CKPT=()
for m in "${MODEL_NAMES[@]}"; do
    if [ -z "${MODEL_CLS_MAP[$m]}" ]; then
        echo "[ERROR]  no model: $m"
        echo "  use available: ${!MODEL_CLS_MAP[*]}"
        exit 1
    fi
    MODEL_CLS+=("${MODEL_CLS_MAP[$m]}")
    MODEL_CKPT+=("${MODEL_CKPT_MAP[$m]}")
done

# GPU environment variable by override available: GPU_IDS_OVERRIDE="2 3 4 5 6" bash run_full_benchmark.sh ...
if [ -n "$GPU_IDS_OVERRIDE" ]; then
    GPU_IDS=($GPU_IDS_OVERRIDE)
else
    GPU_IDS=(0 1 2 3 4 5 6 7)
fi
N_GPUS=${#GPU_IDS[@]}

# ─────────────────────────────────────────────────────────────
# task
# ─────────────────────────────────────────────────────────────
if [ -n "$TASKS_OVERRIDE" ]; then
    TASKS=($TASKS_OVERRIDE)
else
    # The 14 tasks the original benchmarks (main_lite_ecg.py / run.sh), plus the
    # PTB-XL sub-task variants this repo adds. code15_diag (not code15) is the
    # canonical CODE-15 task: it carries the `data_length >= 4000` cohort filter.
    TASKS=(ptb ningbo cpsc2018 cpsc_extra georgia chapman chapman_rhythm code15_diag ptbxl_all ptbxl_super ptbxl_diag ptbxl_sub ptbxl_form ptbxl_rhythm sph_diag zzu_pecg echonext)
fi

# ─────────────────────────────────────────────────────────────
# config (matches the original ecg-fm-benchmarking: epochs=100, lr=1e-3, const schedule)
# ─────────────────────────────────────────────────────────────
EPOCHS=${EPOCHS:-100}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-100}
FINETUNE_LR="${FINETUNE_LR:-1e-3}"

if [ -n "$MODES_OVERRIDE" ]; then
    MODES=($MODES_OVERRIDE)
elif [ "$EVAL_MODE" = "all" ]; then
    # default 'all' is 3 mode (linear_probe, attention_probe, finetune_linear).
    # finetune_attention also :
    #   MODES_OVERRIDE="linear_probe attention_probe finetune_linear finetune_attention" bash run_full_benchmark.sh
    MODES=(linear_probe attention_probe finetune_linear)
else
    MODES=($EVAL_MODE)
fi

# ─────────────────────────────────────────────────────────────
# model 1  all task single GPU from sequential run do function
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

            # already doneed  skip (test_metrics.txt or val_metrics.txt re- )
            if [ -f "$save_dir/test_metrics.txt" ] || [ -f "$save_dir/val_metrics.txt" ]; then
                echo ""
                echo "  [SKIP] $model_name / $task / $mode (already done)"
                continue
            fi

            echo ""
            echo "────────────────────────────────────────────────────────────"
            echo " [GPU $gpu] $model_name / $task / $mode  ($(date '+%H:%M:%S'))"
            echo "────────────────────────────────────────────────────────────"

            # The original runs 16-mixed everywhere except the S4-based models
            # (run.sh adds `--precision 32` for s4 and cpc).
            local precision_arg=""
            case "$model_name" in
                cpc|s4) precision_arg="--precision 32" ;;
            esac

            CUDA_VISIBLE_DEVICES=$gpu python run.py \
                --task "$task" --eval_mode "$mode" \
                --encoder_cls "$encoder_cls" \
                --encoder_ckpt "$encoder_ckpt" \
                --epochs $epochs $lr_arg $precision_arg \
                ${EXTRA_RUN_ARGS:-} \
                --save_dir "$save_dir" \
                2>&1

            echo ""
        done
    done

    echo "══════ [GPU $gpu] $model_name done ($(date '+%H:%M:%S')) ══════"
}

# ─────────────────────────────────────────────────────────────
# main: model per parallel run
# ─────────────────────────────────────────────────────────────
{
echo "======================================================================"
echo " Full Benchmark: ${#MODEL_NAMES[@]} models × ${#TASKS[@]} tasks × ${#MODES[@]} modes"
echo " GPUs: ${GPU_IDS[*]} (single GPU per model, parallel)"
echo " Modes: ${MODES[*]}"
echo " Results: $RESULT_DIR"
echo " Started: $(date '+%Y-%m-%d %H:%M:%S')"
echo "======================================================================"

# PID → GPU mapping (associative array)
declare -A PID2GPU
# GPU use 
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
            echo "[$(date '+%H:%M:%S')] GPU $released_gpu  (PID $pid  )"
        fi
    done
}

for i in "${!MODEL_NAMES[@]}"; do
    # all GPU use  of if so, one   
    while [ ${#PID2GPU[@]} -ge $N_GPUS ]; do
        wait -n 2>/dev/null
        release_finished_gpus
    done

    # empty GPU 
    gpu=$(find_free_gpu)
    GPU_BUSY[$gpu]=1

    echo "[$(date '+%H:%M:%S')] Starting ${MODEL_NAMES[$i]} on GPU $gpu"
    run_model "$gpu" "${MODEL_NAMES[$i]}" "${MODEL_CLS[$i]}" "${MODEL_CKPT[$i]}" &
    bg_pid=$!
    PID2GPU[$bg_pid]=$gpu
done

# others 
for pid in "${!PID2GPU[@]}"; do
    wait "$pid" 2>/dev/null
done

# ─────────────────────────────────────────────────────────────
# final results table
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
echo "done! $(date '+%Y-%m-%d %H:%M:%S')"
echo "results: $RESULT_DIR"

} >> "$LOG" 2>&1   # append: identical timestamp resume at existing log  

echo "benchmark start!"
echo "  results: $RESULT_DIR"
echo "  log: $LOG"
echo "  : tail -f $LOG"
