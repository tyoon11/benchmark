#!/usr/bin/env bash
# =============================================================
# Task-level benchmark scheduler
#
# Runs missing <model, task, mode> jobs from an existing timestamp directory
# across a fixed GPU pool. This is intended for the tail of a benchmark run,
# where model-level scheduling leaves GPUs idle.
#
# Usage:
#   TIMESTAMP=20260511_172035 GPU_IDS_OVERRIDE="0 2 3 6" \
#     bash run_task_parallel_benchmark.sh
#
#   DRY_RUN=1 TIMESTAMP=20260511_172035 bash run_task_parallel_benchmark.sh
#
# Optional:
#   WRAPPER_PIDS="302916 302928" KILL_WRAPPERS_ON_DONE=1 bash ...
# =============================================================

set -u
set -o pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR" || exit 1

TIMESTAMP=${TIMESTAMP:-${1:-}}
if [ -z "$TIMESTAMP" ]; then
    echo "[ERROR] TIMESTAMP is required. Example: TIMESTAMP=20260511_172035 bash $0" >&2
    exit 2
fi

RESULT_DIR=${RESULT_DIR:-"results/$TIMESTAMP"}
ABS_RESULT_DIR="$SCRIPT_DIR/$RESULT_DIR"
LOG="$ABS_RESULT_DIR/task_parallel.log"
BENCHMARK_LOG="$ABS_RESULT_DIR/benchmark.log"
LOCK_ROOT="$ABS_RESULT_DIR/.task_locks"
DRY_RUN=${DRY_RUN:-0}
KILL_WRAPPERS_ON_DONE=${KILL_WRAPPERS_ON_DONE:-0}
WRAPPER_PIDS=${WRAPPER_PIDS:-}
HEADER_ONLY=${HEADER_ONLY:-0}

export MORYECG_REPO=${MORYECG_REPO:-/home/irteam/local-node-d/tykim/MoryECG}
export MORYECG_CACHE=${MORYECG_CACHE:-/home/irteam/local-node-d/tykim/MoryECG/cache_v4}
export ECG_DATA_ROOT=${ECG_DATA_ROOT:-/home/irteam/ddn-opendata1}
export ECG_CKPT_ROOT=${ECG_CKPT_ROOT:-/home/irteam/ddn-opendata1/model/ECGFMs}
# Resolve libstdc++ from the env we actually run in; a missing LD_PRELOAD path
# makes the loader warn on every process. Needed for CPC's S4/pykeops JIT.
CONDA_BIN=${CONDA_BIN:-/home/irteam/local-node-d/_conda/envs/hbkim/bin}
_LIBSTDCXX="$(dirname "$CONDA_BIN")/lib/libstdc++.so.6"
[ -f "$_LIBSTDCXX" ] && export LD_PRELOAD=${LD_PRELOAD:-$_LIBSTDCXX}
export PYTHONUNBUFFERED=${PYTHONUNBUFFERED:-1}

if [ -z "${PYTHON_BIN:-}" ] && [ -x "$CONDA_BIN/python" ]; then
    PYTHON_BIN="$CONDA_BIN/python"
else
    PYTHON_BIN=${PYTHON_BIN:-python}
fi
export PATH="$CONDA_BIN:$PATH"

source "$SCRIPT_DIR/configs/models.sh"

if [ -n "${MODELS_OVERRIDE:-}" ]; then
    MODEL_NAMES=($MODELS_OVERRIDE)
else
    MODEL_NAMES=(ecg_founder ecg_jepa st_mem merl ecgfm_ked hubert_ecg ecg_fm cpc moryecg_cb1024)
fi

if [ -n "${TASKS_OVERRIDE:-}" ]; then
    TASKS=($TASKS_OVERRIDE)
else
    TASKS=(ptb ningbo cpsc2018 cpsc_extra georgia chapman chapman_rhythm code15 ptbxl_all ptbxl_super ptbxl_diag ptbxl_sub ptbxl_form ptbxl_rhythm sph_diag zzu_pecg echonext mimic)
fi

if [ -n "${MODES_OVERRIDE:-}" ]; then
    MODES=($MODES_OVERRIDE)
else
    MODES=(linear_probe attention_probe finetune_linear)
fi

if [ -n "${GPU_IDS_OVERRIDE:-}" ]; then
    GPU_IDS=($GPU_IDS_OVERRIDE)
else
    GPU_IDS=(0 2 3 6)
fi

EPOCHS=${EPOCHS:-100}
FINETUNE_EPOCHS=${FINETUNE_EPOCHS:-100}
FINETUNE_LR=${FINETUNE_LR:-1e-3}
IDLE_SLEEP=${IDLE_SLEEP:-60}

if [ ! -d "$ABS_RESULT_DIR" ]; then
    echo "[ERROR] Result directory does not exist: $ABS_RESULT_DIR" >&2
    exit 2
fi

log() {
    local msg="$*"
    printf "%s %s\n" "$(date '+%Y-%m-%d %H:%M:%S')" "$msg" | tee -a "$LOG"
}

run_name_for() {
    printf "%s_%s_%s" "$1" "$2" "$3"
}

save_dir_for() {
    printf "%s/%s" "$RESULT_DIR" "$1"
}

abs_save_dir_for() {
    printf "%s/%s" "$ABS_RESULT_DIR" "$1"
}

active_for_run() {
    local run_name=$1
    local rel="$RESULT_DIR/$run_name"
    local abs="$ABS_RESULT_DIR/$run_name"

    ps -eo args= \
        | grep -F "run.py" \
        | grep -F -- "--save_dir" \
        | grep -F -e "$rel" -e "$abs" >/dev/null 2>&1
}

expected_total() {
    echo $((${#MODEL_NAMES[@]} * ${#TASKS[@]} * ${#MODES[@]}))
}

completed_total() {
    local n=0
    local model task mode run_name d
    for model in "${MODEL_NAMES[@]}"; do
        for task in "${TASKS[@]}"; do
            for mode in "${MODES[@]}"; do
                run_name=$(run_name_for "$model" "$task" "$mode")
                d=$(abs_save_dir_for "$run_name")
                [ -f "$d/test_metrics.txt" ] && n=$((n + 1))
            done
        done
    done
    echo "$n"
}

remaining_total() {
    local total done
    total=$(expected_total)
    done=$(completed_total)
    echo $((total - done))
}

active_incomplete_total() {
    local n=0
    local model task mode run_name d
    for model in "${MODEL_NAMES[@]}"; do
        for task in "${TASKS[@]}"; do
            for mode in "${MODES[@]}"; do
                run_name=$(run_name_for "$model" "$task" "$mode")
                d=$(abs_save_dir_for "$run_name")
                [ -f "$d/test_metrics.txt" ] && continue
                if active_for_run "$run_name"; then
                    n=$((n + 1))
                fi
            done
        done
    done
    echo "$n"
}

claimable_total() {
    local n=0
    local model task mode run_name d lock_dir
    for model in "${MODEL_NAMES[@]}"; do
        for task in "${TASKS[@]}"; do
            for mode in "${MODES[@]}"; do
                run_name=$(run_name_for "$model" "$task" "$mode")
                d=$(abs_save_dir_for "$run_name")
                lock_dir="$LOCK_ROOT/$run_name.lock"
                [ -f "$d/test_metrics.txt" ] && continue
                [ -d "$lock_dir" ] && continue
                if active_for_run "$run_name"; then
                    continue
                fi
                n=$((n + 1))
            done
        done
    done
    echo "$n"
}

print_dry_run() {
    local total done remaining active claimable
    total=$(expected_total)
    done=$(completed_total)
    remaining=$((total - done))

    # DRY_RUN must not create lock dirs, so compute claimable without lock checks.
    active=$(active_incomplete_total)
    claimable=0

    echo "Task-level benchmark dry run"
    echo "  timestamp    : $TIMESTAMP"
    echo "  result_dir   : $ABS_RESULT_DIR"
    echo "  gpus         : ${GPU_IDS[*]}"
    echo "  models       : ${MODEL_NAMES[*]}"
    echo "  tasks        : ${#TASKS[@]}"
    echo "  modes        : ${MODES[*]}"
    echo "  expected     : $total"
    echo "  completed    : $done"
    echo "  remaining    : $remaining"
    echo "  active skip  : $active"
    echo ""
    echo "Pending jobs:"

    local model task mode run_name d state
    for model in "${MODEL_NAMES[@]}"; do
        for task in "${TASKS[@]}"; do
            for mode in "${MODES[@]}"; do
                run_name=$(run_name_for "$model" "$task" "$mode")
                d=$(abs_save_dir_for "$run_name")
                [ -f "$d/test_metrics.txt" ] && continue
                if active_for_run "$run_name"; then
                    state="ACTIVE_SKIP"
                elif [ -d "$d" ]; then
                    state="PARTIAL_RETRY"
                    claimable=$((claimable + 1))
                else
                    state="MISSING_RUN"
                    claimable=$((claimable + 1))
                fi
                printf "  %-13s %s %s %s\n" "$state" "$model" "$task" "$mode"
            done
        done
    done
    echo ""
    echo "  claimable now: $claimable"
}

run_one() {
    local gpu=$1
    local model=$2
    local task=$3
    local mode=$4
    local run_name save_dir abs_save_dir lock_dir encoder_cls encoder_ckpt epochs lr_arg run_log rc

    run_name=$(run_name_for "$model" "$task" "$mode")
    save_dir=$(save_dir_for "$run_name")
    abs_save_dir=$(abs_save_dir_for "$run_name")
    lock_dir="$LOCK_ROOT/$run_name.lock"

    if ! mkdir "$lock_dir" 2>/dev/null; then
        return 1
    fi

    {
        echo "gpu=$gpu"
        echo "model=$model"
        echo "task=$task"
        echo "mode=$mode"
        echo "claimed_at=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "pid=$$"
    } > "$lock_dir/meta"

    if [ -f "$abs_save_dir/test_metrics.txt" ]; then
        echo "already_complete_at=$(date '+%Y-%m-%d %H:%M:%S')" >> "$lock_dir/meta"
        return 0
    fi

    if active_for_run "$run_name"; then
        echo "active_skip_at=$(date '+%Y-%m-%d %H:%M:%S')" >> "$lock_dir/meta"
        rmdir "$lock_dir" 2>/dev/null || true
        return 1
    fi

    encoder_cls=${MODEL_CLS_MAP[$model]:-}
    encoder_ckpt=${MODEL_CKPT_MAP[$model]:-}
    if [ -z "$encoder_cls" ] || [ -z "$encoder_ckpt" ]; then
        log "[GPU $gpu] ERROR missing model registry entry for $model"
        echo "registry_error_at=$(date '+%Y-%m-%d %H:%M:%S')" >> "$lock_dir/meta"
        return 0
    fi

    if [ -d "$abs_save_dir" ]; then
        log "[GPU $gpu] retry partial: $run_name"
        rm -rf "$abs_save_dir"
    fi
    mkdir -p "$abs_save_dir"

    epochs=$EPOCHS
    lr_arg=()
    if [[ "$mode" == finetune_* ]]; then
        epochs=$FINETUNE_EPOCHS
        lr_arg=(--lr "$FINETUNE_LR")
    fi

    run_log="$abs_save_dir/run.log"
    log "[GPU $gpu] START $model / $task / $mode"
    {
        echo "======================================================================"
        echo "Task parallel run"
        echo "  started : $(date '+%Y-%m-%d %H:%M:%S')"
        echo "  gpu     : $gpu"
        echo "  model   : $model"
        echo "  task    : $task"
        echo "  mode    : $mode"
        echo "  save_dir: $save_dir"
        echo "======================================================================"
    } > "$run_log"

    local precision_arg=()
    case "$model" in
        cpc|s4) precision_arg=(--precision 32) ;;
    esac

    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" run.py \
        --task "$task" --eval_mode "$mode" \
        --encoder_cls "$encoder_cls" \
        --encoder_ckpt "$encoder_ckpt" \
        --epochs "$epochs" "${lr_arg[@]}" "${precision_arg[@]}" \
        --save_dir "$save_dir" \
        2>&1 | tee -a "$run_log" >> "$BENCHMARK_LOG"
    rc=$?

    echo "finished_at=$(date '+%Y-%m-%d %H:%M:%S')" >> "$lock_dir/meta"
    echo "exit_code=$rc" >> "$lock_dir/meta"

    if [ -f "$abs_save_dir/test_metrics.txt" ]; then
        log "[GPU $gpu] DONE  $run_name"
    else
        log "[GPU $gpu] FAIL  $run_name (exit=$rc, no test_metrics.txt)"
    fi

    return 0
}

worker_loop() {
    local gpu=$1
    local total done remaining active claimable did_work model task mode run_name d

    log "[GPU $gpu] worker online"
    while true; do
        total=$(expected_total)
        done=$(completed_total)
        remaining=$((total - done))
        if [ "$remaining" -le 0 ]; then
            log "[GPU $gpu] all jobs complete ($done / $total)"
            return 0
        fi

        active=$(active_incomplete_total)
        claimable=$(claimable_total)
        if [ "$claimable" -le 0 ]; then
            if [ "$active" -gt 0 ]; then
                log "[GPU $gpu] waiting: $remaining remaining, $active active elsewhere"
                sleep "$IDLE_SLEEP"
                continue
            fi
            log "[GPU $gpu] no claimable jobs remain ($done / $total complete, $remaining incomplete)"
            return 0
        fi

        did_work=0
        for model in "${MODEL_NAMES[@]}"; do
            for task in "${TASKS[@]}"; do
                for mode in "${MODES[@]}"; do
                    run_name=$(run_name_for "$model" "$task" "$mode")
                    d=$(abs_save_dir_for "$run_name")
                    [ -f "$d/test_metrics.txt" ] && continue
                    active_for_run "$run_name" && continue
                    run_one "$gpu" "$model" "$task" "$mode"
                    if [ "$?" -eq 0 ]; then
                        did_work=1
                        break 3
                    fi
                done
            done
        done

        if [ "$did_work" -eq 0 ]; then
            sleep "$IDLE_SLEEP"
        fi
    done
}

finish_wrappers_if_requested() {
    if [ "$KILL_WRAPPERS_ON_DONE" != "1" ] || [ -z "$WRAPPER_PIDS" ]; then
        return 0
    fi
    log "KILL_WRAPPERS_ON_DONE=1: terminating wrapper PIDs: $WRAPPER_PIDS"
    for pid in $WRAPPER_PIDS; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -TERM "$pid" 2>/dev/null || true
        fi
    done
}

if [ "$DRY_RUN" = "1" ]; then
    print_dry_run
    exit 0
fi

mkdir -p "$LOCK_ROOT"
touch "$LOG"

log "======================================================================"
log "Task-level benchmark scheduler"
log "  timestamp : $TIMESTAMP"
log "  result_dir: $ABS_RESULT_DIR"
log "  gpus      : ${GPU_IDS[*]}"
log "  models    : ${MODEL_NAMES[*]}"
log "  tasks     : ${TASKS[*]}"
log "  modes     : ${MODES[*]}"
log "  python    : $PYTHON_BIN"
initial_expected=$(expected_total)
initial_completed=$(completed_total)
log "  expected  : $initial_expected"
log "  completed : $initial_completed"
log "======================================================================"

if [ "$HEADER_ONLY" = "1" ]; then
    log "HEADER_ONLY=1: exiting before worker launch"
    exit 0
fi

pids=()
for gpu in "${GPU_IDS[@]}"; do
    worker_loop "$gpu" &
    pids+=($!)
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

final_done=$(completed_total)
final_total=$(expected_total)
log "Task-level scheduler finished: $final_done / $final_total complete"

if [ "$final_done" -ge "$final_total" ]; then
    finish_wrappers_if_requested
fi
