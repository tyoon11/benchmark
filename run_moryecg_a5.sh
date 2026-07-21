#!/usr/bin/env bash
# =============================================================
# MoRyECG A5 (axial_s4) — one-model, all-task benchmark launcher
#
# Spreads the single moryecg_a5 model across the whole GPU pool at the
# <task, mode> granularity (via run_task_parallel_benchmark.sh), so all
# GPUs stay busy for one model — unlike run_full_benchmark.sh which assigns
# one GPU per model and would leave 7 GPUs idle for a single model.
#
# Run it the moment training finishes (checkpoint best.pt must exist):
#   bash run_moryecg_a5.sh                 # 3 modes, all tasks, GPUs 0-7
#
# Common overrides (env vars):
#   MODES_OVERRIDE="linear_probe"          # subset of modes
#   TASKS_OVERRIDE="ptbxl_super chapman"   # subset of tasks
#   GPU_IDS_OVERRIDE="0 1 2 3"             # subset of GPUs
#   TIMESTAMP=20260602_120000              # resume into an existing run dir
#   DRY_RUN=1                              # show the job plan, run nothing
#
# Watch:  tail -f results/<timestamp>/benchmark.log
# =============================================================
set -u
set -o pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR" || exit 1

CKPT="$SCRIPT_DIR/../checkpoints/pretrain_axial_s4_a5_heedb_full_cb1024/best.pt"
if [ ! -f "$CKPT" ] && [ "${DRY_RUN:-0}" != "1" ]; then
    echo "[ERROR] A5 checkpoint not found: $CKPT" >&2
    echo "        Training not finished yet? (best.pt is written on each new best epoch)" >&2
    exit 2
fi

# Fix this launcher to the A5 model only; everything else flows through the
# task-parallel scheduler's own defaults / overrides.
export MODELS_OVERRIDE="moryecg_a5"

# Fresh timestamp unless the caller is resuming an existing run dir.
if [ -z "${TIMESTAMP:-}" ]; then
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
fi
export TIMESTAMP
RESULT_DIR="results/$TIMESTAMP"
mkdir -p "$SCRIPT_DIR/$RESULT_DIR"

# Default to the full 8-GPU pool (the scheduler's own default is "0 2 3 6").
export GPU_IDS_OVERRIDE="${GPU_IDS_OVERRIDE:-0 1 2 3 4 5 6 7}"

echo "MoRyECG A5 benchmark launching"
echo "  model     : moryecg_a5  (axial_s4, d_model=384)"
echo "  checkpoint: $CKPT"
echo "  result_dir: $RESULT_DIR"
echo "  gpus      : $GPU_IDS_OVERRIDE"
echo "  modes     : ${MODES_OVERRIDE:-linear_probe attention_probe finetune_linear}"
echo "  watch     : tail -f $RESULT_DIR/benchmark.log"
echo ""

exec bash "$SCRIPT_DIR/run_task_parallel_benchmark.sh"
