#!/usr/bin/env bash
# =============================================================
# MoRyECG A5 — empirical bootstrap (this model only)
#
# 1) extract test predictions for every moryecg_a5_<task>_<mode> dir
#    (GPU inference, split across the GPU pool) → preds.npy/targets.npy/ids.npy
# 2) single-model 95% bootstrap CI (CPU) → bootstrap.json + bootstrap_summary.csv
#
# Pairwise / cross-model summary stages are intentionally skipped: the other
# models were already bootstrapped, and this result dir holds only moryecg_a5.
#
# Encoder checkpoint: run1 archive (epoch 5) — the one the benchmark heads were
# trained with. The main best.pt was overwritten by a later run2 pretrain.
#
# Usage:
#   bash run_bootstrap_a5.sh                              # defaults below
#   bash run_bootstrap_a5.sh results/20260604_142109 "0,1,2,3,4,5,6"
# Env overrides: N_ITERS, WORKERS, FORCE=1, SKIP_EXTRACT=1
# =============================================================
set -u
set -o pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

RESULT_DIR=${1:-results/20260604_142109}
GPUS=${2:-0,1,2,3,4,5,6}
FILTER=moryecg_a5
N_ITERS=${N_ITERS:-1000}
WORKERS=${WORKERS:-32}
SKIP_EXTRACT=${SKIP_EXTRACT:-0}
FORCE=${FORCE:-0}

# hbkim env (matches the running pretrain; pure-torch S4 needs no pykeops/preload)
source "$HOME/.conda/etc/profile.d/conda.sh" 2>/dev/null || true
conda activate hbkim 2>/dev/null || true

export MORYECG_REPO=${MORYECG_REPO:-/home/irteam/local-node-d/tykim/MoryECG}
export MORYECG_CACHE=${MORYECG_CACHE:-/home/irteam/local-node-d/tykim/MoryECG/cache_v4}
export ECG_DATA_ROOT=${ECG_DATA_ROOT:-/home/irteam/ddn-opendata1}
export ECG_CKPT_ROOT=${ECG_CKPT_ROOT:-/home/irteam/ddn-opendata1/model/ECGFMs}
# run1 archive checkpoint (feature-consistent with the trained benchmark heads)
export MORYECG_A5_CKPT=${MORYECG_A5_CKPT:-$MORYECG_REPO/checkpoints/pretrain_axial_s4_a5_heedb_full_cb1024/_run1_archive_20260608/best.pt}
export PYTHONUNBUFFERED=1

force_arg=""; [ "$FORCE" = "1" ] && force_arg="--force"

echo "============================================="
echo " MoRyECG A5 bootstrap"
echo "  RESULT_DIR : $RESULT_DIR"
echo "  GPUS       : $GPUS"
echo "  N_ITERS    : $N_ITERS   WORKERS: $WORKERS"
echo "  A5 ckpt    : $MORYECG_A5_CKPT"
echo "  cache      : $MORYECG_CACHE"
echo "============================================="

# ── 1) Extract predictions (multi-GPU round-robin over dirs) ──────────
if [ "$SKIP_EXTRACT" != "1" ]; then
    echo "── [1/2] Extract test predictions ──"
    IFS=',' read -ra GPU_LIST <<< "$GPUS"
    N_GPUS=${#GPU_LIST[@]}
    ALL_DIRS=$(ls -d "$RESULT_DIR"/${FILTER}_*/ 2>/dev/null | sort)
    TOTAL=$(echo "$ALL_DIRS" | grep -c . || true)
    echo "  $TOTAL dirs across $N_GPUS GPUs"

    TMPDIR=$(mktemp -d); trap 'rm -rf "$TMPDIR"' EXIT
    for i in $(seq 0 $((N_GPUS - 1))); do
        echo "$ALL_DIRS" | awk -v i="$i" -v n="$N_GPUS" 'NF && (NR-1) % n == i' > "$TMPDIR/gpu_$i.list"
    done
    pids=()
    for i in $(seq 0 $((N_GPUS - 1))); do
        gpu="${GPU_LIST[$i]}"
        (
            while IFS= read -r dir; do
                [ -z "$dir" ] && continue
                CUDA_VISIBLE_DEVICES="$gpu" python scripts/extract_predictions.py \
                    --result_dir "${dir%/}" $force_arg 2>&1 | sed "s|^|[GPU $gpu] |"
            done < "$TMPDIR/gpu_$i.list"
        ) &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
fi

# ── 2) Single-model 95% bootstrap CI (CPU) ────────────────────────────
echo ""
echo "── [2/2] Bootstrap CI (n=$N_ITERS, workers=$WORKERS) ──"
python scripts/bootstrap_ci.py --root "$RESULT_DIR" \
    --filter "$FILTER" --n_iters "$N_ITERS" --workers "$WORKERS" $force_arg

echo ""
echo "============================================="
echo " done. outputs:"
echo "   - $RESULT_DIR/${FILTER}_*/preds.npy, targets.npy, ids.npy"
echo "   - $RESULT_DIR/${FILTER}_*/bootstrap.json"
echo "   - $RESULT_DIR/bootstrap_summary.csv"
echo "============================================="
