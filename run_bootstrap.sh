#!/bin/bash
# =============================================================
#  bootstrap batch run
#
# 1) each result directory(<model>_<task>_<mode>) for best.pt load
#    → test set inference → preds.npy/targets.npy/ids.npy save
# 2) single-model 95% CI (n=1000)
# 3) (task, mode) pairwise diff CI + tied-rank
#
# Usage:
#   bash run_bootstrap.sh                                                # default directory
#   bash run_bootstrap.sh /path/to/results/20260428_203028                # 
#   bash run_bootstrap.sh /path/to/results/20260428_203028 "0,1,2,3"      #  GPU
#   FILTER=cpc bash run_bootstrap.sh /path/to/results/...                 #  extract only
#   SKIP_EXTRACT=1 bash run_bootstrap.sh /path/to/results/...             # extract skip
#
# option environment variable:
#   N_ITERS         (default 1000)
#   WORKERS         CPU bootstrap parallel worker (default = nproc)
#   FILTER          (extract/CI stage substring filter)
#   SKIP_EXTRACT=1  (already extract  )
#   FORCE=1         (preds.npy/bootstrap.json )
# =============================================================

set -e

RESULT_DIR=${1:-/path/to/results/<timestamp>}
GPUS=${2:-0}
N_ITERS=${N_ITERS:-1000}
WORKERS=${WORKERS:-$(nproc)}
FILTER=${FILTER:-}
SKIP_EXTRACT=${SKIP_EXTRACT:-0}
FORCE=${FORCE:-0}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

# CPC(S4/pykeops)  do GLIBCXX  — preload (run_full_benchmark.sh and identical)
export LD_PRELOAD=/home/irteam/local-node-d/_conda/envs/tykim/lib/libstdc++.so.6
export ECG_DATA_ROOT=${ECG_DATA_ROOT:-/home/irteam/ddn-opendata1}
export ECG_CKPT_ROOT=${ECG_CKPT_ROOT:-/home/irteam/ddn-opendata1/model/ECGFMs}

force_arg=""; [ "$FORCE" = "1" ] && force_arg="--force"
filter_arg=""; [ -n "$FILTER" ] && filter_arg="--filter $FILTER"

echo "============================================="
echo " Bootstrap pipeline"
echo "  RESULT_DIR : $RESULT_DIR"
echo "  GPUS       : $GPUS"
echo "  N_ITERS    : $N_ITERS"
echo "  WORKERS    : $WORKERS  (CPU bootstrap parallel)"
echo "  FILTER     : ${FILTER:-(none)}"
echo "  SKIP_EXTRACT : $SKIP_EXTRACT"
echo "  FORCE      : $FORCE"
echo "============================================="

# ── 1) inference extract ──────────────────────────────────────────────
if [ "$SKIP_EXTRACT" != "1" ]; then
    echo ""
    echo "── [1/3] Extract test predictions ──"

    IFS=',' read -ra GPU_LIST <<< "$GPUS"
    N_GPUS=${#GPU_LIST[@]}

    if [ "$N_GPUS" -le 1 ]; then
        # Single-GPU:  in all directory handling
        CUDA_VISIBLE_DEVICES=$GPUS python scripts/extract_predictions.py \
            --root "$RESULT_DIR" $filter_arg $force_arg
    else
        # Multi-GPU: directory list GPU only etc. to → parallel run
        ALL_DIRS=$(ls -d "$RESULT_DIR"/*/ 2>/dev/null | sort)
        if [ -n "$FILTER" ]; then
            ALL_DIRS=$(echo "$ALL_DIRS" | grep "$FILTER" || true)
        fi
        TOTAL=$(echo "$ALL_DIRS" | wc -l)
        echo "  Total dirs: $TOTAL  →  split across $N_GPUS GPUs"

        TMPDIR=$(mktemp -d)
        trap "rm -rf $TMPDIR" EXIT
        for i in $(seq 0 $((N_GPUS - 1))); do
            echo "$ALL_DIRS" | awk -v i="$i" -v n="$N_GPUS" 'NR % n == i' > "$TMPDIR/gpu_$i.list"
        done

        pids=()
        for i in $(seq 0 $((N_GPUS - 1))); do
            gpu="${GPU_LIST[$i]}"
            (
                while IFS= read -r dir; do
                    [ -z "$dir" ] && continue
                    CUDA_VISIBLE_DEVICES=$gpu python scripts/extract_predictions.py \
                        --result_dir "${dir%/}" $force_arg \
                        2>&1 | sed "s|^|[GPU $gpu] |"
                done < "$TMPDIR/gpu_$i.list"
            ) &
            pids+=($!)
        done
        for pid in "${pids[@]}"; do wait "$pid"; done
    fi
fi

# ── 2) Single-model 95% CI (CPU parallel) ──────────────────────────
echo ""
echo "── [2/3] Single-model bootstrap CI (n=$N_ITERS, workers=$WORKERS) ──"
python scripts/bootstrap_ci.py --root "$RESULT_DIR" \
    --n_iters $N_ITERS --workers $WORKERS $filter_arg $force_arg

# ── 3) Pairwise diff + tied-rank (CPU parallel) ────────────────────
echo ""
echo "── [3/4] Pairwise bootstrap + tied-rank (workers=$WORKERS) ──"
python scripts/bootstrap_pairwise.py --root "$RESULT_DIR" \
    --n_iters $N_ITERS --workers $WORKERS

# ── 4) Paper-style summary tables ─────────────────────────────
echo ""
echo "── [4/4] Build paper-style summary tables ──"
python scripts/make_summary_table.py --root "$RESULT_DIR"

echo ""
echo "============================================="
echo " done. outputs:"
echo "   - <result>/preds.npy, targets.npy, ids.npy, preds_meta.json"
echo "   - <result>/bootstrap.json"
echo "   - $RESULT_DIR/bootstrap_summary.csv"
echo "   - $RESULT_DIR/pairwise/pairwise_diff_<task>_<mode>.csv"
echo "   - $RESULT_DIR/pairwise/tied_groups_<task>_<mode>.txt"
echo "   - $RESULT_DIR/pairwise/pairwise_summary.csv"
echo "   - $RESULT_DIR/pairwise/summary_<mode>.csv         ← raw scores (paper-style pivot)"
echo "   - $RESULT_DIR/pairwise/summary_<mode>_marked.csv  ← bold/underline "
echo "   - $RESULT_DIR/pairwise/summary_<mode>.md          ← markdown table (bold/underline )"
echo "   - $RESULT_DIR/pairwise/summary_ci_long.csv        ←  + 95% CI long-format"
echo "============================================="
