#!/usr/bin/env bash
# =============================================================
# A5 run2 (epoch-70 best) — benchmark → bootstrap → combined Notion table
# End-to-end, all 7 GPUs. Run after stopping the pretrain.
#
#   TS=YYYYMMDD_HHMMSS bash orchestrate_a5_run2.sh
# =============================================================
set -u
cd "$(dirname "$(realpath "$0")")"

REPO=/home/irteam/local-node-d/tykim/MoryECG
PYHB=/home/irteam/local-node-d/_conda/envs/hbkim/bin/python
A5CKPT="$REPO/checkpoints/pretrain_axial_s4_a5_heedb_full_cb1024/best.pt"   # run2 epoch-70 best
CACHE="$REPO/cache_v4"
OLD=results/20260511_172035
TS=${TS:?set TS}
NEW="results/$TS"
GPUS="0 1 2 3 4 5 6"

echo "[$(date '+%F %T')] ===== A5 run2 orchestration : TS=$TS ====="
echo "  encoder ckpt: $A5CKPT"

# ── Phase A: benchmark moryecg_a5 (run2) — 18 standard tasks × 3 modes = 54 jobs ──
mkdir -p "$NEW"
echo "[$(date '+%F %T')] PHASE A: benchmark (54 jobs, all GPUs)"
MODELS_OVERRIDE=moryecg_a5 TIMESTAMP="$TS" GPU_IDS_OVERRIDE="$GPUS" \
    bash run_task_parallel_benchmark.sh
echo "[$(date '+%F %T')] PHASE A done — $(find "$NEW" -name test_metrics.txt | wc -l)/54 test_metrics"

# ── Phase B: extract test predictions (run2 ckpt) for the new dirs, all GPUs ──
echo "[$(date '+%F %T')] PHASE B: extract predictions"
export MORYECG_REPO=$REPO MORYECG_CACHE=$CACHE
export ECG_DATA_ROOT=/home/irteam/ddn-opendata1 ECG_CKPT_ROOT=/home/irteam/ddn-opendata1/model/ECGFMs
export MORYECG_A5_CKPT="$A5CKPT" PYTHONUNBUFFERED=1
ALL_DIRS=$(ls -d "$NEW"/moryecg_a5_*/ 2>/dev/null | sort)
GPU_ARR=($GPUS); NG=${#GPU_ARR[@]}
i=0; pids=()
for dir in $ALL_DIRS; do
    gpu=${GPU_ARR[$((i % NG))]}
    CUDA_VISIBLE_DEVICES="$gpu" $PYHB scripts/extract_predictions.py --result_dir "${dir%/}" \
        >"${dir%/}/extract.log" 2>&1 &
    pids+=($!); i=$((i+1))
    # cap concurrency at NG
    if [ "${#pids[@]}" -ge "$NG" ]; then wait "${pids[0]}"; pids=("${pids[@]:1}"); fi
done
wait
echo "[$(date '+%F %T')] PHASE B done — $(ls "$NEW"/moryecg_a5_*/preds.npy 2>/dev/null | wc -l) preds"

# ── Phase C: combine with baselines → bootstrap → pairwise → table ──
echo "[$(date '+%F %T')] PHASE C: combine + bootstrap + table"
# repoint a5 symlinks in OLD from run1 → run2 (18 standard tasks × 3 modes)
find "$OLD" -maxdepth 1 -name "moryecg_a5_*" -type l -delete 2>/dev/null
STD="ptb ningbo cpsc2018 cpsc_extra georgia chapman chapman_rhythm code15 ptbxl_all ptbxl_super ptbxl_diag ptbxl_sub ptbxl_form ptbxl_rhythm sph_diag zzu_pecg echonext mimic"
for t in $STD; do for m in linear_probe attention_probe finetune_linear; do
    src="$(pwd)/$NEW/moryecg_a5_${t}_${m}"
    [ -d "$src" ] && ln -sfn "$src" "$OLD/moryecg_a5_${t}_${m}"
done; done

$PYHB scripts/bootstrap_ci.py       --root "$OLD" --n_iters 1000 --workers 48
$PYHB scripts/bootstrap_pairwise.py --root "$OLD" --n_iters 1000 --workers 48
$PYHB scripts/make_summary_table.py --root "$OLD"
$PYHB scripts/make_notion_bootstrap_md.py --root "$OLD"
echo "[$(date '+%F %T')] ALL DONE → $OLD/bootstrap_results.md"
