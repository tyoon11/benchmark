#!/bin/bash
# =============================================================
# 경험적 부트스트랩 일괄 실행
#
# 1) 각 result 폴더(<model>_<task>_<mode>)에 대해 best.pt 로드
#    → test set 추론 → preds.npy/targets.npy/ids.npy 저장
# 2) 단일-모델 95% CI (n=1000)
# 3) (task, mode)별 pairwise diff CI + tied-rank
#
# 사용법:
#   bash run_bootstrap.sh                                                # default 폴더
#   bash run_bootstrap.sh /path/to/results/20260428_203028                # 명시
#   bash run_bootstrap.sh /path/to/results/20260428_203028 "0,1,2,3"      # 멀티 GPU
#   FILTER=cpc bash run_bootstrap.sh /path/to/results/...                 # 부분 추출만
#   SKIP_EXTRACT=1 bash run_bootstrap.sh /path/to/results/...             # 추출 skip
#
# 옵션 환경변수:
#   N_ITERS         (default 1000)
#   WORKERS         CPU 부트스트랩 병렬 worker (default = nproc)
#   FILTER          (extract/CI 단계 substring 필터)
#   SKIP_EXTRACT=1  (이미 추출 끝났을 때)
#   FORCE=1         (preds.npy/bootstrap.json 덮어쓰기)
# =============================================================

set -e

RESULT_DIR=${1:-/home/irteam/ddn-opendata1/tykim/benchmark/results/20260428_203028}
GPUS=${2:-0}
N_ITERS=${N_ITERS:-1000}
WORKERS=${WORKERS:-$(nproc)}
FILTER=${FILTER:-}
SKIP_EXTRACT=${SKIP_EXTRACT:-0}
FORCE=${FORCE:-0}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
cd "$SCRIPT_DIR"

# CPC(S4/pykeops)가 요구하는 GLIBCXX 심볼 — preload (run_full_benchmark.sh와 동일)
export LD_PRELOAD=/home/irteam/local-node-a/_conda/envs/tykim/lib/libstdc++.so.6
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

# ── 1) 추론 추출 ──────────────────────────────────────────────
if [ "$SKIP_EXTRACT" != "1" ]; then
    echo ""
    echo "── [1/3] Extract test predictions ──"

    IFS=',' read -ra GPU_LIST <<< "$GPUS"
    N_GPUS=${#GPU_LIST[@]}

    if [ "$N_GPUS" -le 1 ]; then
        # Single-GPU: 한 번에 모든 폴더 처리
        CUDA_VISIBLE_DEVICES=$GPUS python scripts/extract_predictions.py \
            --root "$RESULT_DIR" $filter_arg $force_arg
    else
        # Multi-GPU: 폴더 리스트를 GPU 수만큼 균등 분할 → 병렬 실행
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

# ── 2) Single-model 95% CI (CPU 병렬) ──────────────────────────
echo ""
echo "── [2/3] Single-model bootstrap CI (n=$N_ITERS, workers=$WORKERS) ──"
python scripts/bootstrap_ci.py --root "$RESULT_DIR" \
    --n_iters $N_ITERS --workers $WORKERS $filter_arg $force_arg

# ── 3) Pairwise diff + tied-rank (CPU 병렬) ────────────────────
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
echo " 완료. 산출물:"
echo "   - <result>/preds.npy, targets.npy, ids.npy, preds_meta.json"
echo "   - <result>/bootstrap.json"
echo "   - $RESULT_DIR/bootstrap_summary.csv"
echo "   - $RESULT_DIR/pairwise/pairwise_diff_<task>_<mode>.csv"
echo "   - $RESULT_DIR/pairwise/tied_groups_<task>_<mode>.txt"
echo "   - $RESULT_DIR/pairwise/pairwise_summary.csv"
echo "   - $RESULT_DIR/pairwise/summary_<mode>.csv         ← raw scores (paper-style pivot)"
echo "   - $RESULT_DIR/pairwise/summary_<mode>_marked.csv  ← bold/underline 마킹"
echo "   - $RESULT_DIR/pairwise/summary_<mode>.md          ← markdown 표 (bold/underline 렌더링)"
echo "   - $RESULT_DIR/pairwise/summary_ci_long.csv        ← 점추정 + 95% CI long-format"
echo "============================================="
