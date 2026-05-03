#!/bin/bash
# RankGPT (listwise) baseline at Flan-T5-Large + Flan-T5-XL on DL19+DL20.
# Comparable to the paper's Table 2 numbers; useful for the reviewer-flagged
# 'how does PAGC compare to a comparative baseline' question.

set -euo pipefail
REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate gccp-reproduce
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HUB_CACHE=/media/4TB/share/models/huggingface
export TRANSFORMERS_OFFLINE=1

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] RankGPT baseline runs on GPU $CUDA_VISIBLE_DEVICES"

for MODEL in flan-t5-large flan-t5-xl; do
    for DS in dl19 dl20; do
        OUT="results/trec-dl/${DS}/${MODEL}_rankgpt"
        if [ -s "$OUT/metrics.json" ]; then
            echo "[$(date '+%H:%M:%S')] skip (exists): $MODEL / $DS"
            continue
        fi
        LOG="$LOG_DIR/rankgpt_${MODEL}_${DS}_$(date '+%Y%m%d_%H%M%S').log"
        echo
        echo "[$(date '+%H:%M:%S')] === RankGPT: $MODEL / $DS ==="
        python experiments/listwise_baseline/run_rankgpt.py \
            --dataset "$DS" --model "$MODEL" 2>&1 | tee "$LOG"
    done
done

echo
echo "[$(date '+%H:%M:%S')] all done. Started $START"
