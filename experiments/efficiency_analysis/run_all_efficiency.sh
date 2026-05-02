#!/bin/bash
# Per-query latency for the three encoder-decoder backbones used in the paper.
# Runs on GPU 0 (the decoder-only LLM run takes GPU 1 via CUDA_VISIBLE_DEVICES=1).

set -euo pipefail

REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
# Use the read-only local cache for Flan-T5/UL2 (the default
# ~/.cache/huggingface/hub/ has only stub configs since the prior
# disk-full incident on /media/4TB).
export HF_HUB_CACHE=/media/4TB/share/models/huggingface
export TRANSFORMERS_OFFLINE=1

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate gccp-reproduce

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] starting efficiency runs (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"

for MODEL in flan-t5-large flan-t5-xl flan-ul2; do
    LOG="$LOG_DIR/efficiency_${MODEL}_$(date '+%Y%m%d_%H%M%S').log"
    echo
    echo "[$(date '+%H:%M:%S')] === $MODEL ==="
    echo "Log: $LOG"
    python experiments/efficiency_analysis/measure_latency.py \
        --dataset dl19 \
        --model "$MODEL" \
        --num_queries 10 \
        --warmup_queries 2 2>&1 | tee "$LOG"
done

echo
echo "[$(date '+%H:%M:%S')] all efficiency runs done. Started: $START"
