#!/bin/bash
# Run Qwen-2.5-72B-Instruct-AWQ on DL19 (and DL20 if time permits) using
# the gccp-decoder-large env (torch 2.5.1 + transformers 4.51.3 + autoawq 0.2.9).

set -euo pipefail
REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate gccp-decoder-large
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/media/20TB/shared/models/huggingface

QWEN72B=/media/20TB/shared/models/qwen/Qwen2.5-72B-Instruct-AWQ/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] Qwen-2.5-72B-AWQ on GPU $CUDA_VISIBLE_DEVICES"

for DS in dl19 dl20; do
    OUT="results/trec-dl/${DS}/qwen2.5-72b-awq_bm25"
    if [ -s "$OUT/metrics.json" ]; then
        echo "[$(date '+%H:%M:%S')] skip (exists): $DS"
        continue
    fi
    LOG="$LOG_DIR/decoder_qwen2.5-72b-awq_${DS}_$(date '+%Y%m%d_%H%M%S').log"
    echo
    echo "[$(date '+%H:%M:%S')] === 72B-AWQ / $DS ==="
    echo "Log: $LOG"
    python experiments/decoder_only_models/run_decoder.py \
        --dataset "$DS" --model "$QWEN72B" --short_name "qwen2.5-72b-awq" 2>&1 | tee "$LOG"
done

echo
echo "[$(date '+%H:%M:%S')] all done. Started $START"
