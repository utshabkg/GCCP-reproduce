#!/bin/bash
# Qwen-2.5-72B-AWQ on 2 BEIR sets (SciFact, DBPedia-Entity) using the
# E5 first-stage we already produced. Tests whether the contrastive
# anchor mechanism transfers to large quantized decoders beyond TREC-DL.
set -eo pipefail
REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate gccp-decoder-large
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/media/20TB/shared/models/huggingface

QWEN72B=/media/20TB/shared/models/qwen/Qwen2.5-72B-Instruct-AWQ/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] Qwen-2.5-72B-AWQ on BEIR via E5 / GPU $CUDA_VISIBLE_DEVICES"

# 72B-AWQ at ~4 min/query is too slow for full BEIR test sets; cap to
# 50 queries per set for a directional transfer-claim. SciFact (300q)
# and DBPedia-Entity (400q) are sampled to first 50 queries each;
# we report this scope explicitly in the paper.
NUM_Q=50
for DS in scifact dbpedia-entity; do
    OUT="results/beir/${DS}/qwen2.5-72b-awq_e5"
    if [ -s "$OUT/metrics.json" ]; then
        echo "[$(date '+%H:%M:%S')] skip (exists): $DS"
        continue
    fi
    LOG="$LOG_DIR/decoder_qwen2.5-72b-awq_beir_${DS}_$(date '+%Y%m%d_%H%M%S').log"
    echo
    echo "[$(date '+%H:%M:%S')] === 72B-AWQ / BEIR-$DS (first $NUM_Q queries) ==="
    echo "Log: $LOG"
    python experiments/decoder_only_models/run_decoder_beir.py \
        --dataset "$DS" --model "$QWEN72B" --short_name "qwen2.5-72b-awq" \
        --num_queries "$NUM_Q" 2>&1 | tee "$LOG"
done

echo
echo "[$(date '+%H:%M:%S')] all done. Started $START"
