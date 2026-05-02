#!/bin/bash
# Generate E5 retrieval results + rerank with Flan-T5-XL on 3 BEIR datasets.
#
# Stage 1 runs in gccp-decoder (sentence-transformers + ir_datasets + faiss).
# Stage 2 runs in gccp-reproduce (transformers 4.36 + Flan-T5).

set -euo pipefail
REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

# Default GPU choice; override with CUDA_VISIBLE_DEVICES env var.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_CACHE=/media/4TB/share/models/huggingface
export TRANSFORMERS_OFFLINE=0

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] BEIR-E5 pipeline starting"

DATASETS=("scifact" "nfcorpus" "trec-covid")

# ------------------ Stage 1: E5 retrieval (gccp-decoder) ----------------
conda activate gccp-decoder

for DS in "${DATASETS[@]}"; do
    OUT="$REPO_ROOT/data/beir_e5_${DS}.json"
    if [ -s "$OUT" ]; then
        echo "[$(date '+%H:%M:%S')] === E5 retrieval skip (exists): $DS ==="
        continue
    fi
    LOG="$LOG_DIR/beir_e5_retrieval_${DS}_$(date '+%Y%m%d_%H%M%S').log"
    echo
    echo "[$(date '+%H:%M:%S')] === E5 retrieval: $DS ==="
    echo "Log: $LOG"
    python experiments/beir_e5/generate_beir_e5.py --dataset "$DS" 2>&1 | tee "$LOG"
done

conda deactivate

# ------------------ Stage 2: rerank with Flan-T5-XL (gccp-reproduce) -----
conda activate gccp-reproduce
export HF_HUB_CACHE=/media/4TB/share/models/huggingface
export TRANSFORMERS_OFFLINE=1

for DS in "${DATASETS[@]}"; do
    LOG="$LOG_DIR/beir_e5_rerank_${DS}_$(date '+%Y%m%d_%H%M%S').log"
    echo
    echo "[$(date '+%H:%M:%S')] === rerank: $DS / Flan-T5-XL ==="
    echo "Log: $LOG"
    python experiments/beir_e5/rerank_beir_e5.py --dataset "$DS" --model flan-t5-xl 2>&1 | tee "$LOG"
done

echo
echo "[$(date '+%H:%M:%S')] all done. Started $START"
