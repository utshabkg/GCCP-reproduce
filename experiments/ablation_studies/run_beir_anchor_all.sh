#!/bin/bash
# Run anchor-construction ablation across all 8 BEIR datasets with Flan-T5-Large.
# Pinned to GPU 0; if GPU 1 is needed, set CUDA_VISIBLE_DEVICES=1 manually.
# Larger sets (trec-covid 50q, robust04 250q, dbpedia 400q) dominate runtime.

set -euo pipefail
REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate gccp-reproduce
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_CACHE=/media/4TB/share/models/huggingface
export TRANSFORMERS_OFFLINE=1

# Order: small -> large so we get partial results early
DATASETS=("scifact" "nfcorpus" "webis-touche2020" "trec-news" "signal1m" "robust04" "trec-covid" "dbpedia-entity")

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] BEIR anchor ablation on GPU $CUDA_VISIBLE_DEVICES"
echo "Started at $START"

for DS in "${DATASETS[@]}"; do
    OUT="results/ablations/beir_anchor_${DS}_flan-t5-large.json"
    if [ -s "$OUT" ]; then
        echo "[$(date '+%H:%M:%S')] === skip (exists): $DS ==="
        continue
    fi
    LOG="$LOG_DIR/beir_anchor_${DS}_$(date '+%Y%m%d_%H%M%S').log"
    echo
    echo "[$(date '+%H:%M:%S')] === BEIR anchor: $DS / Flan-T5-Large ==="
    python experiments/ablation_studies/run_beir_anchor.py \
        --dataset "$DS" --model flan-t5-large 2>&1 | tee "$LOG"
done

echo
echo "[$(date '+%H:%M:%S')] all BEIR anchor done. Started $START"
