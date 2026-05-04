#!/bin/bash
# Anchor-variant sweep on all 8 BEIR sets, Flan-T5-Large, BM25.
# Tests the four most plausible operationalizations of the paper's
# "Random" and "Top" baselines that could explain the +0.017 gap
# between our reported numbers and the paper's Table 5.
set -eo pipefail
REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate gccp-reproduce
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HUB_CACHE=/media/4TB/share/models/huggingface
export TRANSFORMERS_OFFLINE=1

DATASETS=("scifact" "nfcorpus" "trec-covid" "webis-touche2020" "trec-news" "robust04" "signal1m" "dbpedia-entity")

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] anchor-variant sweep on GPU $CUDA_VISIBLE_DEVICES (started $START)"

for DS in "${DATASETS[@]}"; do
    OUT="$REPO_ROOT/results/ablations/beir_anchor_variants_${DS}_flan-t5-large.json"
    if [ -s "$OUT" ]; then
        echo "[$(date '+%H:%M:%S')] === skip (exists): $DS ==="
        continue
    fi
    LOG="$LOG_DIR/anchor_variants_${DS}_$(date '+%Y%m%d_%H%M%S').log"
    echo
    echo "[$(date '+%H:%M:%S')] === anchor variants: $DS ==="
    echo "Log: $LOG"
    python experiments/ablation_studies/run_anchor_variants.py \
        --dataset "$DS" --model flan-t5-large 2>&1 | tee "$LOG"
done

echo
echo "[$(date '+%H:%M:%S')] all done. Started $START"
