#!/bin/bash
# Wait for stage 1 (scifact + nfcorpus) and the UL2 efficiency run to be
# done, then run BEIR-E5 stage 2 reranks on GPU 0 with Flan-T5-XL.

set -euo pipefail

REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true

# Wait for stage 1 outputs (scifact, nfcorpus) and for UL2 to free GPU 0.
echo "[$(date '+%H:%M:%S')] waiting for stage 1 outputs and GPU 0 free..."
while true; do
    have_data=true
    for ds in scifact nfcorpus; do
        if [ ! -s "$REPO_ROOT/data/beir_e5_${ds}.json" ]; then
            have_data=false
        fi
    done
    have_efficiency=true
    if [ ! -s "$REPO_ROOT/results/efficiency/dl19_flan-ul2.json" ]; then
        have_efficiency=false
    fi
    if [ "$have_data" = true ] && [ "$have_efficiency" = true ]; then
        break
    fi
    sleep 30
done

echo "[$(date '+%H:%M:%S')] dependencies ready -- launching stage 2"

conda activate gccp-reproduce
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_CACHE=/media/4TB/share/models/huggingface
export TRANSFORMERS_OFFLINE=1

START="$(date '+%Y-%m-%d %H:%M:%S')"
for DS in scifact nfcorpus; do
    LOG="$LOG_DIR/beir_e5_rerank_${DS}_$(date '+%Y%m%d_%H%M%S').log"
    echo
    echo "[$(date '+%H:%M:%S')] === rerank: $DS / Flan-T5-XL ==="
    echo "Log: $LOG"
    python experiments/beir_e5/rerank_beir_e5.py --dataset "$DS" --model flan-t5-xl 2>&1 | tee "$LOG"
done

echo
echo "[$(date '+%H:%M:%S')] stage 2 done. Started $START"
