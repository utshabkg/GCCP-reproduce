#!/bin/bash
# Extend BEIR-E5 to the remaining 6 BEIR sets (we already have scifact + nfcorpus).
# Stage 1 (E5 retrieval) on GPU; stage 2 (Flan-T5-XL rerank) on the same GPU.

set -euo pipefail
REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true

DATASETS=("trec-covid" "webis-touche2020" "dbpedia-entity" "trec-news" "robust04" "signal1m")

# ---------------- Stage 1: E5 retrieval (gccp-decoder env, GPU) -----------
conda activate gccp-decoder
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
# Tell sentence-transformers to use the locally cached E5 model
export HF_HUB_CACHE=/media/20TB/shared/models/intfloat/e5-base-v2
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=0  # ir_datasets may need to fetch BEIR archives

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] BEIR-E5 stage 1 (remaining 6 datasets) on GPU $CUDA_VISIBLE_DEVICES"
echo "Started at $START"

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

# ---------------- Stage 2: Flan-T5-XL rerank (gccp-reproduce env) ---------
conda activate gccp-reproduce
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HUB_CACHE=/media/4TB/share/models/huggingface
export TRANSFORMERS_OFFLINE=1

for DS in "${DATASETS[@]}"; do
    OUT="results/beir/${DS}/flan-t5-xl_e5"
    if [ -s "$OUT/metrics.json" ]; then
        echo "[$(date '+%H:%M:%S')] === rerank skip (exists): $DS ==="
        continue
    fi
    LOG="$LOG_DIR/beir_e5_rerank_${DS}_$(date '+%Y%m%d_%H%M%S').log"
    echo
    echo "[$(date '+%H:%M:%S')] === rerank: $DS / Flan-T5-XL ==="
    echo "Log: $LOG"
    python experiments/beir_e5/rerank_beir_e5.py --dataset "$DS" --model flan-t5-xl 2>&1 | tee "$LOG"
done

# Update stat tests with the new BEIR-E5 score files
python experiments/statistical_tests/run_all_stat_tests.py 2>&1 \
    | tee "$LOG_DIR/stat_tests_after_beir_e5_$(date '+%Y%m%d_%H%M%S').log"

echo
echo "[$(date '+%H:%M:%S')] all done. Started $START"
