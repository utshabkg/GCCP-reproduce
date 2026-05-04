#!/bin/bash
# Full 8-set BEIR-E5 overnight run.
#
# Phase A (gccp-reproduce env, has pyserini): dump corpora for trec-news,
#   robust04, signal1m to data/beir_e5_pyserini_dump/<ds>/corpus.jsonl
# Phase B (gccp-decoder env, has sentence-transformers + faiss): E5 retrieval
#   for those 3 datasets via --from_dump.
# Phase C (gccp-reproduce env): Flan-T5-XL stage 2 rerank on all 6 sets.
# Phase D: re-run statistical tests across the broader BEIR-E5 picture.
set -eo pipefail
# Note: -u is intentionally OFF — conda's qt-main_activate.sh references
# an unbound QT_XCB_GL_INTEGRATION which trips set -u on env switch.
REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

PYSERINI_DATASETS=("trec-news" "robust04" "signal1m")
ALL_DATASETS=("trec-covid" "webis-touche2020" "dbpedia-entity" "trec-news" "robust04" "signal1m")

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] BEIR-E5 full-8 overnight on GPU $CUDA_VISIBLE_DEVICES"
echo "Started at $START"

# ---------------- Phase A: pyserini dumps ---------------------------------
conda activate gccp-reproduce
for DS in "${PYSERINI_DATASETS[@]}"; do
    DUMP="$REPO_ROOT/data/beir_e5_pyserini_dump/${DS}/corpus.jsonl"
    if [ -s "$DUMP" ]; then
        echo "[$(date '+%H:%M:%S')] === pyserini-dump skip (exists): $DS ==="
        continue
    fi
    LOG="$LOG_DIR/beir_e5_pyserini_dump_${DS}_$(date '+%Y%m%d_%H%M%S').log"
    echo
    echo "[$(date '+%H:%M:%S')] === pyserini dump: $DS ==="
    echo "Log: $LOG"
    python experiments/beir_e5/dump_pyserini_corpus.py --dataset "$DS" 2>&1 | tee "$LOG"
done
conda deactivate

# ---------------- Phase B: E5 retrieval (gccp-decoder env) ---------------
conda activate gccp-decoder
export HF_HUB_CACHE=/media/20TB/shared/models/intfloat/e5-base-v2
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=0

for DS in "${PYSERINI_DATASETS[@]}"; do
    OUT="$REPO_ROOT/data/beir_e5_${DS}.json"
    if [ -s "$OUT" ]; then
        echo "[$(date '+%H:%M:%S')] === E5 retrieval skip (exists): $DS ==="
        continue
    fi
    LOG="$LOG_DIR/beir_e5_retrieval_${DS}_$(date '+%Y%m%d_%H%M%S').log"
    echo
    echo "[$(date '+%H:%M:%S')] === E5 retrieval (from_dump): $DS ==="
    echo "Log: $LOG"
    python experiments/beir_e5/generate_beir_e5.py --dataset "$DS" --from_dump 2>&1 | tee "$LOG"
done
conda deactivate

# ---------------- Phase C: Flan-T5-XL rerank (gccp-reproduce env) ---------
conda activate gccp-reproduce
unset HF_HUB_CACHE
export HF_HUB_CACHE=/media/4TB/share/models/huggingface
export TRANSFORMERS_OFFLINE=1

for DS in "${ALL_DATASETS[@]}"; do
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

# ---------------- Phase D: stat tests --------------------------------------
python experiments/statistical_tests/run_all_stat_tests.py 2>&1 \
    | tee "$LOG_DIR/stat_tests_after_beir_e5_full8_$(date '+%Y%m%d_%H%M%S').log"

echo
echo "[$(date '+%H:%M:%S')] all done. Started $START"
