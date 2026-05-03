#!/bin/bash
# Score-saving runs needed to (1) significance-test Qwen-2.5-7B vs Flan-T5-XL on
# DL19, and (2) extend the aggregation ablation beyond the single
# DL19/T5-Large/BM25 cell that the paper draft currently leans on.
#
# Output:
#   results/trec-dl/dl19/flan-t5-xl_bm25/{rg_yn,gccp,pagc}_scores.json
#   results/trec-dl/dl20/flan-t5-xl_bm25/{rg_yn,gccp,pagc}_scores.json
#   results/ablations/aggregation_dl19_t5xl_bm25.json
#   results/ablations/aggregation_dl20_t5xl_bm25.json

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

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] starting extra-cells score-saving runs (Flan-T5-XL on BM25, DL19+DL20)"

for DS in dl19 dl20; do
    OUT="results/trec-dl/$DS/flan-t5-xl_bm25"
    LOG="$LOG_DIR/${DS}_t5xl_bm25_$(date '+%Y%m%d_%H%M%S').log"
    if [ -s "$OUT/pagc_scores.json" ]; then
        echo "[$(date '+%H:%M:%S')] skip (exists): $OUT"
        continue
    fi
    echo
    echo "[$(date '+%H:%M:%S')] === $DS / Flan-T5-XL / BM25 ==="
    echo "Output: $OUT"
    echo "Log:    $LOG"
    python scripts/run_experiment.py \
        --dataset "$DS" \
        --model flan-t5-xl \
        --use_pyserini \
        --output_dir "$OUT" 2>&1 | tee "$LOG"
done

# Aggregation ablation on each new BM25 score set
for DS in dl19 dl20; do
    SCORES="results/trec-dl/$DS/flan-t5-xl_bm25"
    OUT="results/ablations/aggregation_${DS}_t5xl_bm25.json"
    echo
    echo "[$(date '+%H:%M:%S')] aggregation ablation: $DS / T5-XL / BM25"
    python experiments/aggregation_ablation/aggregate.py \
        --rg_yn_scores "$SCORES/rg_yn_scores.json" \
        --gccp_scores  "$SCORES/gccp_scores.json" \
        --qrels        "data/${DS}_qrels.json" \
        --output       "$OUT" \
        --label        "$DS / Flan-T5-XL / BM25"
done

# Aggregation ablation on the existing scifact + nfcorpus E5 score sets
for DS in scifact nfcorpus; do
    SCORES="results/beir/$DS/flan-t5-xl_e5"
    OUT="results/ablations/aggregation_beir_${DS}_t5xl_e5.json"
    echo
    echo "[$(date '+%H:%M:%S')] aggregation ablation: BEIR-$DS / T5-XL / E5"
    python experiments/aggregation_ablation/aggregate.py \
        --rg_yn_scores "$SCORES/rg_yn_scores.json" \
        --gccp_scores  "$SCORES/gccp_scores.json" \
        --qrels        "data/beir_${DS}_qrels.json" \
        --output       "$OUT" \
        --label        "BEIR-$DS / Flan-T5-XL / E5"
done

# Aggregation ablation on existing E5 / T5-XL DL19+DL20 (already exists, but rerun for completeness)
for DS in dl19 dl20; do
    SCORES="results/trec-dl/$DS/flan-t5-xl_e5"
    OUT="results/ablations/aggregation_${DS}_t5xl_e5.json"
    if [ -s "$OUT" ]; then continue; fi
    python experiments/aggregation_ablation/aggregate.py \
        --rg_yn_scores "$SCORES/rg_yn_scores.json" \
        --gccp_scores  "$SCORES/gccp_scores.json" \
        --qrels        "data/${DS}_qrels.json" \
        --output       "$OUT" \
        --label        "$DS / Flan-T5-XL / E5"
done

# Re-run paired bootstrap with the new T5-XL/BM25 settings included
python experiments/statistical_tests/run_all_stat_tests.py 2>&1 \
    | tee "$LOG_DIR/stat_tests_after_extra_cells_$(date '+%Y%m%d_%H%M%S').log"

# Significance test for Qwen-2.5-7B vs Flan-T5-XL on DL19 (now possible)
python experiments/statistical_tests/qwen_vs_t5xl.py 2>&1 \
    | tee "$LOG_DIR/qwen_vs_t5xl_$(date '+%Y%m%d_%H%M%S').log"

echo
echo "[$(date '+%H:%M:%S')] all done. Started $START"
