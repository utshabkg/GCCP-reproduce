#!/bin/bash
# Re-run DL19 and DL20 with Flan-T5-Large + pyserini BM25 to save per-query scores
# (the original 2026-03 runs only saved aggregate metrics).
# Output goes to results/trec-dl/dl{19,20}/flan-t5-large_bm25/ to keep
# separate from the existing flan-t5-large_metrics.json snapshot.

set -euo pipefail

REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate gccp-reproduce

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] starting DL19 + DL20 BM25 score-saving runs (Flan-T5-Large)"
echo "Pipeline: pyserini BM25 (k1=0.9, b=0.4) -> RG-YN + GCCP -> save scores"
echo "Started at $START"
echo

for DS in dl19 dl20; do
    OUT="results/trec-dl/$DS/flan-t5-large_bm25"
    LOG="$LOG_DIR/${DS}_t5large_bm25_$(date '+%Y%m%d_%H%M%S').log"
    echo "[$(date '+%H:%M:%S')] === $DS (Flan-T5-Large + BM25) ==="
    echo "Output dir: $OUT"
    echo "Log:        $LOG"
    python scripts/run_experiment.py \
        --dataset "$DS" \
        --model flan-t5-large \
        --use_pyserini \
        --output_dir "$OUT" 2>&1 | tee "$LOG"
    echo "[$(date '+%H:%M:%S')] === done $DS ==="
    echo
done

# Run the aggregation ablation on each
for DS in dl19 dl20; do
    SCORES_DIR="results/trec-dl/$DS/flan-t5-large_bm25"
    OUT="results/ablations/aggregation_${DS}_t5large_bm25.json"
    echo "[$(date '+%H:%M:%S')] aggregation ablation: $DS / T5-Large / BM25"
    python experiments/aggregation_ablation/aggregate.py \
        --rg_yn_scores "$SCORES_DIR/rg_yn_scores.json" \
        --gccp_scores  "$SCORES_DIR/gccp_scores.json" \
        --qrels        "data/${DS}_qrels.json" \
        --output       "$OUT" \
        --label        "$DS / Flan-T5-Large / BM25"
    echo
done

echo "[$(date '+%H:%M:%S')] all done. Started: $START"
