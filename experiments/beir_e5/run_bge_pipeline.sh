#!/bin/bash
# Sequential BGE retrieval + rerank pipeline for the second-retriever
# generalization check (Section 7.7 robustness paragraph).
#
# Run order:
#   1. DBPedia BGE retrieval  (long pole, ~3 hrs)  -- assumed already running
#   2. Robust04 BGE retrieval (~528k passages)     -- from Pyserini dump
#   3. NFCorpus BGE retrieval (already done; skip if file exists)
#   4. Flan-T5-XL rerank on all 3 retrieved sets
#   5. Paired bootstrap + Holm
#
# Usage: bash experiments/beir_e5/run_bge_pipeline.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

mkdir -p logs

stamp() { date '+%H:%M:%S'; }

# ---- Step 1: wait for DBPedia BGE retrieval to finish ----
# (Started separately via Bash background; this script picks up after.)
echo "[$(stamp)] Waiting for data/beir_bge_dbpedia-entity.json"
until [ -s data/beir_bge_dbpedia-entity.json ]; do sleep 120; done
echo "[$(stamp)] DBPedia BGE retrieval file present, continuing"

# ---- Step 2: Robust04 BGE retrieval (Pyserini dump) ----
if [ ! -s data/beir_bge_robust04.json ]; then
    echo "[$(stamp)] Robust04 BGE retrieval starting"
    conda run -n gccp-decoder python -u experiments/beir_e5/generate_beir_bge.py \
        --dataset robust04 --from_dump 2>&1 | tee logs/bge_robust04.log
else
    echo "[$(stamp)] Robust04 BGE retrieval file exists -- skipping"
fi

# ---- Step 3: NFCorpus BGE retrieval (already cached but re-run with consistent settings) ----
if [ ! -s data/beir_bge_nfcorpus.json ]; then
    echo "[$(stamp)] NFCorpus BGE retrieval starting"
    conda run -n gccp-decoder python -u experiments/beir_e5/generate_beir_bge.py \
        --dataset nfcorpus 2>&1 | tee logs/bge_nfcorpus.log
else
    echo "[$(stamp)] NFCorpus BGE retrieval file exists -- skipping"
fi

# ---- Step 4: Flan-T5-XL rerank on each retrieved set ----
for ds in dbpedia-entity robust04 nfcorpus; do
    if [ -s "results/beir/${ds}/flan-t5-xl_bge/metrics.json" ]; then
        echo "[$(stamp)] Rerank for ${ds} exists -- skipping"
        continue
    fi
    echo "[$(stamp)] Reranking ${ds} with Flan-T5-XL"
    conda run -n gccp-reproduce python -u experiments/beir_e5/rerank_beir_bge.py \
        --dataset "${ds}" --model flan-t5-xl 2>&1 | tee "logs/bge_rerank_${ds}.log"
done

# ---- Step 5: Paired bootstrap + Holm (auto-discovers new score dirs) ----
echo "[$(stamp)] Running paired bootstrap + Holm correction"
conda run -n gccp-reproduce python -u experiments/statistical_tests/run_all_stat_tests.py \
    2>&1 | tee logs/bge_stat_tests.log

echo "[$(stamp)] BGE pipeline DONE"
