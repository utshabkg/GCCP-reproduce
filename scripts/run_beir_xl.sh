#!/bin/bash
# Run BEIR evaluation with Flan-T5-XL

source ~/miniconda3/etc/profile.d/conda.sh
conda activate gccp-reproduce
cd /media/12TB/shared/shared_projects/GCCP-reproduce

DATASETS=("scifact" "nfcorpus" "webis-touche2020" "dbpedia-entity" "trec-covid" "trec-news" "robust04" "signal1m")

for ds in "${DATASETS[@]}"; do
    LOG_FILE="logs/beir_${ds}_t5xl_$(date +%Y%m%d_%H%M%S).log"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running $ds with Flan-T5-XL at $(date)" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    python scripts/run_beir.py --dataset "$ds" --model flan-t5-xl 2>&1 | tee -a "$LOG_FILE"
    echo "" | tee -a "$LOG_FILE"
done

echo "All BEIR T5-XL experiments complete at $(date)"
