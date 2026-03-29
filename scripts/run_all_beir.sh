#!/bin/bash
# Run all BEIR datasets with fixed NDCG calculation

cd /media/12TB/shared/shared_projects/GCCP-reproduce
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gccp-reproduce

DATASETS=("scifact" "nfcorpus" "touche" "dbpedia-entity" "robust04" "signal1m")
MODEL="flan-t5-large"

for dataset in "${DATASETS[@]}"; do
    echo "=========================================="
    echo "Running: $dataset"
    echo "=========================================="
    python scripts/run_beir.py --dataset "$dataset" --model "$MODEL" 2>&1 | tee "logs/beir_${dataset}_fixed_$(date +%Y%m%d_%H%M%S).log"
    echo ""
done

echo "All BEIR datasets completed!"
