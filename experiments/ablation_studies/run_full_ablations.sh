#!/bin/bash
# Full-size anchor + parameter ablations on DL19 + DL20 with Flan-T5-Large.
# Replaces Ethan's preliminary 5-query sweep with the full query sets.
#
# Output:
#   results/ablations/full_dl19_t5large/{anchor_methods,param_*,ablation_summary}.json
#   results/ablations/full_dl20_t5large/{anchor_methods,param_*,ablation_summary}.json

set -euo pipefail

REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate gccp-reproduce

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] starting full-size ablations on DL19 + DL20 (Flan-T5-Large)"
echo "Started at $START"

for DS in dl19 dl20; do
    OUT="results/ablations/full_${DS}_t5large"
    LOG="$LOG_DIR/full_ablations_${DS}_t5large_$(date '+%Y%m%d_%H%M%S').log"
    mkdir -p "$OUT"
    echo
    echo "[$(date '+%H:%M:%S')] === $DS / Flan-T5-Large (full query set) ==="
    echo "Output: $OUT"
    echo "Log:    $LOG"
    python experiments/ablation_studies/run_all_ablations.py \
        --dataset "$DS" \
        --model flan-t5-large \
        --output_dir "$OUT" \
        --seed 929 2>&1 | tee "$LOG"
    echo "[$(date '+%H:%M:%S')] === done $DS ==="
done

echo
echo "[$(date '+%H:%M:%S')] all done. Started: $START"
