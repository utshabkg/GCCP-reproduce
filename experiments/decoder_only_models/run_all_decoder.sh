#!/bin/bash
# Run RG-YN + GCCP + PAGC with two decoder-only LLMs on TREC DL 2019/2020.
#
# We pin to GPU 1 (assumed free; the full-ablation byobu window uses GPU 0).
# Models:
#   - meta-llama/Meta-Llama-3.1-8B-Instruct  (~16 GB FP16)
#   - Qwen/Qwen2.5-7B-Instruct               (~14 GB FP16)
#
# Output:
#   results/trec-dl/dl{19,20}/{llama3-8b,qwen2.5-7b}_bm25/{metrics,*_scores}.json
set -euo pipefail

REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
# Decoder-only models (LLaMA-3.1, Qwen-2.5) need transformers >= 4.43,
# but the gccp-reproduce env is pinned to 4.36 for the T5 pipeline.
# Use a separate env (gccp-decoder) to avoid disrupting in-flight runs.
conda activate gccp-decoder

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] starting decoder-only LLM runs"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

run_one() {
    local model="$1" short="$2" ds="$3"
    local log="$LOG_DIR/decoder_${short}_${ds}_$(date '+%Y%m%d_%H%M%S').log"
    echo
    echo "[$(date '+%H:%M:%S')] === $short / $ds ==="
    echo "Model: $model"
    echo "Log:   $log"
    python experiments/decoder_only_models/run_decoder.py \
        --dataset "$ds" \
        --model "$model" \
        --short_name "$short" 2>&1 | tee "$log"
    echo "[$(date '+%H:%M:%S')] === done $short / $ds ==="
}

for ds in dl19 dl20; do
    run_one "meta-llama/Meta-Llama-3.1-8B-Instruct" "llama3-8b" "$ds"
done

for ds in dl19 dl20; do
    run_one "Qwen/Qwen2.5-7B-Instruct" "qwen2.5-7b" "$ds"
done

echo
echo "[$(date '+%H:%M:%S')] all done. Started: $START"
