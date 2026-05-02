#!/bin/bash
# Run RG-YN + GCCP + PAGC with decoder-only LLMs on TREC DL 2019/2020.
#
# Uses local model snapshots from /media/20TB/shared/models/ to avoid
# Hugging Face downloads (gccp-reproduce env's preferred cache at
# /media/4TB is full). TRANSFORMERS_OFFLINE=1 prevents accidental network
# fetches.
#
# We pin to GPU 1 by default; pass CUDA_VISIBLE_DEVICES to override.
#
# Output:
#   results/trec-dl/dl{19,20}/{llama3.1-8b,qwen2.5-7b,mistral-7b-v0.3}_bm25/
#   results/trec-dl/dl19/qwen2.5-72b-awq_bm25/  (DL19 only -- 72B is slow)
set -euo pipefail

REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# Keep HF cache on the partition with space, in case anything still tries
# to write there (locks, .no_exist sentinel files).
export HF_HOME="${HF_HOME:-/media/20TB/shared/models/huggingface}"

LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || \
  source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate gccp-decoder

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] starting decoder-only runs"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "Started at $START"

# Local snapshot paths (no HF fetch needed)
LLAMA_PATH=/media/20TB/shared/models/meta-llama/Llama-3.1-8B-Instruct/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659
QWEN7B_PATH=/media/20TB/shared/models/qwen/Qwen2.5-7B-Instruct/snapshots/a09a35458c702b33eeacc393d103063234e8bc28
MISTRAL_PATH=/media/20TB/shared/models/mistralai/Mistral-7B-Instruct-v0.3/models--mistralai--Mistral-7B-Instruct-v0.3/snapshots/c170c708c41dac9275d15a8fff4eca08d52bab71
QWEN72B_PATH=/media/20TB/shared/models/qwen/Qwen2.5-72B-Instruct-AWQ/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/698703eae6604af048a3d2f509995dc302088217

run_one() {
    local model_path="$1" short="$2" ds="$3"
    local log="$LOG_DIR/decoder_${short}_${ds}_$(date '+%Y%m%d_%H%M%S').log"
    echo
    echo "[$(date '+%H:%M:%S')] === $short / $ds ==="
    echo "Path: $model_path"
    echo "Log:  $log"
    python experiments/decoder_only_models/run_decoder.py \
        --dataset "$ds" \
        --model "$model_path" \
        --short_name "$short" 2>&1 | tee "$log"
    echo "[$(date '+%H:%M:%S')] === done $short / $ds ==="
}

# 7-8B trio on both datasets (cross-family comparison at fixed scale)
for ds in dl19 dl20; do
    run_one "$LLAMA_PATH"   "llama3.1-8b"    "$ds"
done

for ds in dl19 dl20; do
    run_one "$QWEN7B_PATH"  "qwen2.5-7b"     "$ds"
done

for ds in dl19 dl20; do
    run_one "$MISTRAL_PATH" "mistral-7b-v0.3" "$ds"
done

# Larger variant: Qwen-2.5-72B-AWQ on DL19 only (~8x slower than 7B,
# included as a scaling data point analogous to Flan-UL2 on the
# encoder-decoder side).
run_one "$QWEN72B_PATH" "qwen2.5-72b-awq" "dl19"

echo
echo "[$(date '+%H:%M:%S')] all done. Started: $START"
