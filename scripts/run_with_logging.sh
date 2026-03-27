#!/bin/bash
# run_with_logging.sh - Wrapper script to run experiments with logging
#
# Usage:
#   ./scripts/run_with_logging.sh --dataset dl19 --model flan-ul2 --output_dir results/dl19_ul2 --use_pyserini
#
# This script:
#   1. Creates a log file with timestamp
#   2. Logs all output to both console and file
#   3. Records start/end times

set -e

# Parse arguments to extract output_dir for log placement
OUTPUT_DIR="results"
for arg in "$@"; do
    if [[ "$prev_arg" == "--output_dir" ]]; then
        OUTPUT_DIR="$arg"
    fi
    prev_arg="$arg"
done

# Create output directory and logs directory
mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# Generate log filename with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/experiment_${TIMESTAMP}.log"

echo "========================================"
echo "GCCP Experiment Runner"
echo "========================================"
echo "Start time: $(date)"
echo "Arguments: $@"
echo "Log file: $LOG_FILE"
echo "========================================"

# Activate conda environment and run
source ~/anaconda3/etc/profile.d/conda.sh
conda activate gccp-reproduce

# Run experiment with tee to log output
python scripts/run_experiment.py "$@" 2>&1 | tee "$LOG_FILE"

# Also copy log to output directory
cp "$LOG_FILE" "$OUTPUT_DIR/run.log"

echo ""
echo "========================================"
echo "Experiment completed at: $(date)"
echo "Log saved to: $LOG_FILE"
echo "Log also copied to: $OUTPUT_DIR/run.log"
echo "========================================"
