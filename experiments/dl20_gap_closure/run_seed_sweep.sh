#!/bin/bash
# DL20 / Flan-T5-Large / BM25 seed sweep.
# The paper's code sets random.seed(929) (undocumented). We test whether the
# residual ~5% PAGC gap on DL20 T5-Large is plausibly just seed variance.
# Seeds: {0, 42, 929, 12345, 2023}.
#
# Output: results/trec-dl/dl20/flan-t5-large_bm25_seed{N}/ for each seed,
# plus a small summary script to print mean+std across seeds.

set -euo pipefail
REPO_ROOT="/media/12TB/shared/shared_projects/GCCP-reproduce"
cd "$REPO_ROOT"
LOG_DIR="$REPO_ROOT/logs"
mkdir -p "$LOG_DIR"

source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate gccp-reproduce
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HUB_CACHE=/media/4TB/share/models/huggingface
export TRANSFORMERS_OFFLINE=1

START="$(date '+%Y-%m-%d %H:%M:%S')"
echo "[$(date '+%H:%M:%S')] DL20 T5-Large seed sweep on GPU 1"

for SEED in 0 42 929 12345 2023; do
    OUT="results/trec-dl/dl20/flan-t5-large_bm25_seed${SEED}"
    LOG="$LOG_DIR/dl20_t5large_seed${SEED}_$(date '+%Y%m%d_%H%M%S').log"
    if [ -s "$OUT/metrics.json" ]; then
        echo "[$(date '+%H:%M:%S')] skip (exists): seed=$SEED"
        continue
    fi
    echo
    echo "[$(date '+%H:%M:%S')] === seed $SEED ==="
    PYTHONHASHSEED="$SEED" GCCP_SEED="$SEED" python -c "
import os, random, numpy as np, torch
seed = int(os.environ['GCCP_SEED'])
random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
import runpy, sys
sys.argv = ['run_experiment.py', '--dataset', 'dl20', '--model', 'flan-t5-large', '--use_pyserini', '--output_dir', '$OUT']
runpy.run_path('scripts/run_experiment.py', run_name='__main__')
" 2>&1 | tee "$LOG"
done

echo
echo "[$(date '+%H:%M:%S')] all seeds done. Summary:"
python -c "
import json, glob, statistics
results = {}
for d in sorted(glob.glob('results/trec-dl/dl20/flan-t5-large_bm25_seed*')):
    seed = d.split('seed')[-1]
    m = json.load(open(d + '/metrics.json'))['results']
    results[seed] = m
print(f'{\"seed\":>6} {\"BM25\":>8} {\"RG-YN\":>8} {\"GCCP\":>8} {\"PAGC\":>8}')
for seed, m in results.items():
    print(f'{seed:>6} {m[\"bm25\"][\"ndcg@10\"]:>8.4f} {m[\"rg_yn\"][\"ndcg@10\"]:>8.4f} {m[\"gccp\"][\"ndcg@10\"]:>8.4f} {m[\"pagc\"][\"ndcg@10\"]:>8.4f}')
for k in ['rg_yn','gccp','pagc']:
    vals = [m[k]['ndcg@10'] for m in results.values()]
    print(f'  {k}: mean={statistics.mean(vals):.4f} std={statistics.stdev(vals):.4f} range=[{min(vals):.4f},{max(vals):.4f}]')
"
echo "[$(date '+%H:%M:%S')] all done. Started $START"
