#!/bin/bash
# Full BGE second-retriever pipeline with checkpoints, Discord notifications,
# and 5-min progress heartbeats for long-running tqdm-based steps.
#
# Idempotent: re-running skips any step whose output already exists.
# Designed to run inside a detached byobu session.

set -uo pipefail

# Resolve project root absolutely (don't depend on CWD)
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

source ~/.env  # provides DISCORD_WEBHOOK
export DISCORD_WEBHOOK
mkdir -p logs

RUN_STAMP="$(date '+%Y%m%d_%H%M%S')"
MASTER_LOG="$REPO_ROOT/logs/bge_pipeline_${RUN_STAMP}.log"
exec > >(tee -a "$MASTER_LOG") 2>&1
echo "[$(date)] master log: $MASTER_LOG"

PERIOD=300  # heartbeat every 5 min

stamp() { date '+%H:%M:%S'; }
notify() {
    local msg="$1"
    if [ -n "${DISCORD_WEBHOOK:-}" ]; then
        curl -s -X POST -H 'Content-Type: application/json' \
             -d "$(jq -nc --arg c "$msg" '{content: $c}')" \
             "$DISCORD_WEBHOOK" >/dev/null 2>&1 || true
    fi
    echo "[$(stamp)] [discord] $msg"
}

run_step() {
    # run_step <step_name> <checkpoint> <CMD...>
    local name="$1"; local checkpoint="$2"; shift 2
    if [ -s "$checkpoint" ]; then
        notify "✓ skip [$name] — checkpoint $checkpoint exists"
        return 0
    fi
    local step_log="$REPO_ROOT/logs/bge_${name}_${RUN_STAMP}.log"
    notify "🚀 start [$name] — log $step_log"
    # Launch step in background, capture PID, start heartbeat watcher
    "$@" > "$step_log" 2>&1 &
    local step_pid=$!
    # Spawn heartbeat (uses the same conda env-independent python3 so it
    # always runs even if the main conda env is busy)
    python3 "$REPO_ROOT/experiments/beir_e5/_progress_heartbeat.py" \
        "$name" "$step_log" "$step_pid" "$PERIOD" >/dev/null 2>&1 &
    local hb_pid=$!
    wait "$step_pid"
    local rc=$?
    kill "$hb_pid" 2>/dev/null || true
    if [ "$rc" -eq 0 ]; then
        local tail_summary
        tail_summary=$(tail -3 "$step_log" | tr '\n' ' ' | cut -c1-220)
        notify "✅ done [$name] | $tail_summary"
        return 0
    else
        local err_tail
        err_tail=$(tail -10 "$step_log" | tr '\n' ' ' | cut -c1-500)
        notify "❌ FAIL [$name] — exit=$rc — tail: $err_tail"
        return $rc
    fi
}

notify "🟢 BGE pipeline starting (stamp=$RUN_STAMP, host=$(hostname))"

# ============================================================
# Step 1-3: BGE retrieval (all 3 already done; will skip via checkpoints)
# ============================================================
run_step "bge-retrieve-dbpedia" \
    "$REPO_ROOT/data/beir_bge_dbpedia-entity.json" \
    conda run --no-capture-output -n gccp-decoder python -u \
    "$REPO_ROOT/experiments/beir_e5/generate_beir_bge.py" --dataset dbpedia-entity || exit 1

run_step "bge-retrieve-robust04" \
    "$REPO_ROOT/data/beir_bge_robust04.json" \
    conda run --no-capture-output -n gccp-decoder python -u \
    "$REPO_ROOT/experiments/beir_e5/generate_beir_bge.py" --dataset robust04 --from_dump || exit 1

run_step "bge-retrieve-nfcorpus" \
    "$REPO_ROOT/data/beir_bge_nfcorpus.json" \
    conda run --no-capture-output -n gccp-decoder python -u \
    "$REPO_ROOT/experiments/beir_e5/generate_beir_bge.py" --dataset nfcorpus || exit 1

# ============================================================
# Step 4: Flan-T5-XL rerank — the long pole, ~2 hours total
# ============================================================
for ds in dbpedia-entity robust04 nfcorpus; do
    run_step "rerank-${ds}" \
        "$REPO_ROOT/results/beir/${ds}/flan-t5-xl_bge/metrics.json" \
        conda run --no-capture-output -n gccp-reproduce python -u \
        "$REPO_ROOT/experiments/beir_e5/rerank_beir_bge.py" --dataset "${ds}" --model flan-t5-xl || exit 1
done

# ============================================================
# Step 5: Paired bootstrap + Holm (auto-discovers new BGE score dirs)
# ============================================================
rm -f "$REPO_ROOT/results/stat_tests/all_paired_bootstrap.json"
run_step "stat-tests-with-bge" \
    "$REPO_ROOT/results/stat_tests/all_paired_bootstrap.json" \
    conda run --no-capture-output -n gccp-reproduce python -u \
    "$REPO_ROOT/experiments/statistical_tests/run_all_stat_tests.py" || exit 1

# ============================================================
# Step 6: Quick summary of BGE deltas
# ============================================================
notify "📊 extracting BGE deltas"
python3 - <<'PYEOF' 2>&1 | tee -a "$MASTER_LOG"
import json
summary = json.load(open('results/stat_tests/all_paired_bootstrap.json'))
print("\n=== BGE results ===")
for ds in ['dbpedia-entity', 'robust04', 'nfcorpus']:
    key = f'beir/{ds}/flan-t5-xl_bge'
    if key not in summary:
        print(f"  {ds}: MISSING")
        continue
    s = summary[key]
    print(f"\n  {ds} (n_q={s['n_queries']}):")
    for fam, c in s['comparisons'].items():
        d = c['delta_mean']; p = c['p_value']; ph = c.get('p_value_holm', None)
        print(f"    {fam:18s}  Delta={d:+.4f}  p_raw={p:.4f}  p_holm={ph}")
PYEOF
DELTAS_TAIL=$(tail -30 "$MASTER_LOG" | grep -E 'Delta|n_q=' | tail -20 | tr '\n' ' ' | cut -c1-1500)
notify "🎉 BGE pipeline COMPLETE | $DELTAS_TAIL"

echo "DONE" > "$REPO_ROOT/logs/bge_pipeline_done.marker"
