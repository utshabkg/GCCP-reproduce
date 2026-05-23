#!/bin/bash
# Watchdog: after DBPedia rerank completes, verify Robust04 is streaming.
# If it isn't (bash cached the pre-patch conda-run), kill + relaunch the
# pipeline cleanly (checkpoint-based restart skips DBPedia).

set -uo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# DISCORD_WEBHOOK
source ~/.env

DBPEDIA_METRICS="$REPO_ROOT/results/beir/dbpedia-entity/flan-t5-xl_bge/metrics.json"
# The currently-running pipeline stamped its files with 20260522_161918
ROBUST04_LOG="$REPO_ROOT/logs/bge_rerank-robust04_20260522_161918.log"

notify() {
    [ -n "${DISCORD_WEBHOOK:-}" ] && curl -s -X POST \
        -H 'Content-Type: application/json' \
        -d "$(jq -nc --arg c "$1" '{content: $c}')" \
        "$DISCORD_WEBHOOK" >/dev/null 2>&1 || true
    echo "[$(date '+%H:%M:%S')] [watchdog] $1"
}

notify "🐕 watchdog up — waiting for DBPedia rerank to write metrics.json"

# Phase 1: wait for DBPedia metrics.json to land (60s polling)
while [ ! -s "$DBPEDIA_METRICS" ]; do
    sleep 60
done
notify "🐕 DBPedia metrics.json present — pausing 120s, then checking Robust04 stream"

# Phase 2: give the next iteration 2 minutes to start producing output
sleep 120

if [ -s "$ROBUST04_LOG" ]; then
    notify "🐕 Robust04 log streaming — bash re-read worked, no intervention"
    exit 0
fi

# Phase 3: bash cached the pre-patch loop. Kill + clean relaunch.
notify "🐕 Robust04 log empty after 2min — killing buffered pipeline and relaunching"
byobu kill-session -t bge-pipeline 2>/dev/null || true
sleep 1
pkill -f 'rerank_beir_bge' 2>/dev/null || true
pkill -f '_progress_heartbeat' 2>/dev/null || true
sleep 3

byobu new-session -d -s bge-pipeline \
    "bash $REPO_ROOT/experiments/beir_e5/run_bge_full.sh; echo '--- exited, press enter ---'; read"
notify "🐕 relaunched pipeline; should skip DBPedia via checkpoint and start Robust04 with --no-capture-output"
