#!/usr/bin/env python3
"""
Standalone Discord heartbeat watcher for long-running tqdm-based steps.

Tails a step log every PERIOD seconds, extracts the latest tqdm progress
line ('queries: N/M [E<R, ...]') and pings DISCORD_WEBHOOK with the
parsed counts + ETA. Exits when the parent PID exits, OR when the log
hasn't been updated in 3*PERIOD seconds, OR when a "Done in" line
appears.

Usage:
    python _progress_heartbeat.py STEP_NAME LOG_PATH WATCH_PID PERIOD_SEC

Designed to be spawned in the background by run_bge_full.sh.
"""
import os, re, sys, time, json, urllib.request
from pathlib import Path

step, log_path, watch_pid, period = sys.argv[1], Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
webhook = os.environ.get('DISCORD_WEBHOOK', '')

def alive(pid):
    try:
        os.kill(pid, 0); return True
    except OSError:
        return False

def post(msg):
    if not webhook:
        return
    try:
        data = json.dumps({'content': msg}).encode()
        req = urllib.request.Request(webhook, data=data,
            headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:
        pass

TQDM_RE = re.compile(
    r'queries:\s*(\d+)%\|.*?\|\s*(\d+)/(\d+)\s*\[(\d+:\d+(?::\d+)?)<(\d+:\d+(?::\d+)?),\s*([\d.]+)\s*(s|it)/(it|s)\]'
)
DONE_RE = re.compile(r'Done in', re.IGNORECASE)

last_progress = None
last_log_mtime = 0
last_log_size = 0
stagnant_iters = 0
last_post_time = 0

while alive(watch_pid):
    time.sleep(period)
    if not log_path.exists():
        continue
    st = log_path.stat()
    # Read tail (last 64KB) to find latest tqdm line and Done marker
    with log_path.open('rb') as f:
        f.seek(max(0, st.st_size - 65536))
        raw = f.read().decode('utf-8', errors='replace')
    # tqdm uses \r to overwrite; split on both \r and \n
    lines = re.split(r'[\r\n]+', raw)
    # Find last tqdm progress line
    latest = None
    for ln in reversed(lines):
        m = TQDM_RE.search(ln)
        if m:
            latest = m; break
    done = any(DONE_RE.search(ln) for ln in lines[-10:])
    if done:
        post(f'✅ [{step}] completed (parsed from log)')
        break
    if latest:
        pct, done_n, total, elapsed, eta, rate, _, _ = latest.groups()
        progress = f'{done_n}/{total} ({pct}%)'
        if progress != last_progress:
            post(f'🔄 [{step}] {progress} — elapsed {elapsed} — ETA {eta} — {rate}s/q')
            last_progress = progress
            stagnant_iters = 0
        else:
            stagnant_iters += 1
            # If progress hasn't advanced for >3 periods, alert
            if stagnant_iters == 3:
                post(f'⚠️ [{step}] no progress in ~{3*period//60}min — last: {progress}')
    # Detect log silence
    if st.st_mtime == last_log_mtime and st.st_size == last_log_size:
        # log file has not been written to since last poll
        pass
    last_log_mtime = st.st_mtime
    last_log_size = st.st_size

# Watch PID exited
post(f'🏁 [{step}] watch PID exited — heartbeat stopping')
