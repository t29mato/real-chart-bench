#!/usr/bin/env bash
# Run the CHART-Infographics batches for GPT-6.1-Sol through Codex, a few at a
# time. Before each launch, read the newest Codex session's rate_limits; while
# the 5-hour window is at or above LIMIT %, wait for it to reset.
# Usage: scripts/eval/run_codex_chartinfo_queue.sh <work_dir> <first> <last> [parallel] [limit]
set -u
W=$1; FIRST=$2; LAST=$3; PAR=${4:-3}; LIMIT=${5:-85}
R=$(cd "$(dirname "$0")/../.." && pwd)
used() {
  f=$(ls -t $(find ~/.codex/sessions -name "*.jsonl") 2>/dev/null | head -1)
  grep '"rate_limits"' "$f" | tail -1 | python3 -c "
import json,sys,time
try:
    r=json.loads(sys.stdin.read())['payload']['rate_limits']['primary']
    print(0 if r['resets_at']<time.time() else int(r['used_percent']))
except Exception: print(0)"
}
for k in $(seq "$FIRST" "$LAST"); do
  b=$(printf "batch%02d" "$k")
  [ -f "$W/$b/gpt-6.1-sol.last_message.txt" ] && { echo "$b done already"; continue; }
  while [ "$(jobs -rp | wc -l)" -ge "$PAR" ]; do sleep 30; done
  while [ "$(used)" -ge "$LIMIT" ]; do echo "$(date +%T) codex 5h window >= $LIMIT%, waiting"; sleep 600; done
  echo "$(date +%T) start $b"
  "$R/scripts/eval/run_codex_batch.sh" "$W/$b/gpt-6.1-sol" gpt-6.1-sol > "$W/$b.gpt.runner.log" 2>&1 &
  sleep 20
done
wait
echo "queue finished"
