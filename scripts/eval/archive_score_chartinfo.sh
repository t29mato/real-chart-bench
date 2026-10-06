#!/usr/bin/env bash
# Archive one model's answers for a CHART-Infographics batch and score them.
# Usage: scripts/eval/archive_score_chartinfo.sh <work_dir> <batchNN> <model>
set -euo pipefail
W=$1; B=$2; M=$3
R=$(cd "$(dirname "$0")/../.." && pwd); D=$R/data/chartinfo_runs/$B
cp "$W/$B/$M/predictions.json" "$D/$M.predictions.json"
for f in events.jsonl last_message.txt; do
  [ -f "$W/$B/$M.$f" ] && cp "$W/$B/$M.$f" "$D/$M.$f"
done
"$R/.venv/bin/python" "$R/scripts/eval/score_chartinfo_pilot.py" "$D/$M.predictions.json" \
  "$D/ground_truth.json" --key "$D/_key.json" > "$D/$M.score.txt" 2>/dev/null
grep "全" "$D/$M.score.txt" | head -1
