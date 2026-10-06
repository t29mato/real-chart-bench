#!/usr/bin/env bash
# Resume a local-model Codex batch whose session ended before every task was
# answered (the local model sometimes ends its turn with a progress message
# instead of a tool call). Same session (`codex exec resume <id>`), same
# provider and sandbox as run_codex_local_batch.sh; the prompt is neutral, as
# for the GPT batch resumed after a server error (design 7.81): it names no
# figure. The log goes to <batch>.resume<k>.events.jsonl (archived with the
# batch; collect_run_costs.py sums it into the same session).
#
# The caller holds the GPU lock.
#
# Usage: scripts/eval/resume_codex_local_batch.sh <sealed_batch_dir> <ollama_model> <context_tokens> <k>
set -u
d=$(realpath "$1"); model=$2; ctx=$3; k=$4
export CODEX_HOME=${CODEX_HOME:-$HOME/.cache/real-chart-bench/codex-local/home}
session=$(grep -o '"thread_id":"[^"]*"' "$d.events.jsonl" | head -1 | cut -d'"' -f4)
cd "$d" || exit 1
codex exec resume "$session" -m "$model" \
  -c model_provider=ollama -c sandbox_mode=workspace-write \
  -c model_context_window="$ctx" -c model_reasoning_effort=medium \
  --skip-git-repo-check --json -o "$d.last_message.txt" \
  "The previous turn ended before every task was answered. Continue the task in INSTRUCTIONS.md from where you left off (predictions.json already has the finished figures). Work only inside this directory." \
  < /dev/null > "$d.resume$k.events.jsonl" 2> "$d.resume$k.stderr.txt"
echo "$d resume$k exit=$?"
