#!/usr/bin/env bash
# One Codex CLI batch (design 7.81): codex exec in the sealed directory, the
# workspace-write sandbox (writes only inside it, no network), reasoning medium.
# Event log and final message go next to the directory as <batch>.events.jsonl
# and <batch>.last_message.txt.
#
# Usage: scripts/eval/run_codex_batch.sh <sealed_batch_dir> <model>
set -u
d=$(realpath "$1"); model=$2
codex exec -m "$model" -c model_reasoning_effort=medium -s workspace-write \
  --skip-git-repo-check -C "$d" --json -o "$d.last_message.txt" \
  "Read $d/INSTRUCTIONS.md and follow it exactly. Work only inside that directory." \
  < /dev/null > "$d.events.jsonl" 2> "$d.stderr.txt"
echo "$d exit=$?"
