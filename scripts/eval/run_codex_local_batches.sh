#!/usr/bin/env bash
# Run sealed Codex batches one after another with the local model
# (run_codex_local_batch.sh). The GPU lock is taken per batch and the Ollama
# model unloaded after each, so a trainer waiting on the lock can run between
# batches. A batch that already has <batch>.last_message.txt is skipped.
#
# Usage: scripts/eval/run_codex_local_batches.sh <ollama_model> <context_tokens> <sealed_batch_dir>...
set -u
here=$(dirname "$(realpath "$0")")
model=$1; ctx=$2; shift 2
for d in "$@"; do
  d=$(realpath "$d")
  if [ -s "$d.last_message.txt" ]; then echo "$d done already"; continue; fi
  echo "$d start $(date -Is)"
  flock /tmp/rcb-gpu.lock bash -c \
    'timeout 14400 "$0" "$1" "$2" "$3"; ollama stop "$2"' \
    "$here/run_codex_local_batch.sh" "$d" "$model" "$ctx"
  echo "$d end $(date -Is)"
done
