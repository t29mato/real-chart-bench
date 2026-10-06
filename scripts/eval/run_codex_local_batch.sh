#!/usr/bin/env bash
# One Codex CLI batch driven by a LOCAL model (design local-model.md,
# "ローカルエージェント"): the same sealed directory, instructions and
# workspace-write sandbox as run_codex_batch.sh, but the model is served by
# Ollama on this machine (`--oss --local-provider ollama`), so nothing leaves
# the machine.
#
# - The model connection is made by the codex process itself, outside the
#   sandbox; the sandboxed shell commands keep network off. Nothing had to be
#   allowed for localhost.
# - CODEX_HOME is a separate directory (default below) so the user's
#   ~/.codex (ChatGPT sign-in, config.toml) is never touched; its sessions/
#   holds the session logs collect_run_costs.py reads.
# - The caller holds the GPU lock (flock /tmp/rcb-gpu.lock) for as long as
#   the Ollama model is loaded.
#
# Usage: scripts/eval/run_codex_local_batch.sh <sealed_batch_dir> <ollama_model> <context_tokens>
set -u
d=$(realpath "$1"); model=$2; ctx=$3
export CODEX_HOME=${CODEX_HOME:-$HOME/.cache/real-chart-bench/codex-local/home}
codex exec --oss --local-provider ollama -m "$model" \
  -c model_context_window="$ctx" -c model_reasoning_effort=medium -s workspace-write \
  --skip-git-repo-check -C "$d" --json -o "$d.last_message.txt" \
  -- "Read $d/INSTRUCTIONS.md and follow it exactly. Work only inside that directory." \
  < /dev/null > "$d.events.jsonl" 2> "$d.stderr.txt"
echo "$d exit=$?"
