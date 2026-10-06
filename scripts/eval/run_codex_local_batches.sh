#!/usr/bin/env bash
# Run sealed Codex batches one after another with the local model
# (run_codex_local_batch.sh), resuming a batch whose session ended before
# every task was answered (resume_codex_local_batch.sh, at most 3 times).
#
# - The GPU lock is taken per session and the Ollama model unloaded after
#   each, so a trainer waiting on the lock can run between sessions.
# - Before loading, the GPU must have NEED_MIB free (a process that does not
#   take the lock would otherwise make the model load fail with out of
#   memory); the driver waits outside the lock until it has.
# - A batch whose predictions.json has every task id is skipped.
#
# Usage: scripts/eval/run_codex_local_batches.sh <ollama_model> <context_tokens> <sealed_batch_dir>...
set -u
here=$(dirname "$(realpath "$0")")
model=$1; ctx=$2; shift 2
NEED_MIB=${NEED_MIB:-21500}

missing() {  # task ids not yet in predictions.json
  python3 -c 'import json,sys
d=sys.argv[1]
t=[x["id"] for x in json.load(open(d+"/tasks.json"))]
try: p=json.load(open(d+"/predictions.json"))
except Exception: p={}
print(sum(1 for x in t if x not in p))' "$1"
}
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
gpu_session() {  # run one session under the lock once the GPU has room
  while :; do
    until [ "$(free_mib)" -ge "$NEED_MIB" ]; do sleep 60; done
    flock /tmp/rcb-gpu.lock bash -c '
      [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge "$0" ] || exit 75
      timeout 14400 "$@"; rc=$?; ollama stop "$3"; exit $rc' "$NEED_MIB" "$@"
    [ $? -ne 75 ] && return
  done
}

for d in "$@"; do
  d=$(realpath "$d")
  echo "$d start $(date -Is)"
  if [ ! -e "$d.events.jsonl" ]; then
    gpu_session "$here/run_codex_local_batch.sh" "$d" "$model" "$ctx"
  fi
  for k in 1 2 3; do
    m=$(missing "$d")
    [ "$m" = "0" ] && break
    [ -e "$d.resume$k.events.jsonl" ] && continue
    echo "$d missing=$m, resume$k $(date -Is)"
    gpu_session "$here/resume_codex_local_batch.sh" "$d" "$model" "$ctx" "$k"
  done
  echo "$d end missing=$(missing "$d") $(date -Is)"
done
