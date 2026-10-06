#!/usr/bin/env bash
# Time-sliced QLoRA training on the shared GPU (docs/design/local-model.md 方式B).
# Each slice holds /tmp/rcb-gpu.lock for at most --wall-minutes, saves and
# exits 3; the loop then waits a little so other queued GPU jobs get the lock
# before the next slice. Exit 0 from train.py = finished.
# Usage: run_train.sh <train.py args...>
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${RCB_TRAIN_PY:-$HOME/.cache/real-chart-bench/vlm-train/bin/python}"
export HF_HUB_OFFLINE=1 PYTORCH_ALLOC_CONF=expandable_segments:True TOKENIZERS_PARALLELISM=false
while true; do
  flock /tmp/rcb-gpu.lock "$PY" "$HERE/train.py" "$@"
  code=$?
  if [ "$code" -eq 0 ]; then echo "[run_train] finished"; exit 0; fi
  if [ "$code" -ne 3 ]; then echo "[run_train] train.py failed with $code"; exit "$code"; fi
  echo "[run_train] slice over; yielding the GPU lock"
  sleep 30
done
