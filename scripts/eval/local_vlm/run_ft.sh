#!/usr/bin/env bash
# worker_ft.py under the shared GPU lock, with the env the local-cuda rows used.
# Usage: run_ft.sh bench|val <adapter|none> <out_dir> [...]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${RCB_TRAIN_PY:-$HOME/.cache/real-chart-bench/vlm-train/bin/python}"
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
exec flock /tmp/rcb-gpu.lock "$PY" "$HERE/worker_ft.py" "$@"
