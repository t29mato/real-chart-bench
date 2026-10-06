#!/usr/bin/env bash
# run_local.py --policy vlm under the shared GPU lock, with the env the
# local-cuda rows used. Usage: run_local.sh --run <name> [--dev N] [--conditions ...]
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${RCB_VLM_PY:-$HOME/.cache/real-chart-bench/vlm-cuda/bin/python}"
export HF_HUB_OFFLINE=1 VLLM_USE_FLASHINFER_SAMPLER=0 TOKENIZERS_PARALLELISM=false
exec flock /tmp/rcb-gpu.lock "$PY" "$HERE/run_local.py" --policy vlm "$@"
