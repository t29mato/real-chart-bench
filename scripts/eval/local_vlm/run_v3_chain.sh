#!/bin/zsh
# v3 (markers-only prompt) full runs, ONE MODEL AT A TIME: Qwen3.5-9B -> Qwen3.8-27B -> Gemma 4 31B.
# Resumable (workers skip figs already in the jsonl). Log: runs/v3_chain.log (timestamped lines).
D=/private/tmp/claude-501/-Users-matotomoya-dev-real-chart-bench/334cd4b0-a98f-4d83-89fc-063cde31bf55/scratchpad/local_vlm
export HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
ts() { while IFS= read -r l; do print -r -- "$(date '+%F %T') $l"; done }
filt() { grep --line-buffered -v -E "Warning|warnings.warn|Fetching" }
run() {  # name hf_home worker repo rev
  echo "=== $1 start"
  HF_HOME=$2 $D/.venv/bin/python $3 $4 $5 $D/runs/$1 ALL calibrated,noaxis 2>&1 | filt
  echo "=== $1 end exit=${pipestatus[1]}"
}
{
  echo "=== chain start pid=$$"
  run qwen3.5-9b-8bit-v3  $D/hf_home              $D/worker_v3.py              mlx-community/Qwen3.5-9B-8bit    16daa4818c54ce5f5436f929d52542eb65bbed9d
  run qwen3.8-27b-8bit-v3 $D/hf_home              $D/worker_v3.py              mlx-community/Qwen3.8-27B-8bit   815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9
  run gemma-4-31b-8bit-v3 $D/gemma/hf_home_gemma  $D/gemma/worker_gemma_v3.py  mlx-community/gemma-4-31b-it-8bit f5f3dc92ab4af76724c36c21eb6bedadb3a851be
  echo "=== chain end"
} 2>&1 | ts >> $D/runs/v3_chain.log
