#!/bin/zsh
# pilot (10 figs, both conditions) then full run (ALL ids, both conditions). Resumable.
cd "$(dirname $0)"
export HF_HOME=$PWD/hf_home_gemma HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
REPO_ID=mlx-community/gemma-4-31b-it-8bit; REV=f5f3dc92ab4af76724c36c21eb6bedadb3a851be
echo "=== pilot start $(date)"
../.venv/bin/python worker_gemma.py $REPO_ID $REV ../runs/gemma-4-31b-8bit @sel.json calibrated,noaxis 2>&1 | grep --line-buffered -v -E "Warning|warnings.warn|Fetching"
echo "=== pilot end $(date)"
echo "=== full start $(date)"
../.venv/bin/python worker_gemma.py $REPO_ID $REV ../runs/gemma-4-31b-8bit-full ALL calibrated,noaxis 2>&1 | grep --line-buffered -v -E "Warning|warnings.warn|Fetching"
echo "=== full end $(date)"
