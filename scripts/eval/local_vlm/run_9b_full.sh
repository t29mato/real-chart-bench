#!/bin/zsh
D=/private/tmp/claude-501/-Users-matotomoya-dev-real-chart-bench/334cd4b0-a98f-4d83-89fc-063cde31bf55/scratchpad/local_vlm
export HF_HOME=$D/hf_home HF_HUB_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
date +START_%s
$D/.venv/bin/python $D/worker_9b_full.py mlx-community/Qwen3.5-9B-8bit 16daa4818c54ce5f5436f929d52542eb65bbed9d $D/runs/qwen3.5-9b-8bit-full ALL calibrated,noaxis
echo EXIT=$?
date +END_%s
