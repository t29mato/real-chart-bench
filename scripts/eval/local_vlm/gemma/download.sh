#!/bin/zsh
# Download gemma-4-31b-it-8bit into a separate HF_HOME, with disk guard (abort if real free < 5GB)
cd "$(dirname $0)"
export HF_HOME=$PWD/hf_home_gemma
REV=f5f3dc92ab4af76724c36c21eb6bedadb3a851be
../.venv/bin/python -c "from huggingface_hub import snapshot_download as s; print('path=' + s('mlx-community/gemma-4-31b-it-8bit', revision='$REV'))" &
PID=$!
while kill -0 $PID 2>/dev/null; do
  FREE_KB=$(df -k / | awk 'NR==2{print $4}')
  echo "$(date +%T) free_gb=$((FREE_KB/1024/1024)) dl_gb=$(du -sk hf_home_gemma 2>/dev/null | awk '{printf "%.1f",$1/1024/1024}')"
  if [ $FREE_KB -lt 5242880 ]; then echo "ABORT: free < 5GB"; kill $PID; sleep 2; kill -9 $PID 2>/dev/null; exit 3; fi
  sleep 15
done
wait $PID; echo "exit=$?"
