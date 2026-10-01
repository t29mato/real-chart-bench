#!/usr/bin/env bash
# Builds LineFormer's isolated Python 3.10 environment on a local Linux + CUDA
# machine (no Colab -- see docs/handoff/2026-10-01-lineformer-rerun-issue.md for
# why the deployment-constraint argument needs a local run).
#
# Same pins as notebooks/lineformer_colab.ipynb (design §7.35-§7.37):
# torch 1.13.1+cu117, mmcv-full 1.7.2, LineFormer's vendored mmdetection.
#
# Usage: scripts/eval/lineformer/setup_local.sh [LINEFORMER_HOME]
#   LINEFORMER_HOME defaults to ~/.cache/real-chart-bench/lineformer
set -euo pipefail

LF_HOME="${1:-$HOME/.cache/real-chart-bench/lineformer}"
VENV="$LF_HOME/venv"
SRC="$LF_HOME/LineFormer"
PY="$VENV/bin/python"
mkdir -p "$LF_HOME"

echo "[setup] LINEFORMER_HOME=$LF_HOME"
uv python install 3.10
[ -x "$PY" ] || uv venv --python 3.10 -q "$VENV"
uv pip install -q --python "$PY" "setuptools<81"
uv pip install -q --python "$PY" torch==1.13.1+cu117 torchvision==0.14.1+cu117 \
    --extra-index-url https://download.pytorch.org/whl/cu117
uv pip install -q --python "$PY" mmcv-full==1.7.2 \
    -f https://download.openmmlab.com/mmcv/dist/cu117/torch1.13.0/index.html
[ -d "$SRC" ] || git clone --depth 1 https://github.com/TheJaeLal/LineFormer.git "$SRC"
uv pip install -q --python "$PY" --no-build-isolation -e "$SRC/mmdetection"
uv pip install -q --python "$PY" chardet scikit-image matplotlib \
    opencv-python pillow scipy==1.9.3 bresenham tqdm "numpy<2"

CKPT_DIR="$SRC/checkpoints"
CKPT="$CKPT_DIR/iter_3000.pth"
mkdir -p "$CKPT_DIR"
if [ ! -f "$CKPT" ]; then
    gdown --folder -q --continue -O "$CKPT_DIR" \
        "https://drive.google.com/drive/folders/1K_zLZwgoUIAJtfjwfCU5Nv33k17R0O5T"
    found="$(find "$CKPT_DIR" -name iter_3000.pth | head -1)"
    [ -n "$found" ] && [ "$found" != "$CKPT" ] && mv "$found" "$CKPT"
fi
[ -f "$CKPT" ] || { echo "[setup] checkpoint download failed" >&2; exit 1; }

cd "$SRC"
MPLBACKEND=Agg "$PY" - <<'PY'
import sys
import torch, mmcv, mmdet
sys.path.insert(0, ".")
import infer  # noqa: F401
print(f"[setup] torch={torch.__version__} cuda={torch.version.cuda} "
      f"available={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
print(f"[setup] mmcv={mmcv.__version__} mmdet={mmdet.__version__}")
PY
echo "[setup] OK"
