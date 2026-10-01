#!/usr/bin/env bash
# Builds LineFormer's isolated Python 3.10 environment on a local Linux + CUDA
# machine (no Colab -- see docs/handoff/2026-10-01-lineformer-rerun-issue.md for
# why the deployment-constraint argument needs a local run).
#
# Differs from notebooks/lineformer_colab.ipynb's pins on purpose: the Colab
# run used torch 1.13.1+cu117 with OpenMMLab's prebuilt mmcv-full wheel, but
# download.openmmlab.com (the only host of those wheels) is unreachable from
# behind the NIMS proxy. So mmcv-full 1.7.2 is built from PyPI's sdist
# against the machine's own CUDA toolkit (nvcc 12.0), which needs a torch
# built for CUDA 12: torch 2.1.2+cu121. LineFormer's vendored mmdetection and
# checkpoint are unchanged. Inference only, so the torch bump is not expected
# to move scores; the versions actually used are printed at the end.
#
# Host compiler: nvcc 12.0 rejects gcc 13, so CC/CXX default to gcc-12.
#
# Usage: scripts/eval/lineformer/setup_local.sh [LINEFORMER_HOME]
#   LINEFORMER_HOME defaults to ~/.cache/real-chart-bench/lineformer
set -euo pipefail

LF_HOME="${1:-$HOME/.cache/real-chart-bench/lineformer}"
VENV="$LF_HOME/venv-cu121"
SRC="$LF_HOME/LineFormer"
PY="$VENV/bin/python"
mkdir -p "$LF_HOME"

echo "[setup] LINEFORMER_HOME=$LF_HOME"
uv python install 3.10
[ -x "$PY" ] || uv venv --python 3.10 -q "$VENV"
uv pip install -q --python "$PY" "setuptools<81"
uv pip install -q --python "$PY" torch==2.1.2+cu121 torchvision==0.16.2+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121
uv pip install -q --python "$PY" "numpy<2" ninja wheel packaging addict yapf \
    opencv-python pyyaml
if ! "$PY" -c "import mmcv.ops" 2>/dev/null; then
    echo "[setup] building mmcv-full 1.7.2 from source (CUDA ops; takes a while)"
    CC="${CC:-gcc-12}" CXX="${CXX:-g++-12}" MMCV_WITH_OPS=1 FORCE_CUDA=1 \
        TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}" MAX_JOBS="${MAX_JOBS:-$(nproc)}" \
        uv pip install --python "$PY" --no-build-isolation --no-binary mmcv-full \
        mmcv-full==1.7.2
fi
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
import mmcv.ops  # noqa: F401 -- the compiled CUDA ops, not just the Python package
sys.path.insert(0, ".")
import infer  # noqa: F401
print(f"[setup] torch={torch.__version__} cuda={torch.version.cuda} "
      f"available={torch.cuda.is_available()} "
      f"device={torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-'}")
print(f"[setup] mmcv={mmcv.__version__} mmdet={mmdet.__version__} "
      f"mmcv_cuda={mmcv.ops.get_compiling_cuda_version()} "
      f"mmcv_cc={mmcv.ops.get_compiler_version()}")
PY
echo "[setup] OK"
