"""Makes weight-only round-to-nearest (RTN) 8-bit and 4-bit copies of a local
VLM for the quantization comparison (design §7.75 (2)).

RTN needs no calibration data, so the result depends only on the weights and
the scheme -- the same family as the Mac run's mlx 8-bit (affine, per group,
no calibration). Only the language model's Linear layers are quantized; the
vision tower and lm_head stay in bf16 (as mlx-vlm leaves the vision tower).

Scheme: llm-compressor W8A16 / W4A16 presets (symmetric int, group size 128),
saved in compressed-tensors format, which vLLM loads directly.

Runs on CPU (the GPU is busy with the bf16 run; RTN does not need one).

Usage: <quantize venv python> quantize_rtn.py <hf_repo_id> <W8A16|W4A16> <out_dir>
"""

import json
import sys
import time

import torch
from huggingface_hub import snapshot_download
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForImageTextToText, AutoProcessor


def main() -> None:
    repo, scheme, out = sys.argv[1:4]
    path = snapshot_download(repo)
    t0 = time.time()
    model = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16)
    processor = AutoProcessor.from_pretrained(path)
    recipe = QuantizationModifier(
        targets="Linear",
        scheme=scheme,
        ignore=["lm_head", "re:.*visual.*", "re:.*vision.*"],
    )
    oneshot(model=model, recipe=recipe)
    model.save_pretrained(out, save_compressed=True)
    processor.save_pretrained(out)
    meta = {
        "source_repo": repo,
        "source_revision": path.rstrip("/").split("/")[-1],
        "scheme": scheme,
        "method": "round-to-nearest, no calibration data (llm-compressor QuantizationModifier)",
        "ignored": ["lm_head", "vision tower"],
        "seconds": round(time.time() - t0, 1),
    }
    with open(f"{out}/rcb_quantization.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta))


if __name__ == "__main__":
    main()
