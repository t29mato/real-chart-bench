"""v3 local-VLM inference on an NVIDIA GPU with vLLM, one precision per run
(design §7.75 (2): how much does quantization cost?).

Same prompt, JSON schema and output parsing as the Mac run -- build_prompt,
schema_for and lenient_parse are imported from worker_v3.py, not copied -- and
the same records, so score_llm_predictions.py scores both alike. What changes
is the engine (vLLM instead of mlx-vlm) and the precision:

  bf16  -- the published weights, unquantized
  fp8   -- vLLM dynamic FP8 (W8A8), the 8-bit format the 4090 runs natively
  w8a16 -- weight-only int8, round-to-nearest, group 128 (quantize_rtn.py)
  w4a16 -- weight-only int4, same method

fp8 is converted in flight by vLLM; w8a16 / w4a16 are made from the same
downloaded weights by scripts/eval/local_vlm/quantize_rtn.py (no calibration
data -- the same family as the Mac's mlx 8-bit) and loaded from
~/.cache/real-chart-bench/quantized/. So the weights are the only source, and
precision is the only variable. (bitsandbytes NF4 was planned, but vLLM 0.30
no longer quantizes with bitsandbytes in flight.)

Figures are sent in chunks (vLLM batches them); a chunk is written as soon as
it finishes and finished figures are skipped on restart. `seconds` is the
chunk's wall time divided by its size: comparable between precisions on this
machine, not with the Mac's one-at-a-time seconds.

Usage: python worker_cuda.py <hf_repo_id> <precision> <out_dir> [conditions]
"""

from __future__ import annotations

import base64
import importlib.metadata as md
import json
import os
import pathlib
import platform
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))

from worker_v3 import build_prompt, lenient_parse, schema_for  # noqa: E402

MAX_TOKENS = 8192
MAX_MODEL_LEN = 16384
CHUNK = 16
QUANT = {"bf16": None, "fp8": "fp8", "w8a16": None, "w4a16": None}
QUANTIZED_DIR = pathlib.Path.home() / ".cache/real-chart-bench/quantized"


def main() -> None:
    repo_id, precision, out_dir = sys.argv[1:4]
    conds = sys.argv[4].split(",") if len(sys.argv) > 4 else ["calibrated", "noaxis"]
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    from huggingface_hub import snapshot_download
    from vllm import LLM, SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    path = snapshot_download(repo_id)  # HF_HUB_OFFLINE=1: resolved from the cache
    source_revision = pathlib.Path(path).name
    quant_meta = None
    if precision in ("w8a16", "w4a16"):
        path = str(QUANTIZED_DIR / f"{repo_id.split('/')[-1].lower()}-{precision}")
        quant_meta = json.loads((pathlib.Path(path) / "rcb_quantization.json").read_text())
    t0 = time.time()
    llm = LLM(
        model=path,
        dtype="bfloat16",
        quantization=QUANT[precision],
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=0.92,
        limit_mm_per_prompt={"image": 1},
        seed=0,
        # CUDA-graph capture needs memory a bf16 9B does not leave on 24GB
        enforce_eager=True,
    )
    load_s = time.time() - t0
    import torch

    env = {
        "repo_id": repo_id,
        "revision": source_revision,
        "local_quantization": quant_meta,
        "precision": precision,
        "vllm_quantization": QUANT[precision],
        "vllm": md.version("vllm"),
        "torch": torch.__version__,
        "transformers": md.version("transformers"),
        "gpu": torch.cuda.get_device_name(0),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        # FlashInfer's sampler JIT-compiles with the system nvcc (12.0 here),
        # which is too old for it; vLLM's own sampler is used instead. Greedy
        # decoding, so the choice of sampler does not change the output.
        "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
        "max_tokens": MAX_TOKENS,
        "max_model_len": MAX_MODEL_LEN,
        "temperature": 0.0,
        "enable_thinking": False,
        "constrained_json_schema": True,
        "chunk": CHUNK,
        "enforce_eager": True,
        "load_seconds": round(load_s, 1),
    }
    print(json.dumps(env), flush=True)
    (out / "env.json").write_text(json.dumps(env, indent=2) + "\n")

    key = json.loads((REPO / "data/llm_run_v3/_key.json").read_text())
    for cond in conds:
        tasks = json.loads((REPO / f"data/llm_run_v3/{cond}/tasks.json").read_text())
        jl = out / f"{cond}.jsonl"
        done = set()
        if jl.exists():
            done = {json.loads(line)["fig"] for line in jl.read_text().splitlines() if line.strip()}
        todo = [t for t in tasks if t["id"] not in done]
        print(f"[{cond}] total={len(tasks)} done={len(done)} todo={len(todo)}", flush=True)
        for start in range(0, len(todo), CHUNK):
            chunk = todo[start : start + CHUNK]
            messages, params = [], []
            for task in chunk:
                fig = task["id"]
                image = (REPO / key[fig]["image_path"]).read_bytes()
                url = "data:image/png;base64," + base64.b64encode(image).decode()
                messages.append(
                    [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image_url", "image_url": {"url": url}},
                                {"type": "text", "text": build_prompt(cond, task)},
                            ],
                        }
                    ]
                )
                params.append(
                    SamplingParams(
                        temperature=0.0,
                        max_tokens=MAX_TOKENS,
                        structured_outputs=StructuredOutputsParams(json=schema_for(fig)),
                    )
                )
            t1 = time.time()
            results = llm.chat(
                messages,
                params,
                chat_template_kwargs={"enable_thinking": False},
                use_tqdm=False,
            )
            per_fig = (time.time() - t1) / len(chunk)
            with jl.open("a") as f:
                for task, res in zip(chunk, results, strict=True):
                    fig = task["id"]
                    o = res.outputs[0]
                    try:
                        series, perr = lenient_parse(o.text, fig)
                    except ValueError as exc:
                        # e.g. a number with thousands of digits exceeds
                        # Python's int-parsing limit: no usable answer for
                        # this figure, not a reason to stop the run
                        series, perr = None, f"{type(exc).__name__}: {str(exc)[:200]}"
                    n_gen = len(o.token_ids)
                    rec = {
                        "fig": fig,
                        "condition": cond,
                        "error": None,
                        "paper_figure": f"{key[fig]['paper_id']}-{key[fig]['figure_id']}",
                        "image_path": key[fig]["image_path"],
                        "raw": o.text,
                        "prompt_tokens": len(res.prompt_token_ids or []),
                        "generation_tokens": n_gen,
                        "finish_reason": o.finish_reason,
                        "truncated": n_gen >= MAX_TOKENS or o.finish_reason == "length",
                        "parsed": series,
                        "parse_error": perr,
                        "seconds": round(per_fig, 2),
                    }
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_done = len(done) + start + len(chunk)
            print(
                f"[{cond} {n_done}/{len(tasks)}] chunk of {len(chunk)} in "
                f"{per_fig * len(chunk):.0f}s ({per_fig:.1f}s/fig)",
                flush=True,
            )


if __name__ == "__main__":
    # vLLM's engine process can keep the parent waiting at interpreter
    # shutdown after an exception (seen 2026-10-05: the GPU stayed held and
    # the queue stalled). Every finished record is already flushed, so exit
    # hard either way.
    import traceback

    try:
        main()
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)
