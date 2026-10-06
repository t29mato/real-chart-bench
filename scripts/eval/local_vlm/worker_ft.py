"""Approach B inference (docs/design/local-model.md, 方式B): Qwen3.5-9B bf16 +
a QLoRA adapter from scripts/train/vlm_lora/train.py, served by vLLM's LoRA
support (no merged copy of the weights on disk).

Everything else is worker_cuda.py's bf16 run: the v3 single-shot prompt,
JSON-schema-constrained greedy decoding, max_tokens 8192, chunks of 16, the
same records (seconds = chunk wall time / chunk size, prompt and generated
tokens per figure), so score_llm_predictions.py scores it like the
local-cuda rows. One deliberate difference: the image is capped at the
training resolution (train.cap_image, MAX_PIXELS); with no adapter this
worker gives the base model under the same cap, the control row.

Modes:
  bench <adapter|none> <out_dir> [conditions]   real-chart-bench figures;
        noaxis tasks from data/llm_run_v3, pixcal from data/llm_run_pixcal
  val <adapter|none> <out_dir> <run_dir> [n] [source]   the run's held-out
        training examples (val_keys.json; optionally of one source only), for
        tuning without touching the benchmark
"""

from __future__ import annotations

import base64
import importlib.metadata as md
import io
import json
import os
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "scripts/train/vlm_lora"))

from worker_v3 import build_prompt, lenient_parse, schema_for  # noqa: E402

MAX_TOKENS = 8192
MAX_MODEL_LEN = 16384
CHUNK = 16
RUN_DIRS = {"noaxis": "data/llm_run_v3", "calibrated": "data/llm_run_v3",
            "pixcal": "data/llm_run_pixcal"}


def image_url(path: pathlib.Path) -> str:
    from PIL import Image
    from train import cap_image

    buf = io.BytesIO()
    cap_image(Image.open(path)).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def bench_jobs(cond: str):
    run_dir = REPO / RUN_DIRS[cond]
    key = json.loads((run_dir / "_key.json").read_text())
    for task in json.loads((run_dir / cond / "tasks.json").read_text()):
        fig = task["id"]
        yield fig, REPO / key[fig]["image_path"], build_prompt(cond, task), {
            "paper_figure": f"{key[fig]['paper_id']}-{key[fig]['figure_id']}",
            "image_path": key[fig]["image_path"],
        }


def val_jobs(run_dir: pathlib.Path, n: int, only: str | None = None):
    from examples import load_examples

    cfg = json.loads((run_dir / "config.json").read_text())
    keys = set(json.loads((run_dir / "val_keys.json").read_text()))
    ex, _ = load_examples([pathlib.Path(d) for d in cfg["data"]],
                          pixcal_fraction=cfg["pixcal_fraction"],
                          val_fraction=cfg["val_fraction"])
    ex = sorted((e for e in ex if e["key"] in keys and only in (None, e["source"])),
                key=lambda e: e["key"])[:n]
    for e in ex:
        # fig = the example key, so records are unique; the prompt's id stays fig_NNN
        yield e["key"], pathlib.Path(e["image"]), e["prompt"], {
            "prompt_id": e["fig_id"], "condition_of_example": e["condition"],
            "answer": e["answer"][e["fig_id"]], "source": e["source"],
        }


def run(llm, lora, jobs, jl: pathlib.Path, cond: str):
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    done = set()
    if jl.exists():
        done = {json.loads(x)["fig"] for x in jl.read_text().splitlines() if x.strip()}
    todo = [j for j in jobs if j[0] not in done]
    print(f"[{cond}] done={len(done)} todo={len(todo)}", flush=True)
    for start in range(0, len(todo), CHUNK):
        chunk = todo[start : start + CHUNK]
        messages, params = [], []
        for fig, img, prompt, meta in chunk:
            pid = meta.get("prompt_id", fig)
            messages.append([{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url(img)}},
                {"type": "text", "text": prompt}]}])
            params.append(SamplingParams(
                temperature=0.0, max_tokens=MAX_TOKENS,
                structured_outputs=StructuredOutputsParams(json=schema_for(pid))))
        t1 = time.time()
        results = llm.chat(messages, params, chat_template_kwargs={"enable_thinking": False},
                           use_tqdm=False, lora_request=lora)
        per_fig = (time.time() - t1) / len(chunk)
        with jl.open("a") as f:
            for (fig, _img, _prompt, meta), res in zip(chunk, results, strict=True):
                o = res.outputs[0]
                try:
                    series, perr = lenient_parse(o.text, meta.get("prompt_id", fig))
                except ValueError as exc:
                    series, perr = None, f"{type(exc).__name__}: {str(exc)[:200]}"
                n_gen = len(o.token_ids)
                rec = {"fig": fig, "condition": cond, "error": None, **meta, "raw": o.text,
                       "prompt_tokens": len(res.prompt_token_ids or []),
                       "generation_tokens": n_gen, "finish_reason": o.finish_reason,
                       "truncated": n_gen >= MAX_TOKENS or o.finish_reason == "length",
                       "parsed": series, "parse_error": perr, "seconds": round(per_fig, 2)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[{cond}] {len(done) + start + len(chunk)} figs, {per_fig:.1f}s/fig", flush=True)


def main() -> None:
    mode, adapter, out_dir = sys.argv[1:4]
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    from train import MAX_PIXELS
    from vllm import LLM
    from vllm.lora.request import LoRARequest

    path = snapshot_download("Qwen/Qwen3.5-9B")
    rank = None
    if adapter != "none":
        rank = json.loads((pathlib.Path(adapter) / "adapter_config.json").read_text())["r"]
    t0 = time.time()
    llm = LLM(model=path, dtype="bfloat16", max_model_len=MAX_MODEL_LEN,
              gpu_memory_utilization=0.92, limit_mm_per_prompt={"image": 1}, seed=0,
              enforce_eager=True,
              **({"enable_lora": True, "max_lora_rank": rank, "max_loras": 1} if rank else {}))
    lora = LoRARequest("ft", 1, str(pathlib.Path(adapter).resolve())) if rank else None
    import torch

    env = {"repo_id": "Qwen/Qwen3.5-9B", "revision": pathlib.Path(path).name,
           "precision": "bf16", "adapter": adapter, "lora_rank": rank,
           "adapter_train_state": (json.loads((pathlib.Path(adapter).parent / "state.json")
                                              .read_text()) if rank else None),
           "max_pixels": MAX_PIXELS, "vllm": md.version("vllm"), "torch": torch.__version__,
           "transformers": md.version("transformers"), "gpu": torch.cuda.get_device_name(0),
           "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
           "VLLM_USE_FLASHINFER_SAMPLER": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
           "max_tokens": MAX_TOKENS, "max_model_len": MAX_MODEL_LEN, "temperature": 0.0,
           "enable_thinking": False, "constrained_json_schema": True, "chunk": CHUNK,
           "enforce_eager": True, "load_seconds": round(time.time() - t0, 1), "mode": mode}
    print(json.dumps(env), flush=True)
    (out / "env.json").write_text(json.dumps(env, indent=2) + "\n")
    if mode == "bench":
        conds = sys.argv[4].split(",") if len(sys.argv) > 4 else ["noaxis", "pixcal"]
        for cond in conds:
            run(llm, lora, list(bench_jobs(cond)), out / f"{cond}.jsonl", cond)
    elif mode == "val":
        run_dir = pathlib.Path(sys.argv[4])
        n = int(sys.argv[5]) if len(sys.argv) > 5 else 200
        only = sys.argv[6] if len(sys.argv) > 6 else None
        name = f"val-{only}.jsonl" if only else "val.jsonl"
        run(llm, lora, list(val_jobs(run_dir, n, only)), out / name, "val")
    else:
        raise SystemExit(f"unknown mode {mode}")


if __name__ == "__main__":
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
