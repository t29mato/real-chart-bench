"""[full-run variant of worker.py: figs arg ALL = every id in tasks.json; per-figure errors recorded]
Single-shot local VLM inference on real-chart-bench figures (pilot, design 7.69).

Usage: python worker.py <hf_repo_id> <revision> <out_dir> <fig ids comma-separated|@sel.json> [conditions]
Appends one JSON line per (condition, figure) to <out_dir>/<condition>.jsonl; skips done ones.
Reads the repo read-only.
"""
from __future__ import annotations

import json
import os
import pathlib
import platform
import re
import sys
import time
import importlib.metadata as md

REPO = pathlib.Path("/Users/matotomoya/dev/real-chart-bench")
HERE = pathlib.Path(__file__).resolve().parent
PROMPT_DIR = HERE / "prompt"
MAX_TOKENS = 8192
T_START = time.time()


def build_prompt(cond: str, task: dict) -> str:
    tmpl = (PROMPT_DIR / "single_shot_prompt.md").read_text()
    condition = (PROMPT_DIR / f"condition_{cond}.md").read_text().strip()
    return (
        tmpl.replace("{CONDITION}", condition)
        .replace("{TASK_JSON}", json.dumps(task, ensure_ascii=False))
        .replace("{ID}", task["id"])
        .strip()
    )


def schema_for(fig: str) -> dict:
    series = {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "x": {"type": "array", "items": {"type": "number"}},
            "y": {"type": "array", "items": {"type": "number"}},
        },
        "required": ["label", "x", "y"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {fig: {"type": "array", "items": series}},
        "required": [fig],
        "additionalProperties": False,
    }


def lenient_parse(text: str, fig: str):
    """First JSON object in the text; returns (series_list, error)."""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    start = t.find("{")
    if start < 0:
        return None, "no '{' in output"
    try:
        obj, _ = json.JSONDecoder().raw_decode(t[start:])
    except json.JSONDecodeError as e:
        return None, f"JSONDecodeError: {e}"
    if isinstance(obj, dict) and fig in obj and isinstance(obj[fig], list):
        return obj[fig], None
    if isinstance(obj, dict) and len(obj) == 1 and isinstance(next(iter(obj.values())), list):
        return next(iter(obj.values())), "key mismatch (accepted single key)"
    return None, f"unexpected shape: keys={list(obj)[:5] if isinstance(obj, dict) else type(obj).__name__}"


def main() -> None:
    repo_id, revision, out_dir, figs_arg = sys.argv[1:5]
    conds = sys.argv[5].split(",") if len(sys.argv) > 5 else ["calibrated", "noaxis"]
    constrained = os.environ.get("CONSTRAINED", "1") == "1"
    figs = json.loads((HERE / figs_arg[1:]).read_text()) if figs_arg.startswith("@") else figs_arg.split(",")
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_vlm import generate, load
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.structured import build_json_schema_logits_processor

    path = snapshot_download(repo_id, revision=revision)  # offline: resolves from cache
    t0 = time.time()
    model, processor = load(path)
    load_s = time.time() - t0
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    ip = getattr(processor, "image_processor", None)
    env = {
        "repo_id": repo_id, "revision": revision, "local_path": path,
        "quantization": model.config.__dict__.get("quantization") if hasattr(model, "config") else None,
        "mlx_vlm": md.version("mlx-vlm"), "mlx": md.version("mlx"),
        "transformers": md.version("transformers"), "python": sys.version.split()[0],
        "platform": platform.platform(),
        "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
        "image_processor_size": getattr(ip, "size", None) if ip is not None else None,
        "max_tokens": MAX_TOKENS, "temperature": 0.0, "enable_thinking": False,
        "constrained_json_schema": constrained, "load_seconds": round(load_s, 1),
    }
    print(json.dumps(env, default=str), flush=True)
    (out / "env.json").write_text(json.dumps(env, indent=2, default=str) + "\n")

    key = json.loads((REPO / "data/llm_run_v2/_key.json").read_text())
    for cond in conds:
        tasks = {t["id"]: t for t in json.loads((REPO / f"data/llm_run_v2/{cond}/tasks.json").read_text())}
        jl = out / f"{cond}.jsonl"
        done = set()
        if jl.exists():
            done = {json.loads(l)["fig"] for l in jl.read_text().splitlines() if l.strip()}
        (out / "prompts").mkdir(exist_ok=True)
        cond_figs = list(tasks) if figs == ["ALL"] else figs
        todo = [f for f in cond_figs if f not in done]
        print(f"[{cond}] total={len(cond_figs)} done={len(cond_figs)-len(todo)} todo={len(todo)}", flush=True)
        for i, fig in enumerate(todo, 1):
            t1 = time.time()
            rec = {"fig": fig, "condition": cond, "error": None}
            try:
                task = tasks[fig]
                rec |= {"paper_figure": f"{key[fig]['paper_id']}-{key[fig]['figure_id']}",
                        "image_path": key[fig]["image_path"]}
                prompt_text = build_prompt(cond, task)
                (out / "prompts" / f"{cond}_{fig}.txt").write_text(prompt_text + "\n")
                image_path = str(REPO / key[fig]["image_path"])
                formatted = apply_chat_template(processor, model.config, prompt_text, num_images=1,
                                                enable_thinking=False)
                kw = dict(max_tokens=MAX_TOKENS, temperature=0.0, enable_thinking=False)
                if constrained:
                    kw["logits_processors"] = [build_json_schema_logits_processor(tokenizer, schema_for(fig))]
                mx.reset_peak_memory()
                t1 = time.time()
                res = generate(model, processor, formatted, image=[image_path], verbose=False, **kw)
                series, perr = lenient_parse(res.text, fig)
                rec |= {
                    "raw": res.text, "prompt_tokens": res.prompt_tokens,
                    "generation_tokens": res.generation_tokens,
                    "prompt_tps": res.prompt_tps, "generation_tps": res.generation_tps,
                    "peak_memory_gb": res.peak_memory, "finish_reason": res.finish_reason,
                    "truncated": res.generation_tokens >= MAX_TOKENS or res.finish_reason == "length",
                    "parsed": series, "parse_error": perr,
                }
            except Exception as e:  # record, keep going
                rec["error"] = f"{type(e).__name__}: {e}"
                rec.setdefault("parsed", None)
            secs = time.time() - t1
            rec["seconds"] = round(secs, 2)
            with jl.open("a") as f:
                f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
            print(f"[{cond} {i}/{len(todo)}] {fig} {secs:.1f}s gen={rec.get('generation_tokens')} "
                  f"prompt={rec.get('prompt_tokens')} finish={rec.get('finish_reason')} "
                  f"parse_err={rec.get('parse_error')} err={rec['error']} "
                  f"nseries={len(rec['parsed']) if rec.get('parsed') else 0} "
                  f"elapsed={time.time()-T_START:.0f}s", flush=True)


if __name__ == "__main__":
    main()
