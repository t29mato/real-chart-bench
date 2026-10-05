# Local VLM runs v2 and v3 (design §7.69, §7.73 (2)/(3))

Open-weight vision-language models run on a laptop, one inference per figure.

- **v3 (current, 2026-10-04〜05):** the same 97 tasks as the Claude LLM run v3
  (`data/llm_run_v3/`), on a single-shot version of the markers-only v3 prompt
  (`prompt_v3/`). The three models ran **one at a time** (`run_v3_chain.sh`:
  Qwen3.5-9B → Qwen3.8-27B → Gemma 4 31B), so `seconds` per figure is
  comparable between models. Raw outputs:
  `data/local_vlm_run_v3/<model_id>/{calibrated,noaxis}.jsonl`, `env.json`, and
  the chain's timestamped log `data/local_vlm_run_v3/v3_chain.log`. Scoring:
  `scripts/eval/score_llm_predictions.py local-v3-calibrated` and
  `local-v3-noaxis` → `results/<model_id>-v0-local-v3[-noaxis].json` (tasks and
  key from `data/llm_run_v3/`, no unit rescale; `local_run.seconds_per_figure`
  holds n / mean / median / max / total).
- **v2 (history, 2026-10-03〜04):** the 101 tasks of the Claude LLM run v2
  (`data/llm_run_v2/`), on a single-shot version of the v2 prompt (`prompt/`).
  The three models ran **concurrently**, so their `seconds` are not comparable.
  Raw outputs: `data/local_vlm_run_v2/<model_id>/{calibrated,noaxis}.jsonl` and
  `env.json`. Scoring: `local-v2-calibrated` / `local-v2-noaxis` →
  `results/<model_id>-v0-local-v2[-noaxis].json`.

The files here are copied as they ran, not tidied, so they are the record of
what produced the archived outputs.

Seconds per figure, v3 (mean / median / max over the 97 figures, failures
included):

| model | axis given | axis withheld | wall clock (both conditions) |
|---|---|---|---|
| Qwen3.5-9B | 29.7 / 15.3 / 233.3 s | 26.6 / 14.9 / 225.0 s | 1 h 31 min |
| Qwen3.8-27B | 69.0 / 47.2 / 670.8 s | 68.2 / 47.4 / 689.3 s | 3 h 42 min |
| Gemma 4 31B | 116.4 / 72.3 / 814.3 s | 113.1 / 63.9 / 812.8 s | 6 h 11 min |

The long tails are the outputs that ran into the 8192-token limit.

## Setup

| | |
|---|---|
| Hardware | Apple M3 Max, 128GB unified memory (laptop), macOS 14.5 |
| Software | Python 3.12.6, mlx-vlm 0.7.4, mlx 0.32.3, transformers 5.18.0 (separate venv, not the repo's) |
| Decoding | greedy (temperature 0), `enable_thinking=False`, JSON-schema constrained decoding (`mlx_vlm.structured.build_json_schema_logits_processor`, schema `{"<task id>": [{"label": str, "x": [number], "y": [number]}]}`), `max_tokens` 8192 |
| Network | `HF_HUB_OFFLINE=1` (weights resolved from the local cache; no network during inference) |
| Image | the original file, not re-encoded; model's default processor resolution, except Gemma (below) |
| Concurrency | v3: one model at a time (`seconds` comparable). v2: the three models ran at the same time, so `seconds` per figure is not comparable between models or with a solo run |

| model_id | HF repo | revision |
|---|---|---|
| `qwen3.5-9b-8bit` | `mlx-community/Qwen3.5-9B-8bit` | `16daa4818c54ce5f5436f929d52542eb65bbed9d` |
| `qwen3.8-27b-8bit` | `mlx-community/Qwen3.8-27B-8bit` | `815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9` |
| `gemma-4-31b-8bit` | `mlx-community/gemma-4-31b-it-8bit` | `f5f3dc92ab4af76724c36c21eb6bedadb3a851be` |

All three are Apache-2.0 (design §7.69). Gemma 4: the image token budget is
raised from the default 280 to 1120 by setting
`processor.image_processor.max_soft_tokens` (`gemma/worker_gemma.py`,
`MAX_SOFT_TOKENS`); `env.json` records both values.

## Files

v3:

- `prompt_v3/single_shot_prompt.md` + `prompt_v3/condition_{calibrated,noaxis}.md` —
  the v3 prompt: the same single-shot adaptation as `prompt/`, applied to
  `scripts/eval/llm_run_v3_prompt.md`. `prompt_v3/NOTES.md` lists the changes,
  `prompt_v3/diff_vs_v3.diff` is the diff against `prompt_v3/original_v3_body.md`,
  `prompt_v3/single_shot_v2_to_v3.diff` the diff against the v2 single-shot prompt
  (the markers-only rule is the only change).
- `worker_v3.py` (Qwen3.5-9B and Qwen3.8-27B) and `gemma/worker_gemma_v3.py` —
  `worker_full.py` / `gemma/worker_gemma.py` with only the prompt directory
  (`prompt_v3/`) and the task / key files (`data/llm_run_v3/`) changed.
- `run_v3_chain.sh` — runs the three models one after another, both conditions
  each, offline; resumable (a worker skips figures already in its jsonl).

v2:

- `prompt/single_shot_prompt.md` + `prompt/condition_{calibrated,noaxis}.md` —
  the prompt. It is `scripts/eval/llm_run_v2_prompt.md` with the agent-only
  parts removed (work directory, "you may write Python", predictions file);
  `prompt/NOTES.md` lists every change, `prompt/diff_vs_v2.diff` is the diff
  against `prompt/original_v2_body.md`. `{TASK_JSON}` is the task's entry from
  `data/llm_run_v2/<condition>/tasks.json`, verbatim.
- `worker_full.py` — Qwen3.5-9B full run (`run_9b_full.sh`).
- `worker.py` — the pilot worker. The Qwen3.8-27B full run was made with this
  one, given every task id (its jsonl key order and log format are this
  worker's); the generation code is the same as `worker_full.py`, which only
  adds `ALL` and per-figure exception capture around the whole step.
- `gemma/worker_gemma.py` — Gemma 4 variant (image budget); `gemma/run_all.sh`
  (pilot on `sel.json`, then the full run), `gemma/download.sh`.

Each jsonl line: `fig` (task id), `condition`, `paper_figure`, `raw` (generated
text), `parsed` (series list, or null), `parse_error`, `error` (exception),
`finish_reason`, `truncated`, `prompt_tokens`, `generation_tokens`, tps,
`peak_memory_gb`, `seconds` (Gemma also `image_soft_tokens`).

Failures are scored as total misses (no answer), as for the LLMs: outputs cut
off at 8192 tokens that did not parse, and, in v2 only, for Qwen3.5-9B noaxis
fig_006 / fig_010, a Python error while parsing (the model emitted an integer
over 4300 digits; the raw text was not saved for those two). In v3 every
failure is a cut-off output (1–5 per model and condition, mostly dense-marker
figures). Each results file lists them under `local_run`.

## Rerun

The scripts hard-code the repository path (`REPO = /Users/matotomoya/dev/real-chart-bench`)
and read `prompt/` next to themselves; change `REPO` for another clone.

```bash
python3.12 -m venv .venv && .venv/bin/pip install mlx-vlm==0.7.4 mlx==0.32.3 transformers==5.18.0
export HF_HOME=$PWD/hf_home
# once, online: fetch the pinned revision
.venv/bin/python -c "from huggingface_hub import snapshot_download as s; s('mlx-community/Qwen3.8-27B-8bit', revision='815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9')"
# then offline; appends to <out>/<condition>.jsonl and skips figures already done
HF_HUB_OFFLINE=1 .venv/bin/python worker_full.py mlx-community/Qwen3.8-27B-8bit \
    815b83c0df8ffd1d1b5244cf75fd6ef14fca9ef9 runs/qwen3.8-27b-8bit ALL calibrated,noaxis
# Gemma: gemma/worker_gemma.py, same arguments (MAX_SOFT_TOKENS=1120 is the default)
```

Copy `runs/<model>/{calibrated,noaxis}.jsonl` and `env.json` to
`data/local_vlm_run_v2/<model_id>/` and run the two scorer conditions.

v3: `run_v3_chain.sh` (paths at its top), or a worker by hand —
`HF_HUB_OFFLINE=1 .venv/bin/python worker_v3.py <repo> <revision> runs/<model>-v3 ALL calibrated,noaxis`
(`gemma/worker_gemma_v3.py` for Gemma). Copy `runs/<model>-v3/{calibrated,noaxis}.jsonl`
and `env.json` to `data/local_vlm_run_v3/<model_id>/` and run `local-v3-calibrated` /
`local-v3-noaxis`.
