"""Prepare the Claude run on PlotQA dot_line 100 (design 7.80): do the agents
that read real figures well also beat the chart-to-table models on the
synthetic charts those models were trained for?

Axis condition: noaxis only -- the chart-to-table rows read the axes
themselves, so that is the row the comparison needs. Tasks, key and report
rules are the synthetic run's own (data/synthetic/plotqa_dot_line/run/, the
same ones the local VLM answered). Prompt: the v3 prompt with the noaxis block,
unchanged except that the images are not from research papers (see
llm_run_synthetic_prompt.md). Four models, two batches, sealed per-(model,
batch) directories outside the repository, INSTRUCTIONS.md inside each.

Usage: python scripts/eval/prepare_llm_run_synthetic.py <work_dir>
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RUN = REPO / "data/synthetic/plotqa_dot_line/run"
PROMPT = REPO / "scripts/eval/llm_run_synthetic_prompt.md"
MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-haiku-4-5"]
N_BATCHES = 2
CONDITION = "noaxis"


def instructions(directory: pathlib.Path, n: int) -> str:
    text = PROMPT.read_text().split("\n---\n", 2)[1].strip()
    return text.replace("{DIR}", str(directory)).replace("{N}", str(n))


def main() -> None:
    work = pathlib.Path(sys.argv[1]).resolve()
    if REPO in work.parents or work == REPO:
        raise SystemExit("work dir must be outside the repository")
    if work.exists():
        raise SystemExit(f"{work} exists -- refusing to overwrite a run in progress")
    key = json.loads((RUN / "_key.json").read_text())
    tasks = json.loads((RUN / CONDITION / "tasks.json").read_text())
    for model in MODELS:
        for k in range(N_BATCHES):
            d = work / CONDITION / model / f"part{k + 1}"
            (d / "images").mkdir(parents=True)
            batch = tasks[k::N_BATCHES]
            for t in batch:
                shutil.copy(REPO / key[t["id"]]["image_path"], d / "images" / t["id"])
            (d / "tasks.json").write_text(json.dumps(batch, indent=2) + "\n")
            (d / "INSTRUCTIONS.md").write_text(instructions(d, len(batch)) + "\n")
    print(f"{len(tasks)} figures x {len(MODELS)} models x {N_BATCHES} batches -> {work}")


if __name__ == "__main__":
    main()
