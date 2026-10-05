"""Prepare Codex CLI (OpenAI, ChatGPT sign-in) runs (design 7.81): the same
sealed-directory setup the Claude agents had, for a GPT agent that also runs
code and views images.

Figures, task contents, fig_NNN names and instructions are the Claude run's
for that condition, so the rows compare figure for figure:

  calibrated / noaxis -- the v3 run (data/llm_run_v3, llm_run_v3_prompt.md)
  pixcal              -- the pixcal run (data/llm_run_pixcal, llm_run_pixcal_prompt.md)

Only scored figures are handed out. A pilot takes `pilot<n>`: n figures drawn
with a fixed seed. `parts<k>` splits the remaining figures (those no pilot of
this model and condition has answered) into k batches part1..partk.

Writes <work>/<condition>/<model>/<batch>/{images/,tasks.json,INSTRUCTIONS.md}.
Answers are archived to data/llm_run_codex/<condition>/<model>/<batch>.*
(archive_llm_run_codex.py); the tasks and key stay the Claude run's.

Usage: python scripts/eval/prepare_llm_run_codex.py <work_dir> <model> <condition> pilot<n>|parts<k>
"""

from __future__ import annotations

import json
import pathlib
import random
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402

ARCHIVE = REPO / "data/llm_run_codex"
SEED = 20261006
SOURCES = {
    "calibrated": (REPO / "data/llm_run_v3", REPO / "scripts/eval/llm_run_v3_prompt.md"),
    "noaxis": (REPO / "data/llm_run_v3", REPO / "scripts/eval/llm_run_v3_prompt.md"),
    "pixcal": (REPO / "data/llm_run_pixcal", REPO / "scripts/eval/llm_run_pixcal_prompt.md"),
}


def instructions(directory: pathlib.Path, n: int, condition: str) -> str:
    text = SOURCES[condition][1].read_text()
    body = text.split("\n---\n")[1].strip()
    if condition != "pixcal":  # the v3 prompt keeps its condition blocks apart
        block = text.split(f"## `{{CONDITION}}`, {condition}\n")[1].split("\n## ")[0].strip()
        notes = (
            text.split("## `{AXIS_NOTES}`, noaxis only\n")[1].strip()
            if condition == "noaxis"
            else ""
        )
        body = body.replace("{CONDITION}", block).replace(
            " {AXIS_NOTES}", " " + notes if notes else ""
        )
    return body.replace("{DIR}", str(directory)).replace("{N}", str(n))


def main() -> None:
    work, model, condition, spec = pathlib.Path(sys.argv[1]).resolve(), *sys.argv[2:5]
    if REPO in work.parents or work == REPO:
        raise SystemExit("work dir must be outside the repository")
    run = SOURCES[condition][0]
    key = json.loads((run / "_key.json").read_text())
    scored = {
        p.figure_id
        for p in select_verified_pairings(load_registry(REPO / "data/verified_pairs/registry.json"))
    }
    tasks = [
        t
        for t in json.loads((run / condition / "tasks.json").read_text())
        if key[t["id"]]["figure_id"] in scored
    ]
    if spec.startswith("pilot"):
        batches = {
            spec: sorted(random.Random(SEED).sample(tasks, int(spec[5:])), key=lambda t: t["id"])
        }
    else:
        done = set()
        for p in (ARCHIVE / condition / model).glob("pilot*.predictions.json"):
            done |= set(json.loads(p.read_text()))
        rest = [t for t in tasks if t["id"] not in done]
        k = int(spec[5:])
        batches = {f"part{i + 1}": rest[i::k] for i in range(k)}
    for name, batch in batches.items():
        d = work / condition / model / name
        if d.exists():
            raise SystemExit(f"{d} exists -- refusing to overwrite a run in progress")
        (d / "images").mkdir(parents=True)
        for t in batch:
            shutil.copy(REPO / key[t["id"]]["image_path"], d / "images" / t["id"])
        (d / "tasks.json").write_text(json.dumps(batch, indent=2) + "\n")
        (d / "INSTRUCTIONS.md").write_text(instructions(d, len(batch), condition) + "\n")
        print(d, len(batch))


if __name__ == "__main__":
    main()
