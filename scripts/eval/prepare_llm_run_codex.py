"""Prepare a Codex CLI (OpenAI, ChatGPT sign-in) run on the v3 tasks
(design 7.81): the same sealed-directory setup the Claude agents had, for a
GPT agent that also runs code and views images.

The figures are the v3 run's (same fig_NNN names, same task contents, key
data/llm_run_v3/_key.json); a pilot takes `n` of them, drawn with a fixed
seed from the scored set. Instructions are llm_run_v3_prompt.md with the
condition block filled in, written as INSTRUCTIONS.md inside the directory.
The selected tasks and their key are archived to
data/llm_run_codex/pilot<n>/{_key.json,<condition>/tasks.json} -- the layout
score_llm_predictions.py reads.

Usage: python scripts/eval/prepare_llm_run_codex.py <work_dir> <model> <condition> <n>
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

V3 = REPO / "data/llm_run_v3"
ARCHIVE = REPO / "data/llm_run_codex"
PROMPT = REPO / "scripts/eval/llm_run_v3_prompt.md"
SEED = 20261006


def instructions(directory: pathlib.Path, n: int, condition: str) -> str:
    text = PROMPT.read_text()
    body = text.split("\n---\n")[1].strip()
    block = text.split(f"## `{{CONDITION}}`, {condition}\n")[1].split("\n## ")[0].strip()
    notes = (
        text.split("## `{AXIS_NOTES}`, noaxis only\n")[1].strip() if condition == "noaxis" else ""
    )
    body = body.replace("{CONDITION}", block).replace(" {AXIS_NOTES}", " " + notes if notes else "")
    return body.replace("{DIR}", str(directory)).replace("{N}", str(n))


def main() -> None:
    work, model, condition, n = (
        pathlib.Path(sys.argv[1]).resolve(),
        sys.argv[2],
        sys.argv[3],
        int(sys.argv[4]),
    )
    if REPO in work.parents or work == REPO:
        raise SystemExit("work dir must be outside the repository")
    d = work / condition / model / f"pilot{n}"
    if d.exists():
        raise SystemExit(f"{d} exists -- refusing to overwrite a run in progress")
    key = json.loads((V3 / "_key.json").read_text())
    scored = {
        p.figure_id
        for p in select_verified_pairings(load_registry(REPO / "data/verified_pairs/registry.json"))
    }
    tasks = [
        t
        for t in json.loads((V3 / condition / "tasks.json").read_text())
        if key[t["id"]]["figure_id"] in scored
    ]
    batch = sorted(random.Random(SEED).sample(tasks, n), key=lambda t: t["id"])
    (d / "images").mkdir(parents=True)
    for t in batch:
        shutil.copy(REPO / key[t["id"]]["image_path"], d / "images" / t["id"])
    (d / "tasks.json").write_text(json.dumps(batch, indent=2) + "\n")
    (d / "INSTRUCTIONS.md").write_text(instructions(d, n, condition) + "\n")
    run = ARCHIVE / f"pilot{n}"
    (run / condition).mkdir(parents=True, exist_ok=True)
    sub_key = {t["id"]: key[t["id"]] for t in batch}
    if (run / "_key.json").exists():
        sub_key = json.loads((run / "_key.json").read_text()) | sub_key
    (run / "_key.json").write_text(json.dumps(sub_key, indent=2) + "\n")
    (run / condition / "tasks.json").write_text(json.dumps(batch, indent=2) + "\n")
    print(d, [t["id"] for t in batch])


if __name__ == "__main__":
    main()
