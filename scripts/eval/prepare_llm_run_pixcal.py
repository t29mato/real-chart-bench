"""Prepare the 'tick pixel positions given' run (design 7.76/7.77): the
person-calibrates-the-axes, AI-extracts-the-points condition.

Each task hands over what a person enters in WebPlotDigitizer: for each axis,
two tick marks with their pixel position in the image file and their value
(data/verified_pairs/tick_calibration.json, values in the ground truth's
space), the axis scale, and the image's pixel size. The model only has to
find the markers. Everything else is the v3 run's: the v3 prompt with a new
condition block (scripts/eval/llm_run_pixcal_prompt.md), the same four models,
two batches, sealed per-(model, batch) directories, a new seed.

Lessons applied from the v3 run (design 7.74): the key is NOT written into the
working tree (only to data/llm_run_pixcal/), and the instructions sit in each
sealed directory as INSTRUCTIONS.md.

Usage: python scripts/eval/prepare_llm_run_pixcal.py <work_dir>
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

MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-haiku-4-5"]
N_BATCHES = 2
SEED = 20261006
ARCHIVE = REPO / "data/llm_run_pixcal"
PROMPT = REPO / "scripts/eval/llm_run_pixcal_prompt.md"


def instructions(directory: pathlib.Path, n: int) -> str:
    text = PROMPT.read_text().split("\n---\n", 2)[1].strip()
    return text.replace("{DIR}", str(directory)).replace("{N}", str(n))


def main() -> None:
    work = pathlib.Path(sys.argv[1]).resolve()
    if REPO in work.parents or work == REPO:
        raise SystemExit("work dir must be outside the repository")
    if work.exists():
        raise SystemExit(f"{work} exists -- refusing to overwrite a run in progress")
    pairings = select_verified_pairings(load_registry(REPO / "data/verified_pairs/registry.json"))
    cal = {
        (c["paper_id"], c["figure_id"]): c
        for c in json.loads((REPO / "data/verified_pairs/tick_calibration.json").read_text())[
            "figures"
        ]
    }
    missing = [p for p in pairings if (p.paper_id, p.figure_id) not in cal]
    if missing:
        raise SystemExit(f"{len(missing)} scored figures have no tick calibration")
    random.Random(SEED).shuffle(pairings)

    from PIL import Image

    key, tasks = {}, []
    for i, p in enumerate(pairings, 1):
        name = f"fig_{i:03d}.png"
        c = cal[(p.paper_id, p.figure_id)]
        w, h = Image.open(REPO / p.image_path).size
        key[name] = {"paper_id": p.paper_id, "figure_id": p.figure_id, "image_path": p.image_path}
        tasks.append(
            {
                "id": name,
                "image_size": [w, h],
                "x_scale": c["x_scale"],
                "y_scale": c["y_scale"],
                "x_ticks": [{"pixel_x": t["px"], "value": t["value"]} for t in c["x"]],
                "y_ticks": [{"pixel_y": t["px"], "value": t["value"]} for t in c["y"]],
            }
        )

    for model in MODELS:
        for k in range(N_BATCHES):
            d = work / "pixcal" / model / f"part{k + 1}"
            (d / "images").mkdir(parents=True)
            batch = tasks[k::N_BATCHES]
            for t in batch:
                shutil.copy(REPO / key[t["id"]]["image_path"], d / "images" / t["id"])
            (d / "tasks.json").write_text(json.dumps(batch, indent=2) + "\n")
            (d / "INSTRUCTIONS.md").write_text(instructions(d, len(batch)) + "\n")

    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "_key.json").write_text(json.dumps(key, indent=2) + "\n")
    (ARCHIVE / "pixcal").mkdir(exist_ok=True)
    (ARCHIVE / "pixcal" / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
    print(f"{len(tasks)} figures x {len(MODELS)} models x {N_BATCHES} batches -> {work}")


if __name__ == "__main__":
    main()
