"""Prepare the 2026-10-01 LLM run: every scoreable figure, two conditions,
the four models a Claude Code subagent can be launched as today.

Why a new run instead of extending the old ones: the earlier runs used
Opus 5 / Sonnet 5 / Fable 5 / Haiku 4.5 and covered the axis-withheld
("noaxis") condition on 10 figures only. The subagents available now are
Opus 5.5 / Sonnet 5.5 / Fable 5.1 / Haiku 4.5. Extending noaxis with newer
models would confound "axis withheld" with "different model", so both
conditions are run again, same models, same protocol, same figures:

  calibrated -- image + x_range/y_range/x_scale/y_scale (= ExtractionTask)
  noaxis     -- image + one reporting rule per axis; no ranges, no scales.
                The model reads the printed ticks itself.

The reporting rule replaces the unit strings the 2026-09-08 run handed over.
Those strings come from ground_truth.json and are not reliable after the
display-unit migration (28500 says "x10^4 S/m" but stores 50000-70000 as
printed). Comparing the registry's ranges with the owner-reviewed printed
tick labels (axis_pixel_candidates.json) shows the stored values are in the
printed tick-label space for every axis except two kinds, which get their
own rule:
  - y axes whose tick labels are log10 values ("log sigma"): stored 10^tick
  - x axes printed in degC whose ground truth is in K: stored degC + 273.15
Everything else: "report the numbers as printed on the tick labels".

Contamination guards (design 7.62/7.63 lessons):
  - images are copied outside the repository as fig_NNN.png in a seeded
    random order, so neither name nor order points back at a paper;
  - one sealed directory per (condition, model, batch) -- no agent can see
    another's answers;
  - the prompt (scripts/eval/llm_run_v2_prompt.md) says not to search for
    ground truth. Mitigation, not a sandbox.

Writes <work>/<condition>/<model>/part<k>/{images/,tasks.json} and
<work>/_key.json (fig name -> paper_id/figure_id), and copies the key into
data/llm_run_v2/ so scoring never depends on the scratch directory.

Usage: python scripts/eval/prepare_llm_run_v2.py <work_dir>
"""

from __future__ import annotations

import json
import pathlib
import random
import re
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402

MODELS = ["claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1", "claude-haiku-4-5"]
CONDITIONS = ["calibrated", "noaxis"]
N_BATCHES = 2
SEED = 20261001
ARCHIVE = REPO / "data/llm_run_v2"


AS_PRINTED = (
    "Report numbers on the same scale as the printed tick labels. Do not apply "
    "a multiplier written in the axis title (e.g. '(10^4 S/m)' or 'x10^4'); if "
    "a tick label itself is written like 5x10^4, report 50000."
)
CELSIUS = re.compile(r"°\s?C|degC|\(C\)|℃")
LOG_LABEL = re.compile(r"\s*(lg|log)", re.I)


def main() -> None:
    work = pathlib.Path(sys.argv[1]).resolve()
    if REPO in work.parents or work == REPO:
        raise SystemExit("work dir must be outside the repository")
    if work.exists():
        raise SystemExit(f"{work} exists -- refusing to overwrite a run in progress")

    pairings = select_verified_pairings(load_registry(REPO / "data/verified_pairs/registry.json"))
    gt = json.loads((REPO / "data/verified_pairs/ground_truth.json").read_text())
    axes = {
        (e["paper_id"], e["figure_id"]): e
        for e in json.loads((REPO / "data/verified_pairs/axis_pixel_candidates.json").read_text())
        if "paper_id" in e
    }
    random.Random(SEED).shuffle(pairings)

    key, tasks = {}, {"calibrated": [], "noaxis": []}
    for i, p in enumerate(pairings, 1):
        name = f"fig_{i:03d}.png"
        key[name] = {
            "paper_id": p.paper_id,
            "figure_id": p.figure_id,
            "image_path": p.image_path,
        }
        tasks["calibrated"].append(
            {
                "id": name,
                "x_range": list(p.x_range),
                "y_range": list(p.y_range),
                "x_scale": p.x_scale.value,
                "y_scale": p.y_scale.value,
            }
        )
        curves = [c for c in gt[p.figure_id] if c.get("x")]
        a = axes.get((p.paper_id, p.figure_id), {})
        x_label, y_label = a.get("x_axis_label_raw") or "", a.get("y_axis_label_raw") or ""
        x_rule, y_rule = AS_PRINTED, AS_PRINTED
        if curves[0].get("unit_x") == "K" and CELSIUS.search(x_label):
            x_rule = "The axis is printed in degrees Celsius; report kelvin (degC + 273.15)."
        if p.y_scale.value == "log" and LOG_LABEL.match(y_label):
            y_rule = (
                "The tick labels are base-10 logarithms of the quantity; report the "
                "quantity itself, i.e. 10^(value read off the axis)."
            )
        tasks["noaxis"].append({"id": name, "x_report": x_rule, "y_report": y_rule})

    for cond in CONDITIONS:
        for model in MODELS:
            for k in range(N_BATCHES):
                d = work / cond / model / f"part{k + 1}"
                (d / "images").mkdir(parents=True)
                batch = tasks[cond][k::N_BATCHES]
                for t in batch:
                    src = REPO / key[t["id"]]["image_path"]
                    shutil.copy(src, d / "images" / t["id"])
                batch_json = json.dumps(batch, ensure_ascii=False, indent=2)
                (d / "tasks.json").write_text(batch_json + "\n")

    (work / "_key.json").write_text(json.dumps(key, indent=2) + "\n")
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    (ARCHIVE / "_key.json").write_text(json.dumps(key, indent=2) + "\n")
    for cond in CONDITIONS:
        (ARCHIVE / cond).mkdir(exist_ok=True)
        (ARCHIVE / cond / "tasks.json").write_text(
            json.dumps(tasks[cond], ensure_ascii=False, indent=2) + "\n"
        )
    print(
        f"{len(pairings)} figures x {len(CONDITIONS)} conditions x {len(MODELS)} models "
        f"x {N_BATCHES} batches -> {work}"
    )


if __name__ == "__main__":
    main()
