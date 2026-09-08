"""Export the same 10 figures with the axis calibration withheld.

The calibrated run scored 0.87-0.98, which reads as "LLMs have solved this"
until you notice what the task hands over. `ExtractionTask` gives every model
the axis extent -- x_range and y_range in data units -- so the model is told
"this frame spans 300-900 K and 0-1.0" and asked only to locate the markers
inside it. Reading the axis is the part that actually breaks in practice: we
spent 2026-09-07 discovering that our *own* recorded axis readings were wrong
on several figures, including a stored 0.68 for a tick printed 0.70.

So this is the same ten figures, same images, same ground truth and same
metric, with one thing removed: no x_range, no y_range, no scale hints. The
model must read the printed tick labels itself and report values in the
figure's own units. That is the task a digitiser actually faces, and the gap
between the two conditions is the honest answer to "how much of this is solved".

Everything else is held constant deliberately -- same subset, same prompt
shape, same scorer -- so the difference between the runs is attributable to the
calibration and nothing else.

The one thing still supplied is the *unit* each axis should be reported in,
taken from the ground truth's own unit strings. Withholding it would measure
the wrong thing: 5904's ground truth is in V/K while its axis is printed in
uV/K, so a model that read the axis perfectly and reported what it saw would
score zero on a factor of a million. That is a units convention, not a reading
failure. The range -- the part a model must derive from the tick labels -- is
still withheld.

Each model gets its own directory this time. Sharing one directory in the first
run let the later models read the earlier ones' answers; they did not, but the
setup should not have allowed it.
"""

from __future__ import annotations

import json
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parents[2]
SRC = pathlib.Path(
    "/tmp/claude-1000/-home-mato-repos-real-chart-bench/"
    "628beb76-383b-42e9-bf21-8d4188daf8dc/scratchpad/llm_eval"
)
OUT = SRC.parent / "llm_eval_noaxis"

MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-haiku-4-5"]


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)

    tasks = json.loads((SRC / "tasks.json").read_text())
    key = json.loads((SRC / "_key.json").read_text())
    gt = json.loads((REPO / "data/verified_pairs/ground_truth.json").read_text())

    stripped = []
    for t in tasks:
        curves = [c for c in gt[key[t["id"]]["figure_id"]] if c.get("x")]
        stripped.append({
            "id": t["id"],
            "x_unit": curves[0].get("unit_x") if curves else None,
            "y_unit": curves[0].get("unit_y") if curves else None,
        })

    # One sealed input directory per model, so no model can see another's
    # answers -- the flaw in the first run.
    for m in MODELS:
        d = OUT / m
        (d / "images").mkdir(parents=True)
        for t in tasks:
            shutil.copy(SRC / "images" / t["id"], d / "images" / t["id"])
        # Figure id and target units only. No ranges, no scales.
        (d / "tasks.json").write_text(json.dumps(stripped, ensure_ascii=False, indent=2) + "\n")

    shutil.copy(SRC / "_key.json", OUT / "_key.json")
    print(f"{len(tasks)} 図 × {len(MODELS)} モデル分を書き出した → {OUT}")
    print("各モデルに渡すのは画像・ID・報告単位のみ。軸レンジとスケールは渡さない。")


if __name__ == "__main__":
    main()
