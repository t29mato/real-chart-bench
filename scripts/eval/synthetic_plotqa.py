"""PlotQA dot_line (100 test plots, data/synthetic/plotqa_dot_line/, CC-BY-4.0)
as a synthetic, marker-style counterpart of the real figures (design §7.75 (B)).

Builds the same DatasetItems the real-figure scorers use, so a synthetic
score is the same point F1 as a real-figure score:
  - ground truth: per series, x = the numeric tick labels the markers sit at
    (years; every dot_line test plot's x labels are numeric), y = annotated
    values -- exact, from the generator;
  - axis range for the "calibrated" condition and for point-metric
    normalization: the first and last major tick values (x by label, y by
    value), linear (PlotQA has no log axes).

Also writes, for the local-VLM worker, tasks and a key in the same layout as
data/llm_run_v3/ (fig_NNN names in a seeded random order):
  data/synthetic/plotqa_dot_line/run/{_key.json, calibrated/tasks.json, noaxis/tasks.json}

Usage: python scripts/eval/synthetic_plotqa.py   (writes the run/ files)
"""

from __future__ import annotations

import json
import pathlib
import random
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.domain.curve import Curve  # noqa: E402
from real_chart_bench.usecase.evaluate_dataset import DatasetItem  # noqa: E402
from real_chart_bench.usecase.model_runner import ExtractionTask  # noqa: E402

ROOT = REPO / "data/synthetic/plotqa_dot_line"
RUN = ROOT / "run"
SEED = 20261005
AS_PRINTED = (
    "Report numbers on the same scale as the printed tick labels. Do not apply "
    "a multiplier written in the axis title (e.g. '(10^4 S/m)' or 'x10^4'); if "
    "a tick label itself is written like 5x10^4, report 50000."
)


def _ranges(entry: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    xs = [float(v) for v in entry["x_axis"]["major_tick_labels"]]
    ys = [float(v) for v in entry["y_axis"]["major_tick_values"]]
    return (min(xs), max(xs)), (min(ys), max(ys))


def figure_id(entry: dict) -> str:
    return f"plotqa-{entry['image_index']}"


def load_entries() -> list[dict]:
    return json.loads((ROOT / "ground_truth.json").read_text())


def load_items() -> list[DatasetItem]:
    items = []
    for e in load_entries():
        x_range, y_range = _ranges(e)
        gt = []
        for s in e["series"]:
            xs = [e["x_labels_numeric"][int(i)] for i in s["x_raw"]]
            gt.append(Curve(x_values=tuple(xs), y_values=tuple(s["y"]), series_label=s["name"]))
        items.append(
            DatasetItem(
                figure_id=figure_id(e),
                task=ExtractionTask(
                    image_bytes=(ROOT / e["image"]).read_bytes(),
                    x_range=x_range,
                    y_range=y_range,
                ),
                ground_truth=gt,
            )
        )
    return items


def main() -> None:
    entries = load_entries()
    order = list(entries)
    random.Random(SEED).shuffle(order)
    key, calibrated, noaxis = {}, [], []
    for i, e in enumerate(order, 1):
        name = f"fig_{i:03d}.png"
        key[name] = {
            "paper_id": "plotqa",
            "figure_id": str(e["image_index"]),
            "image_path": str((ROOT / e["image"]).relative_to(REPO)),
        }
        x_range, y_range = _ranges(e)
        calibrated.append(
            {
                "id": name,
                "x_range": list(x_range),
                "y_range": list(y_range),
                "x_scale": "linear",
                "y_scale": "linear",
            }
        )
        noaxis.append({"id": name, "x_report": AS_PRINTED, "y_report": AS_PRINTED})
    for sub, tasks in (("calibrated", calibrated), ("noaxis", noaxis)):
        (RUN / sub).mkdir(parents=True, exist_ok=True)
        (RUN / sub / "tasks.json").write_text(json.dumps(tasks, indent=2) + "\n")
    (RUN / "_key.json").write_text(json.dumps(key, indent=2) + "\n")
    print(f"{len(entries)} plots -> {RUN}")


if __name__ == "__main__":
    main()
