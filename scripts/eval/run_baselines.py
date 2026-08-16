"""Runs registered baselines against the v0 evaluation set and writes
results/<model_id>.json (design §7.15, 司令塔加速指示 2026-08-16).

v0 evaluation set (documented honestly, see docs/experiments/2026-08-16-
baseline-eval.md): automatic image<->figure_id pairing remains unsolved
(§7.10/§7.12), and even manual pairing attempts on "single-figure" papers
turned out unreliable (2 of 3 manually-checked candidates had a plausible
image but numerically inconsistent ground truth values on closer
inspection). Per 司令塔's verification-gate instruction (design §7.19,
"量より信頼性。ベンチマークの信用が資産"), the real-image portion of this run
is no longer hardcoded inline — it is built exclusively from entries in
data/verified_pairs/registry.json that have passed status=VERIFIED (see
domain/verified_pairing.py, usecase/real_image_gate.py). REJECTED entries
in that registry are excluded by construction, not by convention. This run
therefore uses:

  - every VERIFIED real pair in the registry (currently 1: paper 18759,
    "Figure 3(a)", electrical conductivity vs temperature)
  - 3 synthetic fixtures with exact known ground truth, to exercise the
    harness across scenarios a single real example can't cover alone
    (multi-series, log x-axis, missing/black series the naive baseline
    can't see)

held-out papers are never used here — only PUBLIC-split figures.
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import UTC, datetime

import pymupdf

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.adapter.naive_cv_extractor import NaiveCvModelRunner  # noqa: E402
from real_chart_bench.adapter.panel_layout import PyMuPdfPanelSplitter  # noqa: E402
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.domain.curve import Curve, ScaleType  # noqa: E402
from real_chart_bench.domain.matching import HungarianCurveMatcher  # noqa: E402
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric  # noqa: E402
from real_chart_bench.domain.verified_pairing import VerifiedPairing  # noqa: E402
from real_chart_bench.usecase.evaluate_dataset import (  # noqa: E402
    DatasetItem,
    evaluate_model_on_dataset,
)
from real_chart_bench.usecase.model_runner import ExtractionTask  # noqa: E402
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
REGISTRY_PATH = REPO_ROOT / "data/verified_pairs/registry.json"


def _ground_truth_for(pairing: VerifiedPairing) -> list[Curve]:
    import csv
    import gzip

    raw_curves = []
    with gzip.open(REPO_ROOT / "data/cache/ThermoelectricMaterials_curves.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            if row["figure_id"] == pairing.figure_id:
                raw_curves.append(row)

    return [
        Curve(
            x_values=tuple(json.loads(row["x"])),
            y_values=tuple(json.loads(row["y"])),
            series_label=row["prop_y"],
        )
        for row in raw_curves
    ]


def _dataset_item_for(pairing: VerifiedPairing) -> DatasetItem:
    """Builds a DatasetItem from a VERIFIED registry entry. Never called on
    a REJECTED or unverified pairing -- see build_dataset()."""
    image_path = REPO_ROOT / "data/raw/images" / pairing.paper_id / pairing.image_path
    image_bytes = image_path.read_bytes()

    if pairing.panel_label is not None:
        splitter = PyMuPdfPanelSplitter()
        panels = {p.label: p for p in splitter.split(image_bytes)}
        image_bytes = panels[pairing.panel_label].image_bytes

    task = ExtractionTask(
        image_bytes=image_bytes,
        x_range=pairing.x_range,
        y_range=pairing.y_range,
        x_scale=pairing.x_scale,
    )
    figure_id = f"{pairing.paper_id}-{pairing.figure_id}"
    return DatasetItem(figure_id=figure_id, task=task, ground_truth=_ground_truth_for(pairing))


def _real_gold_items() -> list[DatasetItem]:
    """Every registry entry with status=VERIFIED (design §7.19 gate) —
    REJECTED and unverified pairings are structurally excluded, not just
    conventionally skipped. See data/verified_pairs/registry.json for the
    full audit trail of what was checked and why."""
    registry = load_registry(REGISTRY_PATH)
    return [_dataset_item_for(p) for p in select_verified_pairings(registry)]


def _synthetic_items() -> list[DatasetItem]:
    items = []

    # 1. simple single red line, linear axes
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)
    page.draw_line(pymupdf.Point(20, 20), pymupdf.Point(280, 280), color=(1, 0, 0), width=2)
    png = page.get_pixmap().tobytes("png")
    doc.close()
    gt = Curve(x_values=(0.0, 10.0), y_values=(10.0, 0.0))  # NW->SE = downward-sloping
    items.append(
        DatasetItem(
            figure_id="synthetic-linear-single",
            task=ExtractionTask(image_bytes=png, x_range=(0, 10), y_range=(0, 10)),
            ground_truth=[gt],
        )
    )

    # 2. two-series (red + blue), linear axes
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)
    page.draw_line(pymupdf.Point(20, 280), pymupdf.Point(280, 20), color=(1, 0, 0), width=2)
    page.draw_line(pymupdf.Point(20, 20), pymupdf.Point(280, 280), color=(0, 0, 1), width=2)
    png = page.get_pixmap().tobytes("png")
    doc.close()
    gt_up = Curve(x_values=(0.0, 10.0), y_values=(0.0, 10.0), series_label="up")
    gt_down = Curve(x_values=(0.0, 10.0), y_values=(10.0, 0.0), series_label="down")
    items.append(
        DatasetItem(
            figure_id="synthetic-linear-two-series",
            task=ExtractionTask(image_bytes=png, x_range=(0, 10), y_range=(0, 10)),
            ground_truth=[gt_up, gt_down],
        )
    )

    # 3. black line on log x-axis: naive CV baseline should score ~0 here
    # (it cannot see black/gray lines) -- a deliberate "known weak spot" case
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)
    page.draw_line(pymupdf.Point(20, 20), pymupdf.Point(280, 280), color=(0, 0, 0), width=2)
    png = page.get_pixmap().tobytes("png")
    doc.close()
    gt = Curve(x_values=(1.0, 100.0), y_values=(10.0, 0.0), x_scale=ScaleType.LOG)
    items.append(
        DatasetItem(
            figure_id="synthetic-log-black-line",
            task=ExtractionTask(
                image_bytes=png, x_range=(1, 100), y_range=(0, 10), x_scale=ScaleType.LOG
            ),
            ground_truth=[gt],
        )
    )

    return items


def build_dataset() -> list[DatasetItem]:
    items = _real_gold_items()
    items.extend(_synthetic_items())
    return items


def run(model_id: str, model_name: str, model) -> dict:
    items = build_dataset()
    matcher = HungarianCurveMatcher(metric=NormalizedYDistanceMetric())
    results = evaluate_model_on_dataset(model, items, matcher=matcher)

    per_figure = [
        {
            "figure_id": r.figure_id,
            "summary_score": r.evaluation.summary_score,
            "match_rate": r.evaluation.match_rate,
            "mean_curve_distance": r.evaluation.mean_curve_distance,
            "mean_coverage_ratio": r.evaluation.mean_coverage_ratio,
            "error": r.error,
        }
        for r in results
    ]
    mean_score = sum(p["summary_score"] for p in per_figure) / len(per_figure)

    payload = {
        "model_id": model_id,
        "model_name": model_name,
        "dataset_version": "v0-eval-pilot-2026-08-16",
        "run_at": datetime.now(UTC).isoformat(),
        "n_figures": len(per_figure),
        "mean_summary_score": mean_score,
        "per_figure": per_figure,
    }
    return payload


def main() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    payload = run("naive-cv-v0", "Naive CV (hue-bucket baseline)", NaiveCvModelRunner())
    out_path = RESULTS_DIR / f"{payload['model_id']}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    print(f"wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
