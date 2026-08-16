"""Runs registered baselines against the v0 evaluation set and writes
results/<model_id>.json (design §7.15, 司令塔加速指示 2026-08-16).

v0 evaluation set (documented honestly, see docs/experiments/2026-08-16-
baseline-eval.md): automatic image<->figure_id pairing remains unsolved
(§7.10/§7.12), and even manual pairing attempts on "single-figure" papers
turned out unreliable (2 of 3 manually-checked candidates had a plausible
image but numerically inconsistent ground truth values on closer
inspection). This run therefore uses:

  - 1 manually verified REAL pair (paper 18759, "Figure 3(a)", electrical
    conductivity vs temperature) — carefully cross-checked against the raw
    Starrydata values (see docs/experiments), not just visual-match
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
from real_chart_bench.domain.curve import Curve, ScaleType  # noqa: E402
from real_chart_bench.domain.matching import HungarianCurveMatcher  # noqa: E402
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric  # noqa: E402
from real_chart_bench.usecase.evaluate_dataset import (  # noqa: E402
    DatasetItem,
    evaluate_model_on_dataset,
)
from real_chart_bench.usecase.model_runner import ExtractionTask  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"


def _real_gold_item() -> DatasetItem:
    """paper 18759, figure_id 12217, 'Figure 3(a)', split=public. Manually
    verified: 4 series, values cross-checked against raw Starrydata x/y
    (Temperature K vs Electrical conductivity ohm^-1*m^-1, ~61k-132k range)
    matching the chart's printed axis (Ω^-1 cm^-1 * 100 = Ω^-1 m^-1)."""
    import csv
    import gzip

    raw_curves = []
    with gzip.open(REPO_ROOT / "data/cache/ThermoelectricMaterials_curves.csv.gz", "rt") as f:
        for row in csv.DictReader(f):
            if row["figure_id"] == "12217":
                raw_curves.append(row)

    ground_truth = [
        Curve(
            x_values=tuple(json.loads(row["x"])),
            y_values=tuple(json.loads(row["y"])),
            series_label=row["prop_y"],
        )
        for row in raw_curves
    ]

    image_path = REPO_ROOT / "data/raw/images/18759/p04_embedded_4.jpg"
    splitter = PyMuPdfPanelSplitter()
    panels = {p.label: p for p in splitter.split(image_path.read_bytes())}
    panel_a = panels["a"]

    task = ExtractionTask(
        image_bytes=panel_a.image_bytes,
        x_range=(200.0, 500.0),
        y_range=(25000.0, 135000.0),  # Ω^-1 m^-1 (chart shows Ω^-1 cm^-1 * 100)
    )
    return DatasetItem(figure_id="18759-12217", task=task, ground_truth=ground_truth)


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
    items = [_real_gold_item()]
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
