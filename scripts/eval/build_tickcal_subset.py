"""LineFormer, tick-calibrated: a reference row (design §7.66).

LineFormer detects curves but not axes. Its leaderboard row maps pixels to
data through the full image frame, which is how the n42 Colab run did it and
which costs it most of its score (design §7.64 addendum). This row maps the
same saved pixels through the owner-reviewed tick-mark positions instead --
"LineFormer, given where the axis ticks are" -- on the figures whose tick
positions are still valid (adapter/tick_plot_areas.py). It is LineFormer plus
oracle axis information, so it is labelled as such and ranked only against
the other main-table rows restricted to the same figures.

Writes results/lineformer-pretrained-tickcal.json and, for every row of the
main table, results/<model_id>-tickcal-subset.json. Run after every other
scorer (it reads their results).

Usage: python scripts/eval/build_tickcal_subset.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import UTC, datetime

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eval"))

from run_baselines import METRIC_LABEL, build_dataset  # noqa: E402

from real_chart_bench.adapter.lineformer_model_runner import (  # noqa: E402
    LineFormerPrediction,
    PrecomputedLineFormerModelRunner,
    image_key,
)
from real_chart_bench.adapter.tick_plot_areas import load_tick_plot_areas  # noqa: E402
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.usecase.evaluate_dataset import (  # noqa: E402
    PRIMARY_POINT_TAU,
    evaluate_model_on_dataset,
    matcher_for_task,
)
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402
from real_chart_bench.usecase.result_payload import (  # noqa: E402
    POINT_METRIC_LABEL,
    aggregate_dense_marker_metrics,
    aggregate_point_metrics,
    figure_result_row,
)

RESULTS = REPO / "results"
SUFFIX = "-tickcal-subset"


def main() -> None:
    items, _ = build_dataset()
    items = [i for i in items if not i.figure_id.startswith("synthetic-")]
    pairings = select_verified_pairings(load_registry(REPO / "data/verified_pairs/registry.json"))
    areas = load_tick_plot_areas(REPO / "data/verified_pairs/axis_pixel_candidates.json", pairings)
    subset = [i for i in items if tuple(i.figure_id.split("-", 1)) in areas]
    subset_ids = {i.figure_id for i in subset}

    base = json.loads((RESULTS / "naive-cv-v0.json").read_text())["dataset_version"]
    version = f"{base}-tickcal-subset-n{len(subset)}"

    (lf_file,) = [p for p in RESULTS.glob("lineformer-pretrained-n*.json") if p.stem[23:].isdigit()]
    raw = REPO / json.loads(lf_file.read_text())["raw_predictions"]
    predictions = [
        LineFormerPrediction.from_record(json.loads(line)) for line in raw.open() if line.strip()
    ]
    plot_areas = {
        image_key(i.task.image_bytes): areas[tuple(i.figure_id.split("-", 1))] for i in subset
    }
    runner = PrecomputedLineFormerModelRunner(predictions, plot_areas=plot_areas)
    results = evaluate_model_on_dataset(runner, subset, matcher_for=matcher_for_task)
    per_figure = [figure_result_row(r) for r in results]
    tickcal = {
        "model_id": "lineformer-pretrained-tickcal",
        "model_name": "LineFormer (pretrained) + 目盛位置(参考: 軸情報を外部から付与)",
        "dataset_version": version,
        "run_at": datetime.now(UTC).isoformat(),
        "n_figures": len(per_figure),
        "mean_summary_score": sum(p["summary_score"] for p in per_figure) / len(per_figure),
        "point_metrics": aggregate_point_metrics(per_figure, PRIMARY_POINT_TAU),
        # design 7.72: the figures too dense for point matching, on curve distance
        "dense_marker_metrics": aggregate_dense_marker_metrics(per_figure),
        "metric": METRIC_LABEL,
        "point_metric": POINT_METRIC_LABEL,
        "raw_predictions": str(raw.relative_to(REPO)),
        "note": (
            "Same saved LineFormer pixels as the main row, mapped through the owner-reviewed "
            "tick-mark pixel positions instead of the full image frame. LineFormer does not "
            "find axes itself, so this is LineFormer plus oracle axis information: a reference "
            "for how much of its gap is axis placement, not a method a user could run as is. "
            "Ranked only against the other rows restricted to the same figures."
        ),
        "per_figure": per_figure,
    }
    (RESULTS / "lineformer-pretrained-tickcal.json").write_text(
        json.dumps(tickcal, indent=2) + "\n"
    )
    print(
        f"tickcal: {tickcal['mean_summary_score']:.4f} on {len(per_figure)} figures ({version})"
    )

    for path in sorted(RESULTS.glob("*.json")):
        if path.stem.endswith(SUFFIX):
            path.unlink()
    for path in sorted(RESULTS.glob("*.json")):
        row = json.loads(path.read_text())
        if row.get("dataset_version") != base or path.stem.endswith("tickcal"):
            continue
        kept = [p for p in row["per_figure"] if p["figure_id"] in subset_ids]
        if len(kept) != len(subset_ids):
            raise SystemExit(f"{path.name} lacks some subset figures")
        out = {
            "model_id": row["model_id"] + SUFFIX,
            "model_name": row["model_name"],
            # cloud / local, when the source row records it (design 7.69)
            **({"execution": row["execution"]} if "execution" in row else {}),
            "dataset_version": version,
            "run_at": row["run_at"],
            "n_figures": len(kept),
            "mean_summary_score": sum(p["summary_score"] for p in kept) / len(kept),
            "point_metrics": aggregate_point_metrics(kept, PRIMARY_POINT_TAU),
            # design 7.72: the figures too dense for point matching, on curve distance
            "dense_marker_metrics": aggregate_dense_marker_metrics(kept),
            "metric": row.get("metric"),
            "point_metric": row.get("point_metric"),
            "derived_from": path.name,
            "per_figure": kept,
        }
        (RESULTS / f"{path.stem}{SUFFIX}.json").write_text(json.dumps(out, indent=2) + "\n")
        print(f"  {out['model_name']:<45} {out['mean_summary_score']:.4f}")


if __name__ == "__main__":
    main()
