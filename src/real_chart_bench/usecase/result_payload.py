"""The plain-dict shapes written to results/*.json (design §7.15, §7.67).

Every scorer -- CV baselines, LineFormer, LLM replay, the tick-calibrated
subsets -- builds its per_figure rows and its point_metrics block here, so a
row means the same thing in every file.

per_figure[].marker_density (design §7.72) says whether the figure's markers
are too dense for one-to-one point matching. The point_metrics block covers
the non-dense figures only; the dense ones are summarized on curve distance
in dense_marker_metrics.

per_figure[].point keeps the per-figure counts (matched / predicted / ground
truth) alongside the ratios, so a figure subset's point_metrics block --
macro and micro -- can be rebuilt from the rows alone, without re-scoring.

per_figure[].table (design §7.85) holds the chart-as-table metrics -- RMS and
NMS -- in the same shape and for the same reason: the dataset-level
table_metrics block is rebuildable from the rows. They are secondary; the
primary metric stays macro point_f1 (§7.67). Unlike point_metrics, every
figure is in the table aggregate: RMS matches on headers, so dense markers
(§7.72) are no more ambiguous there than sparse ones.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from real_chart_bench.domain.point_metrics import PointEvaluation
from real_chart_bench.domain.table_metrics import TABLE_METRIC_LABEL, TableMetrics
from real_chart_bench.usecase.evaluate_dataset import FigureResult

POINT_FIELDS = ("point_recall", "point_precision", "point_f1", "point_loc_error")

# design 7.67: the primary metric since 2026-10-03, stored as point_metrics
POINT_METRIC_LABEL = (
    "point-level recall/precision/F1 of marker positions, axis-range-normalized "
    "(log axes in log10), one-to-one Hungarian per series pair (count of pairs "
    "within tau maximized), series assigned by 1 - F1_tau; primary: macro "
    "point_f1 at tau 0.02 (design 7.67)"
)


# design 7.72
DENSE_MARKER_CRITERION = (
    "dense when the median, over the ground-truth points, of the distance to the "
    "nearest point of the same series in the axis-normalized space (log axes in "
    "log10) is below 2*tau, tau = the primary point tau (0.02 -> 0.04); dense "
    "figures leave point_metrics and are summarized on curve distance in "
    "dense_marker_metrics (design 7.72)"
)


def tau_key(tau: float) -> str:
    """0.02 -> "0.02": the key a tau is stored under."""
    return f"{tau:g}"


def point_row(points: Sequence[PointEvaluation]) -> dict:
    norms = {p.norm for p in points}
    if len(norms) > 1:
        raise ValueError(f"mixed point norms in one figure: {sorted(norms)}")
    return {
        "norm": next(iter(norms), None),
        "by_tau": {
            tau_key(p.tau): {
                "point_recall": p.point_recall,
                "point_precision": p.point_precision,
                "point_f1": p.point_f1,
                "point_loc_error": p.point_loc_error,
                "n_matched": p.n_matched,
                "n_predicted": p.n_predicted,
                "n_ground_truth": p.n_ground_truth,
            }
            for p in points
        },
    }


# design 7.85: the chart-as-table variants, and the key each one's F1 is
# reported under. The order is the order they are reported in.
TABLE_VARIANT_KEYS = {
    "rms": ("rms_f1", "rms_precision", "rms_recall"),
    "rms_value_only": (
        "rms_f1_value_only",
        "rms_precision_value_only",
        "rms_recall_value_only",
    ),
    "nms": ("nms_f1", "nms_precision", "nms_recall"),
}


def table_row(metrics: TableMetrics) -> dict:
    """One per_figure["table"] entry (design 7.85). ``score`` -- the summed
    similarity of the matched triples -- is kept so the micro aggregate can be
    pooled from the rows; the triple counts are shared by all three variants."""
    row = {
        "n_predicted": metrics.rms.n_predicted,
        "n_ground_truth": metrics.rms.n_ground_truth,
    }
    for name in TABLE_VARIANT_KEYS:
        value = getattr(metrics, name)
        row[name] = {
            "precision": value.precision,
            "recall": value.recall,
            "f1": value.f1,
            "score": value.score,
        }
    return row


def figure_result_row(result: FigureResult, table: TableMetrics | None = None) -> dict:
    """One per_figure entry. The first six keys are the pre-§7.67 row,
    unchanged; "point" is added only when the figure was point-scored,
    "marker_density" (design 7.72) when its density was computed, and "table"
    (design 7.85) when the scorer also computed the chart-as-table metrics. An
    infinite spacing (no series with two points) is stored as null: JSON has
    no inf."""
    row = {
        "figure_id": result.figure_id,
        "summary_score": result.evaluation.summary_score,
        "match_rate": result.evaluation.match_rate,
        "mean_curve_distance": result.evaluation.mean_curve_distance,
        "mean_coverage_ratio": result.evaluation.mean_coverage_ratio,
        "error": result.error,
    }
    if result.points:
        row["point"] = point_row(result.points)
    if result.marker_density is not None:
        spacing = result.marker_density.median_nn_spacing
        row["marker_density"] = {
            "median_nn_spacing": spacing if math.isfinite(spacing) else None,
            "dense": result.marker_density.dense,
        }
    if table is not None:
        row["table"] = table_row(table)
    return row


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _macro(cells: list[dict]) -> dict:
    loc = [c["point_loc_error"] for c in cells if c["point_loc_error"] is not None]
    return {
        "point_recall": _mean([c["point_recall"] for c in cells]),
        "point_precision": _mean([c["point_precision"] for c in cells]),
        "point_f1": _mean([c["point_f1"] for c in cells]),
        # mean over the figures where at least one point matched
        "point_loc_error": _mean(loc),
        "n_figures_with_matches": len(loc),
    }


def _micro(cells: list[dict]) -> dict:
    matched = sum(c["n_matched"] for c in cells)
    predicted = sum(c["n_predicted"] for c in cells)
    truth = sum(c["n_ground_truth"] for c in cells)
    recall = matched / truth if truth else 1.0
    precision = matched / predicted if predicted else (1.0 if truth == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    loc_sum = sum(
        c["point_loc_error"] * c["n_matched"] for c in cells if c["point_loc_error"] is not None
    )
    return {
        "point_recall": recall,
        "point_precision": precision,
        "point_f1": f1,
        "point_loc_error": loc_sum / matched if matched else None,
        "n_matched": matched,
        "n_predicted": predicted,
        "n_ground_truth": truth,
    }


def _has_density(per_figure: Sequence[dict]) -> bool:
    return bool(per_figure) and all("marker_density" in row for row in per_figure)


def aggregate_point_metrics(per_figure: Sequence[dict], primary_tau: float) -> dict | None:
    """The payload's point_metrics block from per_figure rows: macro (mean of
    per-figure values, primary) and micro (pooled points) per tau, over the
    figures whose markers are not dense (design 7.72).

    None when any row lacks "point" or "marker_density" (a file scored before
    §7.67 / §7.72, or derived from one): a partial aggregate would silently
    describe fewer figures. None also when every figure is dense."""
    if not _has_density(per_figure) or any("point" not in row for row in per_figure):
        return None
    n_all = len(per_figure)
    per_figure = [row for row in per_figure if not row["marker_density"]["dense"]]
    if not per_figure:
        return None
    norms = {row["point"]["norm"] for row in per_figure}
    if len(norms) != 1:
        raise ValueError(f"rows were scored with different point norms: {sorted(norms)}")
    taus = list(per_figure[0]["point"]["by_tau"])
    if tau_key(primary_tau) not in taus:
        raise ValueError(f"primary tau {primary_tau} not among the scored taus {taus}")
    by_tau = {}
    for key in taus:
        cells = [row["point"]["by_tau"][key] for row in per_figure]
        by_tau[key] = {"macro": _macro(cells), "micro": _micro(cells)}
    return {
        "primary_tau": primary_tau,
        "norm": norms.pop(),
        "primary": "macro point_f1 at primary_tau (design 7.67)",
        "n_figures": len(per_figure),
        "n_dense_figures_excluded": n_all - len(per_figure),
        "dense_criterion": DENSE_MARKER_CRITERION,
        "by_tau": by_tau,
    }


def aggregate_dense_marker_metrics(per_figure: Sequence[dict]) -> dict | None:
    """The payload's dense_marker_metrics block (design 7.72): the curve-distance
    means over the figures whose markers are too dense for point matching.

    None when any row lacks "marker_density"; the means are None when no
    figure is dense."""
    if not _has_density(per_figure):
        return None
    rows = [row for row in per_figure if row["marker_density"]["dense"]]
    return {
        "criterion": DENSE_MARKER_CRITERION,
        "n_figures": len(rows),
        "mean_summary_score": _mean([r["summary_score"] for r in rows]),
        "mean_match_rate": _mean([r["match_rate"] for r in rows]),
        "mean_curve_distance": _mean([r["mean_curve_distance"] for r in rows]),
        "mean_coverage_ratio": _mean([r["mean_coverage_ratio"] for r in rows]),
    }


def _table_micro(cells: Sequence[dict], name: str, n_predicted: int, n_ground_truth: int) -> dict:
    """One variant pooled over the figures: the summed triple similarity over
    the pooled triple counts, the same ratios ``evaluate_table`` forms per
    figure."""
    score = sum(c[name]["score"] for c in cells)
    precision = score / n_predicted if n_predicted else 1.0
    recall = score / n_ground_truth if n_ground_truth else 1.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def aggregate_table_metrics(per_figure: Sequence[dict]) -> dict | None:
    """The payload's table_metrics block (design 7.85): the three chart-as-table
    numbers -- rms_f1, rms_f1_value_only, nms_f1 -- macro (mean of the
    per-figure values, as the chart-to-table literature reports RMS F1) and
    micro (triples pooled across figures).

    Every scored figure counts, dense ones included (§7.72 applies to point
    matching, not to header matching). None when any row lacks "table" (a file
    scored before §7.85, or derived from one): a partial aggregate would
    silently describe fewer figures."""
    if not per_figure or any("table" not in row for row in per_figure):
        return None
    cells = [row["table"] for row in per_figure]
    n_predicted = sum(c["n_predicted"] for c in cells)
    n_ground_truth = sum(c["n_ground_truth"] for c in cells)
    macro: dict[str, float | None] = {}
    micro: dict[str, float | None] = {}
    for name, (f1_key, precision_key, recall_key) in TABLE_VARIANT_KEYS.items():
        macro[f1_key] = _mean([c[name]["f1"] for c in cells])
        macro[precision_key] = _mean([c[name]["precision"] for c in cells])
        macro[recall_key] = _mean([c[name]["recall"] for c in cells])
        pooled = _table_micro(cells, name, n_predicted, n_ground_truth)
        micro[f1_key] = pooled["f1"]
        micro[precision_key] = pooled["precision"]
        micro[recall_key] = pooled["recall"]
    return {
        "definition": TABLE_METRIC_LABEL,
        # which of the three the project leads with -- not the benchmark's
        # primary metric, which stays macro point_f1 (design 7.67)
        "lead": "macro rms_f1_value_only (design 7.85)",
        "n_figures": len(cells),
        "n_predicted": n_predicted,
        "n_ground_truth": n_ground_truth,
        "macro": macro,
        "micro": micro,
    }
