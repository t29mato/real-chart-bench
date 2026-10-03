"""The plain-dict shapes written to results/*.json (design §7.15, §7.67).

Every scorer -- CV baselines, LineFormer, LLM replay, the tick-calibrated
subsets -- builds its per_figure rows and its point_metrics block here, so a
row means the same thing in every file.

per_figure[].point keeps the per-figure counts (matched / predicted / ground
truth) alongside the ratios, so a figure subset's point_metrics block --
macro and micro -- can be rebuilt from the rows alone, without re-scoring.
"""

from __future__ import annotations

from collections.abc import Sequence

from real_chart_bench.domain.point_metrics import PointEvaluation
from real_chart_bench.usecase.evaluate_dataset import FigureResult

POINT_FIELDS = ("point_recall", "point_precision", "point_f1", "point_loc_error")

# design 7.67: the primary metric since 2026-10-03, stored as point_metrics
POINT_METRIC_LABEL = (
    "point-level recall/precision/F1 of marker positions, axis-range-normalized "
    "(log axes in log10), one-to-one Hungarian per series pair (count of pairs "
    "within tau maximized), series assigned by 1 - F1_tau; primary: macro "
    "point_f1 at tau 0.02 (design 7.67)"
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


def figure_result_row(result: FigureResult) -> dict:
    """One per_figure entry. The first six keys are the pre-§7.67 row,
    unchanged; "point" is added only when the figure was point-scored."""
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


def aggregate_point_metrics(per_figure: Sequence[dict], primary_tau: float) -> dict | None:
    """The payload's point_metrics block from per_figure rows: macro (mean of
    per-figure values, primary) and micro (pooled points) per tau.

    None when any row lacks "point" (a file scored before §7.67, or derived
    from one): a partial aggregate would silently describe fewer figures."""
    if not per_figure or any("point" not in row for row in per_figure):
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
        "by_tau": by_tau,
    }
