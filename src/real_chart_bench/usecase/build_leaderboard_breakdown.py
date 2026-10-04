"""Breaks a model's leaderboard result down by figure type (design §7.38,
HQ instruction 2026-08-27: "リーダーボードの表示整備(モデル別・図タイプ別
の内訳)"). Pure transform, same shape as build_leaderboard.py — no file
I/O, no HTML — so the grouping logic is unit-testable on its own.

"Figure type" here means: synthetic fixture, or real figure split by x-axis
scale (linear vs log). This is deliberately the cheapest breakdown that's
still honest and useful — it surfaces exactly the categories design docs
already call out as known-hard for the naive-CV baseline (log-x axes,
achromatic/black-and-gray series bundled into "real", §7.22/§7.32), without
inventing a new taxonomy or a new data-collection step. x_scale comes from
the already-committed data/verified_pairs/registry.json (via VerifiedPairing),
not from the results file itself, since results/*.json's per_figure entries
only ever carry a figure_id + scores (see evaluate_dataset.py's FigureResult)
— joining against the registry is this module's whole job.
"""

from __future__ import annotations

from dataclasses import dataclass

from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.verified_pairing import VerifiedPairing

_SYNTHETIC_PREFIX = "synthetic-"


@dataclass(frozen=True)
class CategoryBreakdown:
    category: str
    n_figures: int
    mean_summary_score: float
    # design §7.67: mean per-figure point_f1 at the result's primary tau;
    # None when the result has no point metrics
    mean_point_f1: float | None = None
    # design §7.72: the figures mean_point_f1 covers -- dense-marker figures
    # are left out, as in the result's point_metrics; None without points
    n_point_figures: int | None = None


def categorize_figure(figure_id: str, pairings_by_figure_id: dict[str, VerifiedPairing]) -> str:
    """`pairings_by_figure_id` is keyed by `f"{paper_id}-{figure_id}"`, matching
    how run_baselines.py / the LineFormer notebook build each DatasetItem's
    own figure_id -- see evaluate_dataset.py."""
    if figure_id.startswith(_SYNTHETIC_PREFIX):
        return "synthetic"
    pairing = pairings_by_figure_id.get(figure_id)
    if pairing is None:
        # Defensive, not expected in practice: a results file referencing a
        # figure_id the current registry no longer has (e.g. re-classified
        # since the run). One stale reference must not crash the leaderboard
        # build -- it just falls into its own bucket instead.
        return "real-unknown"
    return "real-log-x" if pairing.x_scale is ScaleType.LOG else "real-linear-x"


def build_model_breakdown(
    result: dict, pairings_by_figure_id: dict[str, VerifiedPairing]
) -> list[CategoryBreakdown]:
    """`result` is one results/*.json payload (pending-run payloads have no
    "per_figure" key and correctly yield an empty breakdown)."""
    block = result.get("point_metrics")
    tau_key = f"{block['primary_tau']:g}" if block else None
    buckets: dict[str, list[dict]] = {}
    for pf in result.get("per_figure", []):
        category = categorize_figure(pf["figure_id"], pairings_by_figure_id)
        buckets.setdefault(category, []).append(pf)

    def point_rows(rows: list[dict]) -> list[dict] | None:
        if tau_key is None or any("point" not in r for r in rows):
            return None
        # a row from before design §7.72 carries no density: kept, as then
        return [r for r in rows if not r.get("marker_density", {}).get("dense", False)]

    def point_f1(rows: list[dict]) -> float | None:
        kept = point_rows(rows)
        if not kept:
            return None
        return sum(r["point"]["by_tau"][tau_key]["point_f1"] for r in kept) / len(kept)

    def n_point(rows: list[dict]) -> int | None:
        kept = point_rows(rows)
        return None if kept is None else len(kept)

    return sorted(
        (
            CategoryBreakdown(
                category=category,
                n_figures=len(rows),
                mean_summary_score=sum(r["summary_score"] for r in rows) / len(rows),
                mean_point_f1=point_f1(rows),
                n_point_figures=n_point(rows),
            )
            for category, rows in buckets.items()
        ),
        key=lambda b: b.category,
    )
