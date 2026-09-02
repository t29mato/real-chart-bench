"""Picks the figures to independently re-digitize for the human-ceiling
measurement (戦略メモ「柱B: GTの信頼性を定量化する」).

The memo's own warning about baseline-evaluation subsets applies just as
much to this one: a subset biased toward one figure "type" over-credits
whatever happens to be strong on that type. Here that means the selected
subset must cover -- not just sample conveniently from -- five axes the
memo names explicitly: linear vs log axes, single vs multiple series,
marker density, number of points, and the y-quantity being plotted.

Pure, no I/O (registry rows / ground-truth curves are passed in already
parsed) -- so this is fully unit-testable without touching the filesystem;
scripts/eval/select_human_ceiling_subset.py is the thin I/O wrapper that
loads data/verified_pairs/registry.json + ground_truth.json and calls this.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

_DIMENSIONS = ("axis_type", "series_bucket", "points_bucket", "density_bucket", "y_quantity")


@dataclass(frozen=True)
class FigureProfile:
    """The five distribution axes select_coverage_subset() must cover, plus
    identity fields. points_bucket/density_bucket are pre-bucketed strings
    (tertiles computed by build_figure_profiles() across the full
    population) rather than raw numbers, so the same coverage/balance
    algorithm treats all five dimensions uniformly as small category sets."""

    figure_id: str
    paper_id: str
    axis_type: str
    series_bucket: str
    points_bucket: str
    density_bucket: str
    y_quantity: str


@dataclass(frozen=True)
class DimensionCoverage:
    dimension: str
    category: str
    full_count: int
    full_fraction: float
    subset_count: int
    subset_fraction: float


def _axis_type(x_scale: str, y_scale: str) -> str:
    x_log = x_scale == "log"
    y_log = y_scale == "log"
    if x_log and y_log:
        return "log-both"
    if x_log:
        return "log-x"
    if y_log:
        return "log-y"
    return "linear-linear"


def _tertile_bucket(value: float, low_cutoff: float, high_cutoff: float) -> str:
    if value <= low_cutoff:
        return "low"
    if value <= high_cutoff:
        return "medium"
    return "high"


def _tertile_cutoffs(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    arr = np.asarray(values, dtype=float)
    low, high = np.percentile(arr, [100 / 3, 200 / 3])
    return float(low), float(high)


def _primary_y_quantity(curves: list[dict[str, Any]]) -> str:
    counts = Counter(c.get("prop_y", "") for c in curves)
    return counts.most_common(1)[0][0]


def build_figure_profiles(
    registry_rows: Sequence[dict[str, Any]],
    ground_truth: dict[str, list[dict[str, Any]]],
) -> list[FigureProfile]:
    """``registry_rows`` are dicts shaped like data/verified_pairs/registry.json
    entries (paper_id, figure_id, x_scale, y_scale); ``ground_truth`` is
    shaped like data/verified_pairs/ground_truth.json (bare figure_id ->
    list of {x, y, prop_y, ...}). Figures with no ground-truth curves at all
    are skipped -- there is nothing to re-digitize against."""
    raw: list[dict[str, Any]] = []
    for row in registry_rows:
        curves = ground_truth.get(row["figure_id"], [])
        if not curves:
            continue
        total_points = sum(len(c["x"]) for c in curves)
        n_series = len(curves)
        raw.append(
            {
                "figure_id": f"{row['paper_id']}-{row['figure_id']}",
                "paper_id": row["paper_id"],
                "axis_type": _axis_type(row.get("x_scale", "linear"), row.get("y_scale", "linear")),
                "series_bucket": "single" if n_series == 1 else "multi",
                "total_points": total_points,
                "avg_points_per_series": total_points / n_series,
                "y_quantity": _primary_y_quantity(curves),
            }
        )

    points_low, points_high = _tertile_cutoffs([r["total_points"] for r in raw])
    density_low, density_high = _tertile_cutoffs([r["avg_points_per_series"] for r in raw])

    return [
        FigureProfile(
            figure_id=r["figure_id"],
            paper_id=r["paper_id"],
            axis_type=r["axis_type"],
            series_bucket=r["series_bucket"],
            points_bucket=_tertile_bucket(r["total_points"], points_low, points_high),
            density_bucket=_tertile_bucket(r["avg_points_per_series"], density_low, density_high),
            y_quantity=r["y_quantity"],
        )
        for r in raw
    ]


def _category_key(profile: FigureProfile) -> frozenset[tuple[str, str]]:
    return frozenset((dim, getattr(profile, dim)) for dim in _DIMENSIONS)


def _population_fractions(
    profiles: Sequence[FigureProfile],
) -> dict[str, dict[str, float]]:
    n = len(profiles)
    fractions: dict[str, dict[str, float]] = {}
    for dim in _DIMENSIONS:
        counts = Counter(getattr(p, dim) for p in profiles)
        fractions[dim] = {cat: count / n for cat, count in counts.items()}
    return fractions


def select_coverage_subset(profiles: Sequence[FigureProfile], *, target_size: int) -> list[str]:
    """Deterministic two-phase selection:

    Phase 1 (coverage): greedy maximum-marginal-coverage set cover over the
    five dimensions' categories, so every category present in the
    population that *can* fit within target_size is touched by at least one
    selected figure.

    Phase 2 (balance): fills up to target_size by greedily minimizing the
    subset's squared deviation from the full population's per-category
    proportions, so the remaining budget is spent making the subset
    representative rather than arbitrary.

    Both phases break ties on figure_id (ascending) for reproducibility.
    """
    remaining = sorted(profiles, key=lambda p: p.figure_id)
    if not remaining:
        return []

    all_categories = set()
    for p in remaining:
        all_categories |= _category_key(p)

    selected: list[FigureProfile] = []
    covered: set[tuple[str, str]] = set()

    # Phase 1: coverage.
    while remaining and covered != all_categories and len(selected) < target_size:
        best = max(remaining, key=lambda p: (len(_category_key(p) - covered), -remaining.index(p)))
        gain = len(_category_key(best) - covered)
        if gain <= 0:
            break
        selected.append(best)
        covered |= _category_key(best)
        remaining.remove(best)

    if len(selected) >= target_size or not remaining:
        return [p.figure_id for p in selected[:target_size]]

    # Phase 2: proportional balance fill.
    pop_fractions = _population_fractions(profiles)
    subset_counts: dict[str, Counter[str]] = {dim: Counter() for dim in _DIMENSIONS}
    for p in selected:
        for dim in _DIMENSIONS:
            subset_counts[dim][getattr(p, dim)] += 1

    while len(selected) < target_size and remaining:
        total_after = len(selected) + 1

        def imbalance(candidate: FigureProfile) -> float:
            score = 0.0
            for dim in _DIMENSIONS:
                cat = getattr(candidate, dim)
                trial = subset_counts[dim].copy()
                trial[cat] += 1
                for pop_cat, pop_frac in pop_fractions[dim].items():
                    subset_frac = trial.get(pop_cat, 0) / total_after
                    score += (subset_frac - pop_frac) ** 2
            return score

        best = min(remaining, key=lambda p: (imbalance(p), p.figure_id))
        selected.append(best)
        for dim in _DIMENSIONS:
            subset_counts[dim][getattr(best, dim)] += 1
        remaining.remove(best)

    return [p.figure_id for p in selected]


def coverage_report(
    profiles: Sequence[FigureProfile], selected_figure_ids: Sequence[str]
) -> list[DimensionCoverage]:
    """Full-population vs. selected-subset distribution, per category, for
    every dimension -- this is the table select_human_ceiling_subset.py
    prints so the achieved coverage can be checked (and published) rather
    than asserted."""
    selected_set = set(selected_figure_ids)
    full_n = len(profiles)
    subset_n = len(selected_set)

    rows = []
    for dim in _DIMENSIONS:
        full_counts = Counter(getattr(p, dim) for p in profiles)
        subset_counts = Counter(
            getattr(p, dim) for p in profiles if p.figure_id in selected_set
        )
        for category in sorted(full_counts):
            full_count = full_counts[category]
            subset_count = subset_counts.get(category, 0)
            rows.append(
                DimensionCoverage(
                    dimension=dim,
                    category=category,
                    full_count=full_count,
                    full_fraction=full_count / full_n if full_n else 0.0,
                    subset_count=subset_count,
                    subset_fraction=subset_count / subset_n if subset_n else 0.0,
                )
            )
    return rows
