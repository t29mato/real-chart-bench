"""Figure-level aggregation (design §3.2.4, §7.4).

Combines the match-level results from a CurveMatcher into a single
EvaluationResult. v0 uses equal weights across three components — match
rate, curve-distance quality, and coverage ratio — per the 2026-08-15
司令塔 review decision (design §7.4); Phase 2 pilot data may motivate
re-weighting later.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.matching import CurveMatcher, SeriesMatchResult


@dataclass(frozen=True)
class EvaluationResult:
    matches: tuple[SeriesMatchResult, ...]
    match_rate: float
    mean_curve_distance: float
    mean_coverage_ratio: float
    summary_score: float


def evaluate_figure(
    predicted: Sequence[Curve],
    ground_truth: Sequence[Curve],
    matcher: CurveMatcher,
) -> EvaluationResult:
    matches = tuple(matcher.match(predicted, ground_truth))

    if not matches:
        # No predicted and no ground-truth series: nothing to detect, nothing
        # missed — trivially perfect rather than undefined (0/0).
        return EvaluationResult(
            matches=(),
            match_rate=1.0,
            mean_curve_distance=0.0,
            mean_coverage_ratio=1.0,
            summary_score=1.0,
        )

    matched_pairs = [m for m in matches if m.comparison is not None]

    denominator = max(len(predicted), len(ground_truth), 1)
    match_rate = len(matched_pairs) / denominator

    if matched_pairs:
        n = len(matched_pairs)
        mean_curve_distance = sum(m.comparison.distance for m in matched_pairs) / n
        mean_coverage_ratio = sum(m.comparison.coverage_ratio for m in matched_pairs) / n
    else:
        mean_curve_distance = 1.0
        mean_coverage_ratio = 0.0

    quality = 1.0 - mean_curve_distance
    summary_score = (match_rate + quality + mean_coverage_ratio) / 3

    return EvaluationResult(
        matches=matches,
        match_rate=match_rate,
        mean_curve_distance=mean_curve_distance,
        mean_coverage_ratio=mean_coverage_ratio,
        summary_score=summary_score,
    )
