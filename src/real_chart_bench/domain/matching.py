"""Series-level bipartite matching between predicted and ground-truth curves
(design §3.2.3, §4.2: ``CurveMatcher`` / ``HungarianCurveMatcher``).

Unmatched ground-truth series are detection misses; unmatched predicted
series are false positives. Both are reported explicitly rather than
silently dropped, so a figure with the wrong number of series doesn't just
collapse to "0 score" — the caller can see exactly what went wrong.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.optimize import linear_sum_assignment

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.metrics import CurveComparisonResult, MetricStrategy


@dataclass(frozen=True)
class SeriesMatchResult:
    """One line of the match report: either a matched pair (both set), a
    detection miss (predicted is None), or a false positive (ground_truth is
    None)."""

    predicted: Curve | None
    ground_truth: Curve | None
    comparison: CurveComparisonResult | None

    def __post_init__(self) -> None:
        if self.predicted is None and self.ground_truth is None:
            raise ValueError("SeriesMatchResult must reference at least one curve")
        if self.comparison is not None and (self.predicted is None or self.ground_truth is None):
            raise ValueError("comparison requires both predicted and ground_truth to be set")

    @property
    def is_detection_miss(self) -> bool:
        return self.predicted is None

    @property
    def is_false_positive(self) -> bool:
        return self.ground_truth is None


class CurveMatcher(Protocol):
    """Port interface for pluggable series-matching strategies (design §4.2)."""

    def match(
        self, predicted: Sequence[Curve], ground_truth: Sequence[Curve]
    ) -> list[SeriesMatchResult]: ...


class HungarianCurveMatcher:
    """Optimal bipartite assignment (Kuhn-Munkres) minimizing total distance,
    as recommended in design §3.1/§3.2 (LineFormer-style multi-series
    handling)."""

    def __init__(self, metric: MetricStrategy) -> None:
        self._metric = metric

    def match(
        self, predicted: Sequence[Curve], ground_truth: Sequence[Curve]
    ) -> list[SeriesMatchResult]:
        # §3.3 boundary cases: either side empty.
        if not predicted and not ground_truth:
            return []
        if not predicted:
            return [SeriesMatchResult(None, gt, None) for gt in ground_truth]
        if not ground_truth:
            return [SeriesMatchResult(pred, None, None) for pred in predicted]

        cost_matrix = np.array(
            [[self._metric.distance(pred, gt) for gt in ground_truth] for pred in predicted]
        )
        pred_idx, gt_idx = linear_sum_assignment(cost_matrix)

        matched_pred_indices = set(pred_idx.tolist())
        matched_gt_indices = set(gt_idx.tolist())

        results: list[SeriesMatchResult] = []
        for p_i, g_i in zip(pred_idx, gt_idx, strict=True):
            comparison = _as_comparison(self._metric, predicted[p_i], ground_truth[g_i])
            results.append(SeriesMatchResult(predicted[p_i], ground_truth[g_i], comparison))

        for p_i, pred in enumerate(predicted):
            if p_i not in matched_pred_indices:
                results.append(SeriesMatchResult(pred, None, None))
        for g_i, gt in enumerate(ground_truth):
            if g_i not in matched_gt_indices:
                results.append(SeriesMatchResult(None, gt, None))

        return results


def _as_comparison(
    metric: MetricStrategy,
    predicted: Curve,
    ground_truth: Curve,
) -> CurveComparisonResult:
    # MetricStrategy only guarantees .distance(); use .compare() when the
    # concrete metric offers the richer result (duck-typed, since Protocol
    # only requires distance()).
    compare = getattr(metric, "compare", None)
    if compare is not None:
        return compare(predicted, ground_truth)
    d = metric.distance(predicted, ground_truth)
    return CurveComparisonResult(mean_normalized_error=d, coverage_ratio=1.0, chamfer_distance=d)
