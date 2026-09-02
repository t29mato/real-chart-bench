"""Curve-distance metrics (design §3.2.1-2, §4.2: ``MetricStrategy``).

NormalizedYDistanceMetric is the v0 primary metric: the predicted curve is
linearly interpolated at each ground-truth x-coordinate, and the resulting
y-error is normalized by the ground-truth y-range (ChartOCR/LineFormer-style
approach — see docs/design/benchmark-architecture.md §3.1 comparison table).

All public results are finite (never NaN/inf) so they can be used directly as
Hungarian-matching costs (see domain/matching.py) without extra guarding.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from real_chart_bench.domain.curve import Curve, ScaleType

# Sentinel cost/error used when two curves have no meaningful overlap to
# compare (see design §3.3: "GTとpredのx範囲が全く重ならない"). Chosen as the
# max of the metric's normal [0, 1] range so it acts as "worst case", not
# "infinitely bad" — keeps downstream aggregation (mean, Hungarian cost
# matrices) well-behaved.
WORST_CASE_DISTANCE = 1.0

# A "this range/difference is essentially zero" tolerance, used only to
# guard against dividing by (or comparing against) a near-zero span --
# never meant as a domain-meaningful noise tolerance. Scaled *relative* to
# the magnitude of the values actually being compared (design §7.47):
# ground_truth curves in this benchmark are stored in whatever unit gives
# the best match to the paper's own axis, which varies entry to entry from
# SI magnitudes (~1e-5) to a paper's display units (~1e3) -- a fixed
# absolute epsilon calibrated to one scale is meaningless (too loose) at
# the other, and can even turn a negligible floating-point-scale residual
# into a wildly inflated normalized error by dividing by a near-zero span
# that "looks" nonzero only because it's expressed in large-magnitude
# units.
_RELATIVE_EPSILON = 1e-9


def _is_negligible(span: float, *reference_values: float) -> bool:
    scale = max((abs(v) for v in reference_values), default=0.0)
    tolerance = _RELATIVE_EPSILON * scale if scale > 0.0 else _RELATIVE_EPSILON
    return abs(span) <= tolerance


@dataclass(frozen=True)
class CurveComparisonResult:
    """Full detail behind a single predicted-vs-ground-truth comparison."""

    mean_normalized_error: float
    coverage_ratio: float
    chamfer_distance: float

    @property
    def distance(self) -> float:
        """Cost in [0, 1], 0 = perfect. Used by CurveMatcher for assignment."""
        if self.coverage_ratio <= 0.0:
            return WORST_CASE_DISTANCE
        return min(self.mean_normalized_error, WORST_CASE_DISTANCE)


class MetricStrategy(Protocol):
    """Port interface for pluggable curve-distance metrics (design §4.2)."""

    def distance(self, predicted: Curve, ground_truth: Curve) -> float:
        """Cost in [0, 1] where 0.0 = perfect match. Never NaN/inf."""
        ...


def _to_x_space(curve: Curve, x_scale: ScaleType) -> np.ndarray:
    x = np.asarray(curve.x_values, dtype=float)
    if x_scale is ScaleType.LOG:
        if np.any(x <= 0):
            raise ValueError(
                "Cannot compute log-space distance: curve contains non-positive "
                "x values, which cannot be log-transformed"
            )
        return np.log10(x)
    return x


def _overlap_ratio(gt_lo: float, gt_hi: float, pred_lo: float, pred_hi: float) -> float:
    if _is_negligible(gt_hi - gt_lo, gt_lo, gt_hi):
        # Ground truth is effectively a single x-point: coverage is binary.
        return 1.0 if pred_lo <= gt_lo <= pred_hi else 0.0
    overlap_lo = max(gt_lo, pred_lo)
    overlap_hi = min(gt_hi, pred_hi)
    if overlap_hi <= overlap_lo:
        return 0.0
    return min(1.0, max(0.0, (overlap_hi - overlap_lo) / (gt_hi - gt_lo)))


class NormalizedYDistanceMetric:
    """v0 primary metric. See module docstring."""

    def compare(self, predicted: Curve, ground_truth: Curve) -> CurveComparisonResult:
        x_scale = ground_truth.x_scale
        pred_x = _to_x_space(predicted, x_scale)
        gt_x = _to_x_space(ground_truth, x_scale)
        pred_y = np.asarray(predicted.y_values, dtype=float)
        gt_y = np.asarray(ground_truth.y_values, dtype=float)

        coverage_ratio = _overlap_ratio(
            gt_lo=float(gt_x.min()),
            gt_hi=float(gt_x.max()),
            pred_lo=float(pred_x.min()),
            pred_hi=float(pred_x.max()),
        )

        if coverage_ratio <= 0.0:
            return CurveComparisonResult(
                mean_normalized_error=WORST_CASE_DISTANCE,
                coverage_ratio=0.0,
                chamfer_distance=WORST_CASE_DISTANCE,
            )

        pred_y_interp = np.interp(gt_x, pred_x, pred_y)
        gt_y_range = float(gt_y.max() - gt_y.min())

        if _is_negligible(gt_y_range, *gt_y.tolist()):
            # Flat (or single-point) ground truth: normalized error is
            # undefined by division, so fall back to an exact-match check
            # (still scale-relative, per point -- see _is_negligible).
            point_errors = np.array(
                [
                    0.0 if _is_negligible(p - g, p, g) else WORST_CASE_DISTANCE
                    for p, g in zip(pred_y_interp.tolist(), gt_y.tolist())
                ]
            )
        else:
            point_errors = np.abs(pred_y_interp - gt_y) / gt_y_range

        mean_error = float(np.mean(point_errors))
        chamfer = _chamfer_distance(
            pred_x,
            pred_y,
            gt_x,
            gt_y,
            gt_y_range or 1.0,
            x_scale,
            ground_truth,
        )

        return CurveComparisonResult(
            mean_normalized_error=mean_error,
            coverage_ratio=coverage_ratio,
            chamfer_distance=chamfer,
        )

    def distance(self, predicted: Curve, ground_truth: Curve) -> float:
        d = self.compare(predicted, ground_truth).distance
        assert math.isfinite(d)
        return d


def _chamfer_distance(
    pred_x: np.ndarray,
    pred_y: np.ndarray,
    gt_x: np.ndarray,
    gt_y: np.ndarray,
    y_range: float,
    x_scale: ScaleType,
    ground_truth: Curve,
) -> float:
    """Bidirectional nearest-neighbour point-set distance (auxiliary metric,
    design §3.2.2), normalized so x and y contribute on a comparable scale.
    """
    x_range = ground_truth.x_max - ground_truth.x_min
    if x_scale is ScaleType.LOG:
        x_range = float(gt_x.max() - gt_x.min())
    x_range = x_range or 1.0
    y_range = y_range or 1.0

    pred_pts = np.stack([pred_x / x_range, pred_y / y_range], axis=1)
    gt_pts = np.stack([gt_x / x_range, gt_y / y_range], axis=1)

    diffs = pred_pts[:, None, :] - gt_pts[None, :, :]
    dists = np.sqrt(np.sum(diffs**2, axis=-1))

    pred_to_gt = float(np.mean(np.min(dists, axis=1)))
    gt_to_pred = float(np.mean(np.min(dists, axis=0)))
    return (pred_to_gt + gt_to_pred) / 2
