"""Point-level metric (design §7.67): did the extractor find the experimental
points -- the marker positions a human digitizer clicks -- rather than only
trace a line near them?

Both sides are mapped into the axis-normalized unit square (log axes in log10
space), then:

1. per (predicted series, ground-truth series) pair, points are assigned one
   to one (Hungarian) and a pair within ``tau`` is a match;
2. series are assigned one to one (Hungarian) on cost ``1 - F1_tau``;
3. points of an unassigned ground-truth series are all missed, points of an
   unassigned predicted series are all false positives.

Point assignment maximizes the *number* of pairs within ``tau`` first and the
total distance only among those (a pair beyond ``tau`` costs more than every
in-``tau`` pair together). A plain minimum-total-distance assignment would
sometimes trade two matches for one closer one.

A value that cannot be placed on the axis -- non-finite, or non-positive on a
log axis -- never matches, but still counts as a point on its side (in
precision's denominator when predicted, recall's when ground truth).

Pure: numpy and scipy only, like ``matching.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from real_chart_bench.domain.curve import Curve, ScaleType

NORMS = ("euclidean", "chebyshev")


@dataclass(frozen=True)
class AxisFrame:
    """The axis extent the points are normalized by: each axis maps to 0..1."""

    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_scale: ScaleType = ScaleType.LINEAR
    y_scale: ScaleType = ScaleType.LINEAR

    def __post_init__(self) -> None:
        for name, rng, scale in (
            ("x", self.x_range, self.x_scale),
            ("y", self.y_range, self.y_scale),
        ):
            lo, hi = float(rng[0]), float(rng[1])
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"{name}_range must be finite")
            if scale is ScaleType.LOG and (lo <= 0 or hi <= 0):
                raise ValueError(f"{name}_range must be positive on a log axis")
            if _axis_value(lo, scale) == _axis_value(hi, scale):
                raise ValueError(f"{name}_range must have non-zero width")

    def normalize(self, curve: Curve) -> np.ndarray:
        """(n, 2) array of normalized points; NaN where a value cannot be placed."""
        xs = _normalize_axis(curve.x_values, self.x_range, self.x_scale)
        ys = _normalize_axis(curve.y_values, self.y_range, self.y_scale)
        return np.column_stack([xs, ys])


def _axis_value(v: float, scale: ScaleType) -> float:
    return math.log10(v) if scale is ScaleType.LOG else v


def _normalize_axis(
    values: Sequence[float], rng: tuple[float, float], scale: ScaleType
) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    lo, hi = _axis_value(float(rng[0]), scale), _axis_value(float(rng[1]), scale)
    if scale is ScaleType.LOG:
        with np.errstate(divide="ignore", invalid="ignore"):
            v = np.where(v > 0, np.log10(np.where(v > 0, v, 1.0)), np.nan)
    out = (v - lo) / (hi - lo)
    return np.where(np.isfinite(out), out, np.nan)


@dataclass(frozen=True)
class PointMatchResult:
    """One series-level line: a matched pair (both set), a missed ground-truth
    series (predicted None), or an extra predicted series (ground_truth None)."""

    predicted: Curve | None
    ground_truth: Curve | None
    n_matched: int
    n_predicted: int
    n_ground_truth: int
    matched_distances: tuple[float, ...] = ()
    # (predicted index, ground-truth index) of each matched point, aligned with
    # matched_distances; indices are into the Curves' own (x-sorted) points
    matched_pairs: tuple[tuple[int, int], ...] = ()

    @property
    def f1(self) -> float:
        return _f1(self.n_matched, self.n_predicted, self.n_ground_truth)


@dataclass(frozen=True)
class PointEvaluation:
    """Per-figure point metrics at one tau."""

    tau: float
    norm: str
    series: tuple[PointMatchResult, ...]
    n_matched: int
    n_predicted: int
    n_ground_truth: int
    point_recall: float
    point_precision: float
    point_f1: float
    # mean normalized distance of the matched points; None when nothing matched
    point_loc_error: float | None


def _f1(n_matched: int, n_predicted: int, n_ground_truth: int) -> float:
    denom = n_predicted + n_ground_truth
    return 2.0 * n_matched / denom if denom else 1.0


def _distances(pred: np.ndarray, gt: np.ndarray, norm: str) -> np.ndarray:
    diff = np.abs(pred[:, None, :] - gt[None, :, :])
    if norm == "chebyshev":
        d = diff.max(axis=2)
    else:
        d = np.sqrt((diff**2).sum(axis=2))
    # a point that cannot be placed (NaN) never matches
    return np.where(np.isfinite(d), d, np.inf)


@dataclass(frozen=True)
class _PointMatch:
    distances: tuple[float, ...] = ()
    pairs: tuple[tuple[int, int], ...] = ()


def _match_points(pred: np.ndarray, gt: np.ndarray, tau: float, norm: str) -> _PointMatch:
    """The matched pairs (and their distances) under a one-to-one assignment
    that maximizes the number of pairs within tau, then minimizes their total."""
    if len(pred) == 0 or len(gt) == 0:
        return _PointMatch()
    d = _distances(pred, gt, norm)
    within = d <= tau
    rows = np.flatnonzero(within.any(axis=1))
    cols = np.flatnonzero(within.any(axis=0))
    if len(rows) == 0:
        return _PointMatch()
    sub = d[np.ix_(rows, cols)]
    sub_within = sub <= tau
    # any non-match must cost more than every possible set of matches
    penalty = tau * (min(len(rows), len(cols)) + 1) + 1.0
    cost = np.where(sub_within, sub, penalty)
    r, c = linear_sum_assignment(cost)
    keep = sub_within[r, c]
    return _PointMatch(
        distances=tuple(float(v) for v in sub[r, c][keep]),
        pairs=tuple((int(rows[i]), int(cols[j])) for i, j in zip(r[keep], c[keep], strict=True)),
    )


def evaluate_points(
    predicted: Sequence[Curve],
    ground_truth: Sequence[Curve],
    frame: AxisFrame,
    tau: float,
    norm: str = "euclidean",
) -> PointEvaluation:
    """Point recall / precision / F1 / location error for one figure.

    ``norm`` is the point distance in the normalized space: "euclidean"
    (design §7.67) or "chebyshev" (max(|dx|, |dy|) -- Scatteract's per-axis
    criterion).

    Empty sides: no ground-truth points -> recall 1 (nothing to find); no
    predicted points -> precision 0 unless there was also nothing to find
    (an empty answer must not raise a macro-averaged precision). Both empty
    -> 1 / 1 / 1. ``point_loc_error`` is None when no point matched.
    """
    if not tau > 0:
        raise ValueError("tau must be positive")
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")

    pred_pts = [frame.normalize(c) for c in predicted]
    gt_pts = [frame.normalize(c) for c in ground_truth]
    n_pred_total = sum(len(c) for c in predicted)
    n_gt_total = sum(len(c) for c in ground_truth)

    series: list[PointMatchResult] = []
    if predicted and ground_truth:
        pair = [[_match_points(p, g, tau, norm) for g in gt_pts] for p in pred_pts]
        cost = np.array(
            [
                [
                    1.0 - _f1(len(pair[i][j].pairs), len(predicted[i]), len(ground_truth[j]))
                    for j in range(len(ground_truth))
                ]
                for i in range(len(predicted))
            ]
        )
        p_idx, g_idx = linear_sum_assignment(cost)
        for i, j in zip(p_idx.tolist(), g_idx.tolist(), strict=True):
            series.append(
                PointMatchResult(
                    predicted=predicted[i],
                    ground_truth=ground_truth[j],
                    n_matched=len(pair[i][j].pairs),
                    n_predicted=len(predicted[i]),
                    n_ground_truth=len(ground_truth[j]),
                    matched_distances=pair[i][j].distances,
                    matched_pairs=pair[i][j].pairs,
                )
            )
        assigned_p, assigned_g = set(p_idx.tolist()), set(g_idx.tolist())
    else:
        assigned_p, assigned_g = set(), set()

    for i, c in enumerate(predicted):
        if i not in assigned_p:
            series.append(PointMatchResult(c, None, 0, len(c), 0))
    for j, c in enumerate(ground_truth):
        if j not in assigned_g:
            series.append(PointMatchResult(None, c, 0, 0, len(c)))

    distances = [d for s in series for d in s.matched_distances]
    n_matched = len(distances)
    recall = n_matched / n_gt_total if n_gt_total else 1.0
    if n_pred_total:
        precision = n_matched / n_pred_total
    else:
        precision = 1.0 if n_gt_total == 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return PointEvaluation(
        tau=tau,
        norm=norm,
        series=tuple(series),
        n_matched=n_matched,
        n_predicted=n_pred_total,
        n_ground_truth=n_gt_total,
        point_recall=recall,
        point_precision=precision,
        point_f1=f1,
        point_loc_error=sum(distances) / n_matched if n_matched else None,
    )
