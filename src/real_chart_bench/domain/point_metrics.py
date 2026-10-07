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

A figure may print a second y axis on the right (design §7.84). Then there are
two frames, sharing the x axis, and a ground-truth series says which one it is
read against (``Curve.y_axis``). A *predicted* series declares nothing: it is
normalized by the frame of whatever ground-truth series it is matched with, so
a series read against the right axis -- whose printed values live in the right
axis's numeric space -- can only match a right-axis series.

Pure: numpy and scipy only, like ``matching.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from real_chart_bench.domain.curve import Curve, ScaleType, YAxis

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


@dataclass(frozen=True)
class FigureFrames:
    """One figure's axis frames (design §7.84): the left y axis, and the right
    one when the figure prints a second y axis. The two share the x axis.

    A bare ``AxisFrame`` is accepted anywhere ``FigureFrames`` is, and means a
    figure with one y axis -- which is every figure of the dataset today, and
    is scored exactly as it was before this existed.
    """

    primary: AxisFrame
    secondary: AxisFrame | None = None

    @staticmethod
    def of(frame: AxisFrame | FigureFrames) -> FigureFrames:
        return frame if isinstance(frame, FigureFrames) else FigureFrames(primary=frame)

    def for_curve(self, curve: Curve) -> AxisFrame:
        """The frame ``curve`` is normalized by.

        A series on the second axis of a figure that has none is a dataset
        error, not a scoring decision: normalizing it by the left axis would
        be exactly the wrong-range normalization §7.84 removes, so it raises
        instead.
        """
        if curve.y_axis is YAxis.PRIMARY:
            return self.primary
        if self.secondary is None:
            raise ValueError(
                f"series {curve.series_label!r} is on the secondary y axis but the "
                "figure has no secondary frame (registry y2_range missing?)"
            )
        return self.secondary


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
    frame: AxisFrame | FigureFrames,
    tau: float,
    norm: str = "euclidean",
) -> PointEvaluation:
    """Point recall / precision / F1 / location error for one figure.

    ``norm`` is the point distance in the normalized space: "euclidean"
    (design §7.67) or "chebyshev" (max(|dx|, |dy|) -- Scatteract's per-axis
    criterion).

    ``frame`` is the figure's axis extent: one ``AxisFrame``, or
    ``FigureFrames`` for a figure with a second y axis (design §7.84). With
    two frames, each (predicted, ground-truth) series pair is compared in the
    frame of its *ground-truth* series -- the prediction is not asked which
    axis it read.

    Empty sides: no ground-truth points -> recall 1 (nothing to find); no
    predicted points -> precision 0 unless there was also nothing to find
    (an empty answer must not raise a macro-averaged precision). Both empty
    -> 1 / 1 / 1. ``point_loc_error`` is None when no point matched.
    """
    if not tau > 0:
        raise ValueError("tau must be positive")
    if norm not in NORMS:
        raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")

    frames = FigureFrames.of(frame)
    gt_frames = [frames.for_curve(c) for c in ground_truth]
    gt_pts = [f.normalize(c) for f, c in zip(gt_frames, ground_truth, strict=True)]
    # a predicted series is normalized once per frame it is compared against:
    # whichever frame the ground-truth series of the pair uses
    pred_pts = {f: [f.normalize(c) for c in predicted] for f in dict.fromkeys(gt_frames)}
    n_pred_total = sum(len(c) for c in predicted)
    n_gt_total = sum(len(c) for c in ground_truth)

    series: list[PointMatchResult] = []
    if predicted and ground_truth:
        pair = [
            [
                _match_points(pred_pts[gt_frames[j]][i], gt_pts[j], tau, norm)
                for j in range(len(ground_truth))
            ]
            for i in range(len(predicted))
        ]
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


# design §7.72: a figure is "dense" -- one-to-one point matching is ambiguous
# there -- when its markers sit closer than this many tau to their neighbour
DENSE_SPACING_TAU_FACTOR = 2.0


def median_nearest_neighbor_spacing(
    ground_truth: Sequence[Curve], frame: AxisFrame | FigureFrames
) -> float:
    """Median, over every ground-truth point, of the Euclidean distance to the
    nearest other point of the *same* series, in the axis-normalized space
    (log axes in log10) -- design §7.72.

    A point that cannot be placed on the axis is neither a neighbour nor
    counted. A series with fewer than two placeable points contributes
    nothing; ``inf`` when no series has two.

    Each series is measured in its own axis's frame (design §7.84), so a
    right-axis series' spacing is a distance on the plot box like any other.
    """
    frames = FigureFrames.of(frame)
    nearest: list[float] = []
    for curve in ground_truth:
        pts = frames.for_curve(curve).normalize(curve)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if len(pts) < 2:
            continue
        d = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(d, np.inf)
        nearest.extend(d.min(axis=1).tolist())
    if not nearest:
        return math.inf
    return float(np.median(nearest))


def is_dense_marker_figure(spacing: float, tau: float) -> bool:
    """design §7.72: dense when the median spacing is strictly below 2*tau
    (exactly 2*tau is not dense)."""
    return spacing < DENSE_SPACING_TAU_FACTOR * tau


def _covered(a: np.ndarray, b: np.ndarray, tau: float) -> int:
    """How many points of ``a`` have some point of ``b`` within ``tau``."""
    if len(a) == 0 or len(b) == 0:
        return 0
    d = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(axis=2))
    return int((d.min(axis=1) <= tau).sum())


def curve_f1(pred: Sequence[np.ndarray], gt: Sequence[np.ndarray], tau: float) -> dict:
    """design 7.83: F1 for dense-marker figures, in an already-normalised
    space. Recall = ground-truth points with a predicted point of the paired
    series within tau; precision = predicted points with a ground-truth point
    of the paired series within tau. No one-to-one point matching, so a curve
    sampled differently from the ground truth is not penalised, and curves
    need not be functions of x. Series are paired one to one (Hungarian on
    1 - pair F1); unpaired series contribute only misses."""
    from scipy.optimize import linear_sum_assignment

    n_gt = sum(len(g) for g in gt)
    n_pred = sum(len(p) for p in pred)
    if n_gt == 0 or n_pred == 0:
        return {"f1": 0.0, "recall": 0.0, "precision": 0.0, "n_gt": n_gt, "n_pred": n_pred}
    rec = np.zeros((len(pred), len(gt)), dtype=int)
    prec = np.zeros((len(pred), len(gt)), dtype=int)
    cost = np.ones((len(pred), len(gt)))
    for i, p in enumerate(pred):
        for j, g in enumerate(gt):
            rec[i, j] = _covered(np.asarray(g, float), np.asarray(p, float), tau)
            prec[i, j] = _covered(np.asarray(p, float), np.asarray(g, float), tau)
            r, q = rec[i, j] / max(len(g), 1), prec[i, j] / max(len(p), 1)
            cost[i, j] = 1 - (2 * r * q / (r + q) if r + q else 0.0)
    rows, cols = linear_sum_assignment(cost)
    recall = sum(rec[i, j] for i, j in zip(rows, cols)) / n_gt
    precision = sum(prec[i, j] for i, j in zip(rows, cols)) / n_pred
    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
    return {
        "f1": float(f1),
        "recall": float(recall),
        "precision": float(precision),
        "n_gt": n_gt,
        "n_pred": n_pred,
    }
