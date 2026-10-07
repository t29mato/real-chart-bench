"""Dual-y figures in the point metric (design §7.84): a figure with a left and
a right y axis, whose series belong to different axes.

The decision under test is design §7.84's approach (a): the ground-truth series
carries the axis, and a predicted series is normalized by whatever frame its
matched ground-truth series uses. A prediction never declares an axis -- the
printed values it returns already say which axis it was read against.

Boundary cases are written before the implementation, as §7.84 requires.
"""

import math

import pytest

from real_chart_bench.domain.curve import Curve, ScaleType, YAxis
from real_chart_bench.domain.point_metrics import (
    AxisFrame,
    FigureFrames,
    evaluate_points,
    median_nearest_neighbor_spacing,
)

UNIT = AxisFrame(x_range=(0.0, 1.0), y_range=(0.0, 1.0))


def _curve(points, series_label="", y_axis=YAxis.PRIMARY):
    return Curve(
        x_values=tuple(float(p[0]) for p in points),
        y_values=tuple(float(p[1]) for p in points),
        series_label=series_label,
        y_axis=y_axis,
    )


def _as_tuple(result):
    """What a figure's row records, so two scorings can be compared whole."""
    return (
        result.n_matched,
        result.n_predicted,
        result.n_ground_truth,
        result.point_recall,
        result.point_precision,
        result.point_f1,
        result.point_loc_error,
        tuple((s.n_matched, s.n_predicted, s.n_ground_truth) for s in result.series),
    )


# --- the regression that matters: a figure with no second axis ----------------


def test_a_single_axis_figure_scores_identically_through_either_frame_type():
    """FigureFrames(primary=f) and f itself must be the same scoring, to the
    last bit: none of the benchmark's current figures is dual-y."""
    gt = [_curve([(0.1, 0.2), (0.5, 0.5), (0.9, 0.7)]), _curve([(0.2, 0.9), (0.6, 0.1)])]
    pred = [_curve([(0.11, 0.21), (0.5, 0.52), (0.9, 0.8)]), _curve([(0.2, 0.9), (0.61, 0.1)])]

    bare = evaluate_points(pred, gt, UNIT, tau=0.02)
    wrapped = evaluate_points(pred, gt, FigureFrames(primary=UNIT), tau=0.02)

    assert _as_tuple(bare) == _as_tuple(wrapped)


def test_a_secondary_frame_nobody_uses_changes_nothing():
    """A figure may carry a second axis that no ground-truth series sits on
    (the right axis belongs to a neighbouring panel inside the crop). The
    score must be the single-axis score."""
    gt = [_curve([(0.1, 0.2), (0.5, 0.5)])]
    pred = [_curve([(0.1, 0.21), (0.5, 0.51)])]
    frames = FigureFrames(
        primary=UNIT, secondary=AxisFrame(x_range=(0.0, 1.0), y_range=(0.0, 300.0))
    )

    assert _as_tuple(evaluate_points(pred, gt, frames, tau=0.02)) == _as_tuple(
        evaluate_points(pred, gt, UNIT, tau=0.02)
    )


def test_an_unspecified_axis_is_the_primary_axis():
    """A ground-truth series that says nothing about its axis is a left-axis
    series -- every row of the existing ground truth is one."""
    assert _curve([(0.0, 0.0)]).y_axis is YAxis.PRIMARY

    gt = [_curve([(0.2, 0.5)])]
    frames = FigureFrames(
        primary=UNIT, secondary=AxisFrame(x_range=(0.0, 1.0), y_range=(0.0, 100.0))
    )
    # 0.5 read against the primary axis matches; read against the secondary it
    # would be 0.005 and would not.
    result = evaluate_points([_curve([(0.2, 0.5)])], gt, frames, tau=0.02)

    assert result.point_f1 == 1.0


# --- two axes ------------------------------------------------------------------

# A figure drawing Seebeck coefficient (left, 0..1) and resistivity (right,
# 0..300) against the same x axis -- the shape materials-science papers use.
LEFT = AxisFrame(x_range=(0.0, 100.0), y_range=(0.0, 1.0))
RIGHT = AxisFrame(x_range=(0.0, 100.0), y_range=(0.0, 300.0))
TWIN = FigureFrames(primary=LEFT, secondary=RIGHT)


def test_each_series_is_normalized_by_its_own_axis():
    gt = [
        _curve([(10.0, 0.20), (50.0, 0.50), (90.0, 0.80)], "S"),
        _curve([(10.0, 60.0), (50.0, 150.0), (90.0, 240.0)], "rho", y_axis=YAxis.SECONDARY),
    ]
    # the same points, each read a hair off in its own axis's units
    pred = [
        _curve([(10.0, 0.21), (50.0, 0.51), (90.0, 0.81)]),
        _curve([(10.0, 63.0), (50.0, 153.0), (90.0, 243.0)]),
    ]

    result = evaluate_points(pred, gt, TWIN, tau=0.02)

    assert result.n_matched == 6
    assert result.point_f1 == 1.0


def test_the_right_axis_series_is_unscoreable_through_one_frame_alone():
    """Why §7.84 exists. The same figure as above, scored the way it would be
    before this section -- one frame, no axis on the ground truth. Normalized
    by the left axis, the right-axis series is hundreds of axis ranges away
    and can never match, however well it was read."""
    gt = [
        _curve([(10.0, 0.20), (50.0, 0.50), (90.0, 0.80)], "S"),
        _curve([(10.0, 60.0), (50.0, 150.0), (90.0, 240.0)], "rho"),
    ]
    pred = [
        _curve([(10.0, 0.21), (50.0, 0.51), (90.0, 0.81)]),
        _curve([(10.0, 63.0), (50.0, 153.0), (90.0, 243.0)]),
    ]

    through_one_frame = evaluate_points(pred, gt, LEFT, tau=0.02)

    assert through_one_frame.n_matched == 3  # only the left-axis series
    assert through_one_frame.point_f1 < 0.51


def test_a_prediction_read_against_the_wrong_axis_earns_nothing():
    """A predicted series holding the right axis's values must not be paid for
    the left axis's ground truth (design §7.84, why candidate (c) was rejected)."""
    gt = [_curve([(10.0, 0.20), (50.0, 0.50), (90.0, 0.80)], "S")]
    pred = [_curve([(10.0, 60.0), (50.0, 150.0), (90.0, 240.0)])]

    result = evaluate_points(pred, gt, TWIN, tau=0.02)

    assert result.n_matched == 0
    assert result.point_f1 == 0.0


def test_a_secondary_series_without_a_second_frame_is_an_error():
    """Silently falling back to the primary axis would reintroduce exactly the
    wrong-range normalization this section removes."""
    gt = [_curve([(10.0, 60.0)], "rho", y_axis=YAxis.SECONDARY)]

    with pytest.raises(ValueError, match="secondary"):
        evaluate_points([], gt, FigureFrames(primary=LEFT), tau=0.02)

    with pytest.raises(ValueError, match="secondary"):
        evaluate_points([], gt, LEFT, tau=0.02)


# --- scales differ between the two axes -----------------------------------------


def test_a_log_right_axis_beside_a_linear_left_axis():
    """Each axis carries its own scale: the left series is placed linearly,
    the right one in log10 space."""
    frames = FigureFrames(
        primary=AxisFrame(x_range=(0.0, 100.0), y_range=(0.0, 1.0)),
        secondary=AxisFrame(x_range=(0.0, 100.0), y_range=(1.0, 1e4), y_scale=ScaleType.LOG),
    )
    gt = [
        _curve([(10.0, 0.25), (90.0, 0.75)], "S"),
        _curve([(10.0, 10.0), (90.0, 1000.0)], "sigma", y_axis=YAxis.SECONDARY),
    ]
    # 10 -> 0.25 and 1000 -> 0.75 of the log axis; nudged by well under tau
    # there (a factor 1.02 is 0.00215 of the four decades) and by 0.01 on the
    # linear axis
    pred = [
        _curve([(10.0, 0.26), (90.0, 0.76)]),
        _curve([(10.0, 10.2), (90.0, 1020.0)]),
    ]

    result = evaluate_points(pred, gt, frames, tau=0.02)

    assert result.n_matched == 4
    assert result.point_f1 == 1.0


def test_a_value_unplaceable_on_a_log_right_axis_never_matches():
    frames = FigureFrames(
        primary=AxisFrame(x_range=(0.0, 100.0), y_range=(0.0, 1.0)),
        secondary=AxisFrame(x_range=(0.0, 100.0), y_range=(1.0, 1e4), y_scale=ScaleType.LOG),
    )
    gt = [_curve([(10.0, 10.0), (90.0, 1000.0)], "sigma", y_axis=YAxis.SECONDARY)]
    pred = [_curve([(10.0, -5.0), (90.0, 1000.0)])]

    result = evaluate_points(pred, gt, frames, tau=0.02)

    assert result.n_matched == 1
    assert result.n_predicted == 2  # the unplaceable point still costs precision
    assert result.point_precision == 0.5


# --- series matching across two axes ---------------------------------------------


def test_series_matching_picks_the_axis_that_makes_each_prediction_fit():
    """Two ground-truth series drawn on top of each other -- the same position
    in the plot box, one read against each axis. A predicted series could
    plausibly be either, and only its values decide."""
    frames = FigureFrames(
        primary=AxisFrame(x_range=(0.0, 100.0), y_range=(0.0, 1.0)),
        secondary=AxisFrame(x_range=(0.0, 100.0), y_range=(0.0, 2.0)),
    )
    # identical normalized positions: 0.5 of the left axis is 1.0 of the right
    gt = [
        _curve([(10.0, 0.5), (90.0, 0.5)], "left"),
        _curve([(10.0, 1.0), (90.0, 1.0)], "right", y_axis=YAxis.SECONDARY),
    ]
    pred = [
        _curve([(10.0, 1.0), (90.0, 1.0)]),  # right-axis values
        _curve([(10.0, 0.5), (90.0, 0.5)]),  # left-axis values
    ]

    result = evaluate_points(pred, gt, frames, tau=0.02)

    assert result.n_matched == 4
    paired = {
        s.ground_truth.series_label: s.predicted.y_values[0]
        for s in result.series
        if s.predicted is not None and s.ground_truth is not None
    }
    assert paired == {"left": 0.5, "right": 1.0}


def test_an_ambiguous_prediction_is_assigned_once_and_the_other_series_is_missed():
    """One predicted series, two ground-truth series it fits equally (one per
    axis). The one-to-one assignment pairs it with one of them; the other is
    all misses -- it must not be paid twice."""
    frames = FigureFrames(
        primary=AxisFrame(x_range=(0.0, 100.0), y_range=(0.0, 1.0)),
        secondary=AxisFrame(x_range=(0.0, 100.0), y_range=(0.0, 2.0)),
    )
    gt = [
        _curve([(10.0, 0.5), (90.0, 0.5)], "left"),
        _curve([(10.0, 1.0), (90.0, 1.0)], "right", y_axis=YAxis.SECONDARY),
    ]
    pred = [_curve([(10.0, 0.5), (90.0, 0.5)])]

    result = evaluate_points(pred, gt, frames, tau=0.02)

    assert result.n_matched == 2
    assert result.n_ground_truth == 4
    assert result.point_recall == 0.5
    assert result.point_precision == 1.0
    assert sum(1 for s in result.series if s.predicted is None) == 1


# --- marker density --------------------------------------------------------------


def test_marker_spacing_measures_each_series_in_its_own_axis():
    """The dense-figure rule (design §7.72) is a distance in the normalized
    space, so a right-axis series has to be normalized by the right axis too."""
    frames = FigureFrames(
        primary=AxisFrame(x_range=(0.0, 100.0), y_range=(0.0, 1.0)),
        secondary=AxisFrame(x_range=(0.0, 100.0), y_range=(0.0, 300.0)),
    )
    # both series sit 0.1 of the plot box apart in x and nothing in y
    left = _curve([(0.0, 0.5), (10.0, 0.5), (20.0, 0.5)], "left")
    right = _curve([(0.0, 150.0), (10.0, 150.0), (20.0, 150.0)], "right", y_axis=YAxis.SECONDARY)

    spacing = median_nearest_neighbor_spacing([left, right], frames)

    assert math.isclose(spacing, 0.1, rel_tol=1e-9)
