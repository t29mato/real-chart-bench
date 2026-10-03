"""Point-level metric (design §7.67): did the extractor find the experimental
points (marker positions), not just trace the line through them?

Boundary cases are the list in §7.67 "TDD の境界ケース", written before the
implementation.
"""

import math

import pytest

from real_chart_bench.domain.curve import Curve, ScaleType
from real_chart_bench.domain.evaluation import evaluate_figure
from real_chart_bench.domain.matching import HungarianCurveMatcher
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric
from real_chart_bench.domain.point_metrics import AxisFrame, evaluate_points

UNIT = AxisFrame(x_range=(0.0, 1.0), y_range=(0.0, 1.0))


def _curve(points, series_label=""):
    return Curve(
        x_values=tuple(float(p[0]) for p in points),
        y_values=tuple(float(p[1]) for p in points),
        series_label=series_label,
    )


# --- perfect prediction ------------------------------------------------------


def test_perfect_prediction_has_full_recall_and_precision_and_zero_error():
    gt = [_curve([(0.1, 0.2), (0.5, 0.5), (0.9, 0.7)]), _curve([(0.2, 0.9), (0.6, 0.1)])]

    result = evaluate_points(gt, gt, UNIT, tau=0.02)

    assert result.point_recall == 1.0
    assert result.point_precision == 1.0
    assert result.point_f1 == 1.0
    assert result.point_loc_error == 0.0
    assert (result.n_matched, result.n_ground_truth, result.n_predicted) == (5, 5, 5)


# --- empty sides ---------------------------------------------------------------


def test_zero_predicted_series_misses_every_ground_truth_point():
    gt = [_curve([(0.1, 0.1), (0.2, 0.2)])]

    result = evaluate_points([], gt, UNIT, tau=0.02)

    assert result.point_recall == 0.0
    # predicting nothing is not "precise": precision is 0, so an empty answer
    # cannot lift a macro-averaged precision
    assert result.point_precision == 0.0
    assert result.point_f1 == 0.0
    assert result.point_loc_error is None
    assert (result.n_matched, result.n_ground_truth, result.n_predicted) == (0, 2, 0)
    assert len(result.series) == 1
    assert result.series[0].predicted is None


def test_zero_ground_truth_series_makes_every_predicted_point_a_false_positive():
    pred = [_curve([(0.1, 0.1), (0.2, 0.2), (0.3, 0.3)])]

    result = evaluate_points(pred, [], UNIT, tau=0.02)

    assert result.point_precision == 0.0
    assert result.point_f1 == 0.0
    assert result.point_recall == 1.0  # nothing to find (vacuous)
    assert result.point_loc_error is None
    assert (result.n_matched, result.n_ground_truth, result.n_predicted) == (0, 0, 3)
    assert result.series[0].ground_truth is None


def test_both_sides_empty_is_a_vacuous_perfect_score():
    result = evaluate_points([], [], UNIT, tau=0.02)

    assert result.point_recall == 1.0
    assert result.point_precision == 1.0
    assert result.point_f1 == 1.0
    assert result.point_loc_error is None
    assert result.series == ()


def test_no_match_within_tau_gives_none_location_error():
    gt = [_curve([(0.1, 0.1)])]
    pred = [_curve([(0.9, 0.9)])]

    result = evaluate_points(pred, gt, UNIT, tau=0.02)

    assert result.n_matched == 0
    assert result.point_f1 == 0.0
    assert result.point_loc_error is None


# --- the tau boundary --------------------------------------------------------------


def test_distance_exactly_tau_matches():
    # 0.25 and 0.5 are exact binary fractions, so the distance is exactly tau
    frame = AxisFrame(x_range=(0.0, 4.0), y_range=(0.0, 1.0))
    gt = [_curve([(1.0, 0.5)])]
    pred = [_curve([(1.0, 0.75)])]

    result = evaluate_points(pred, gt, frame, tau=0.25)

    assert result.n_matched == 1
    assert result.point_loc_error == 0.25


def test_distance_just_above_tau_does_not_match():
    gt = [_curve([(0.5, 0.5)])]
    pred = [_curve([(0.5, 0.5 + 0.02 + 1e-9)])]

    result = evaluate_points(pred, gt, UNIT, tau=0.02)

    assert result.n_matched == 0
    assert result.point_recall == 0.0


def test_distance_is_euclidean_in_normalized_space():
    # 0.012 in x and 0.016 in y: each under tau, euclidean 0.02 -- exactly tau
    gt = [_curve([(0.5, 0.5)])]
    pred = [_curve([(0.512, 0.516)])]

    hit = evaluate_points(pred, gt, UNIT, tau=0.0201)
    miss = evaluate_points(pred, gt, UNIT, tau=0.0199)

    assert hit.n_matched == 1
    assert hit.point_loc_error == pytest.approx(0.02)
    assert miss.n_matched == 0


def test_normalization_is_by_axis_range_not_series_span():
    # y axis 0..1000: a 15-unit miss is 0.015 of the axis, inside tau even
    # though the series itself spans only 20 units
    frame = AxisFrame(x_range=(0.0, 100.0), y_range=(0.0, 1000.0))
    gt = [_curve([(10.0, 500.0), (50.0, 520.0)])]
    pred = [_curve([(10.0, 515.0), (50.0, 505.0)])]

    result = evaluate_points(pred, gt, frame, tau=0.02)

    assert result.n_matched == 2
    assert result.point_loc_error == pytest.approx(0.015)


# --- one-to-one ------------------------------------------------------------------


def test_one_predicted_point_near_two_ground_truth_points_matches_only_one():
    gt = [_curve([(0.50, 0.5), (0.51, 0.5)])]
    pred = [_curve([(0.505, 0.5)])]

    result = evaluate_points(pred, gt, UNIT, tau=0.02)

    assert result.n_matched == 1
    assert result.point_recall == 0.5
    assert result.point_precision == 1.0


def test_assignment_maximizes_the_number_of_matches_not_just_total_distance():
    # p1 sits exactly on g1. Pairing p1-g1 and p2-g2 has the smallest total
    # distance (0 + 0.027) but p2-g2 is beyond tau: one match. Pairing p1-g2
    # and p2-g1 (0.019 each) gives two -- the count is what is maximized.
    gt = [_curve([(0.5, 0.5), (0.519, 0.5)])]
    pred = [_curve([(0.5, 0.5), (0.5, 0.519)])]

    result = evaluate_points(pred, gt, UNIT, tau=0.02)

    assert result.n_matched == 2


# --- dense predictions ---------------------------------------------------------------


def test_dense_prediction_has_high_recall_and_low_precision():
    gt_points = [(0.1 + 0.1 * i, 0.5) for i in range(9)]
    gt = [_curve(gt_points)]
    # ten times as many points along the same line, every GT x included
    dense = [_curve([(0.1 + 0.01 * i, 0.5) for i in range(81)])]

    result = evaluate_points(dense, gt, UNIT, tau=0.02)

    assert result.point_recall == 1.0
    assert result.point_precision == pytest.approx(9 / 81)
    assert result.point_f1 < 0.25


# --- x rounded to tick values ------------------------------------------------------


def test_x_snapped_to_tick_values_loses_recall_where_interpolation_does_not():
    # a straight line y = x through points at x = 0.13, 0.37, 0.61, 0.89;
    # the prediction rounds x to the 0.25-spaced ticks but stays on the line
    frame = AxisFrame(x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    gt_xs = (0.13, 0.37, 0.61, 0.89)
    gt = [_curve([(x, x) for x in gt_xs])]
    snapped = [_curve([(x, x) for x in (0.0, 0.25, 0.5, 0.75, 1.0)])]

    points = evaluate_points(snapped, gt, frame, tau=0.02)
    curve = evaluate_figure(
        snapped, gt, HungarianCurveMatcher(metric=NormalizedYDistanceMetric())
    )

    assert points.point_recall == 0.0
    # interpolating the snapped table at the true x lands on the line: the
    # curve metric sees a near-perfect answer
    assert curve.summary_score > 0.99


# --- log axes ----------------------------------------------------------------------


def test_log_axis_is_normalized_in_log10_space():
    frame = AxisFrame(
        x_range=(0.0, 1.0), y_range=(1.0, 1e4), x_scale=ScaleType.LINEAR, y_scale=ScaleType.LOG
    )
    # 1e2 -> 0.5 of the axis; 10**(2 + 0.06) is 0.015 above in log space
    gt = [_curve([(0.5, 1e2)])]
    near = [_curve([(0.5, 10 ** 2.06)])]
    far_linear_but_near_log = [_curve([(0.5, 10 ** 2.1)])]

    assert evaluate_points(near, gt, frame, tau=0.02).n_matched == 1
    result_far = evaluate_points(far_linear_but_near_log, gt, frame, tau=0.02)
    assert result_far.n_matched == 0  # 0.025 in log space


def test_log_x_axis_is_normalized_in_log10_space():
    frame = AxisFrame(x_range=(1.0, 1000.0), y_range=(0.0, 1.0), x_scale=ScaleType.LOG)
    gt = [_curve([(10.0, 0.5)])]
    pred = [_curve([(10 ** 1.03, 0.5)])]  # 0.01 of a 3-decade axis

    result = evaluate_points(pred, gt, frame, tau=0.02)

    assert result.n_matched == 1
    assert result.point_loc_error == pytest.approx(0.01)


def test_non_positive_values_on_a_log_axis_never_match_but_count_as_predicted():
    frame = AxisFrame(x_range=(0.0, 1.0), y_range=(1.0, 100.0), y_scale=ScaleType.LOG)
    gt = [_curve([(0.2, 10.0), (0.4, 20.0)])]
    pred = [_curve([(0.2, 10.0), (0.4, 0.0), (0.6, -5.0)])]

    result = evaluate_points(pred, gt, frame, tau=0.02)

    assert result.n_matched == 1
    assert result.n_predicted == 3
    assert result.point_precision == pytest.approx(1 / 3)
    assert result.point_recall == 0.5


def test_non_finite_values_never_match():
    gt = [_curve([(0.2, 0.2)])]
    pred = [_curve([(0.2, math.nan), (0.2, 0.2)])]

    result = evaluate_points(pred, gt, UNIT, tau=0.02)

    assert result.n_matched == 1
    assert result.n_predicted == 2


# --- outside the axis range -----------------------------------------------------------


def test_points_outside_the_axis_range_are_still_compared():
    gt = [_curve([(1.10, -0.20)])]  # normalized outside 0..1
    pred = [_curve([(1.11, -0.20)])]

    result = evaluate_points(pred, gt, UNIT, tau=0.02)

    assert result.n_matched == 1
    assert result.point_loc_error == pytest.approx(0.01)


# --- series assignment -------------------------------------------------------------------


def test_series_assignment_maximizes_point_matches_unlike_the_curve_matcher():
    gt_xs = [0.1 * i for i in range(1, 10)]
    gt_ys = [0.3 + 0.03 * i for i in range(9)]
    gt = [_curve(list(zip(gt_xs, gt_ys, strict=True)), series_label="gt")]
    # p_offset: every point 0.03 above -- a near-perfect line, zero points
    # within tau
    p_offset = _curve([(x, y + 0.03) for x, y in zip(gt_xs, gt_ys, strict=True)], "offset")
    # p_exact: five points exactly on markers, four wild ones in between
    p_exact = _curve(
        [(x, y) if i % 2 == 0 else (x, 0.95) for i, (x, y) in enumerate(zip(gt_xs, gt_ys))],
        series_label="exact",
    )

    result = evaluate_points([p_offset, p_exact], gt, UNIT, tau=0.02)
    curve_matches = HungarianCurveMatcher(metric=NormalizedYDistanceMetric()).match(
        [p_offset, p_exact], gt
    )

    matched = [s for s in result.series if s.predicted is not None and s.ground_truth is not None]
    assert len(matched) == 1
    assert matched[0].predicted.series_label == "exact"
    assert result.n_matched == 5
    assert result.point_recall == pytest.approx(5 / 9)
    # every predicted point counts in precision, the unassigned series included
    assert result.point_precision == pytest.approx(5 / 18)
    # the curve metric pairs the gt with the offset line instead
    (curve_pair,) = [m for m in curve_matches if m.comparison is not None]
    assert curve_pair.predicted.series_label == "offset"


def test_two_series_are_assigned_to_their_own_ground_truth():
    gt = [
        _curve([(0.1, 0.1), (0.5, 0.1), (0.9, 0.1)], "low"),
        _curve([(0.1, 0.9), (0.5, 0.9), (0.9, 0.9)], "high"),
    ]
    pred = [
        _curve([(0.1, 0.9), (0.5, 0.9), (0.9, 0.9)], "p-high"),
        _curve([(0.1, 0.1), (0.5, 0.1), (0.9, 0.1)], "p-low"),
    ]

    result = evaluate_points(pred, gt, UNIT, tau=0.02)

    assert result.point_f1 == 1.0
    pairs = {(s.predicted.series_label, s.ground_truth.series_label) for s in result.series}
    assert pairs == {("p-high", "high"), ("p-low", "low")}


def test_points_matched_only_within_the_assigned_series_pair():
    # the predicted series' points sit on gt "b", but it is one series against
    # two: points of a series never match across to another gt series
    gt = [_curve([(0.1, 0.1), (0.3, 0.1)], "a"), _curve([(0.6, 0.6), (0.8, 0.6)], "b")]
    pred = [_curve([(0.1, 0.1), (0.3, 0.1), (0.6, 0.6), (0.8, 0.6)])]

    result = evaluate_points(pred, gt, UNIT, tau=0.02)

    assert result.n_matched == 2
    assert result.point_recall == 0.5
    assert result.point_precision == 0.5
    missed = [s for s in result.series if s.predicted is None]
    assert len(missed) == 1


def test_extra_predicted_series_lowers_precision_only():
    gt = [_curve([(0.1, 0.1), (0.5, 0.5)])]
    fit_line = _curve([(0.1 * i, 0.3) for i in range(10)])

    result = evaluate_points([gt[0], fit_line], gt, UNIT, tau=0.02)

    assert result.point_recall == 1.0
    assert result.point_precision == pytest.approx(2 / 12)
    assert any(s.ground_truth is None for s in result.series)


# --- per-series detail and validation -------------------------------------------------


def test_series_match_records_counts():
    gt = [_curve([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)])]
    pred = [_curve([(0.1, 0.1), (0.5, 0.6)])]

    (series,) = evaluate_points(pred, gt, UNIT, tau=0.02).series

    assert (series.n_matched, series.n_ground_truth, series.n_predicted) == (1, 3, 2)
    assert series.f1 == pytest.approx(2 * 1 / (3 + 2))


def test_axis_frame_rejects_a_zero_width_range():
    with pytest.raises(ValueError):
        AxisFrame(x_range=(1.0, 1.0), y_range=(0.0, 1.0))


def test_axis_frame_rejects_a_non_positive_log_range():
    with pytest.raises(ValueError):
        AxisFrame(x_range=(0.0, 1.0), y_range=(0.0, 10.0), y_scale=ScaleType.LOG)


def test_tau_must_be_positive():
    with pytest.raises(ValueError):
        evaluate_points([], [], UNIT, tau=0.0)


def test_reversed_axis_range_still_normalizes():
    frame = AxisFrame(x_range=(1.0, 0.0), y_range=(0.0, 1.0))
    gt = [_curve([(0.3, 0.3)])]
    pred = [_curve([(0.31, 0.3)])]

    assert evaluate_points(pred, gt, frame, tau=0.02).n_matched == 1


# --- distance norm ----------------------------------------------------------------------


def test_chebyshev_norm_matches_a_diagonal_offset_that_euclidean_rejects():
    # Scatteract (2017): |dx| <= 2% AND |dy| <= 2% of the axis range
    gt = [_curve([(0.5, 0.5)])]
    pred = [_curve([(0.515, 0.515)])]

    euclidean = evaluate_points(pred, gt, UNIT, tau=0.02)
    chebyshev = evaluate_points(pred, gt, UNIT, tau=0.02, norm="chebyshev")

    assert euclidean.n_matched == 0
    assert chebyshev.n_matched == 1
    assert chebyshev.point_loc_error == pytest.approx(0.015)


def test_default_norm_is_euclidean():
    gt = [_curve([(0.5, 0.5)])]
    pred = [_curve([(0.515, 0.515)])]

    assert evaluate_points(pred, gt, UNIT, tau=0.02, norm="euclidean").n_matched == 0
    assert evaluate_points(pred, gt, UNIT, tau=0.02).n_matched == 0


def test_chebyshev_still_rejects_one_coordinate_beyond_tau():
    gt = [_curve([(0.5, 0.5)])]
    pred = [_curve([(0.5, 0.525)])]

    assert evaluate_points(pred, gt, UNIT, tau=0.02, norm="chebyshev").n_matched == 0


def test_unknown_norm_is_rejected():
    with pytest.raises(ValueError):
        evaluate_points([], [], UNIT, tau=0.02, norm="manhattan")
