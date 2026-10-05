"""Figure-level aggregation: evaluate_figure() combines match rate, mean curve
distance and mean coverage ratio into a single summary_score (design §3.2.4,
§7.4: v0 uses equal weights across the three components).
"""

import pytest

from real_chart_bench.domain.curve import Curve, ScaleType
from real_chart_bench.domain.evaluation import evaluate_figure
from real_chart_bench.domain.matching import HungarianCurveMatcher
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric


@pytest.fixture
def matcher():
    return HungarianCurveMatcher(metric=NormalizedYDistanceMetric())


def _line(y_values):
    x = tuple(float(i) for i in range(len(y_values)))
    return Curve(x_values=x, y_values=tuple(float(v) for v in y_values))


def test_perfect_match_yields_summary_score_of_one(matcher):
    curve = _line([1, 2, 3])

    result = evaluate_figure(predicted=[curve], ground_truth=[curve], matcher=matcher)

    assert result.match_rate == pytest.approx(1.0)
    assert result.mean_coverage_ratio == pytest.approx(1.0)
    assert result.summary_score == pytest.approx(1.0)


def test_totally_missed_figure_yields_summary_score_of_zero(matcher):
    gt = [_line([1, 2, 3])]

    result = evaluate_figure(predicted=[], ground_truth=gt, matcher=matcher)

    assert result.match_rate == pytest.approx(0.0)
    assert result.summary_score == pytest.approx(0.0)


def test_no_ground_truth_and_no_predicted_is_a_trivially_perfect_empty_figure(matcher):
    result = evaluate_figure(predicted=[], ground_truth=[], matcher=matcher)

    assert result.matches == ()
    assert result.summary_score == pytest.approx(1.0)


def test_summary_score_is_mean_of_match_rate_quality_and_coverage(matcher):
    gt = _line([0.0, 0.0, 0.0])
    # halfway-decent prediction: overlaps fully but with some y error
    predicted = _line([0.0, 0.0, 1.0])

    result = evaluate_figure(predicted=[predicted], ground_truth=[gt], matcher=matcher)

    expected = (
        result.match_rate
        + (1.0 - result.mean_curve_distance)
        + result.mean_coverage_ratio
    ) / 3
    assert result.summary_score == pytest.approx(expected)


def test_false_positive_series_lowers_match_rate(matcher):
    gt = [_line([1, 2, 3])]
    predicted = [_line([1, 2, 3]), _line([9, 9, 9])]

    result = evaluate_figure(predicted=predicted, ground_truth=gt, matcher=matcher)

    assert result.match_rate == pytest.approx(0.5)  # 1 matched out of max(2 predicted, 1 gt)


# --- non-finite predicted points (owner decision 2026-10-05) -------------------------
# The curve-distance metric ignores a predicted point whose x or y is inf/nan; a
# series left with fewer than two finite points leaves curve scoring altogether.
# (The point metric keeps such points as predicted points that never match --
# see test_point_metrics.py.)

INF = float("inf")
NAN = float("nan")


def _xy(points):
    return Curve(
        x_values=tuple(float(p[0]) for p in points), y_values=tuple(float(p[1]) for p in points)
    )


def test_a_series_ending_in_infinity_scores_as_its_finite_points(matcher):
    gt = [_line([1, 2, 3])]
    finite = _xy([(0, 1), (1, 2), (2, 3)])
    with_inf = _xy([(0, 1), (1, 2), (2, 3), (INF, -INF)])

    expected = evaluate_figure(predicted=[finite], ground_truth=gt, matcher=matcher)
    result = evaluate_figure(predicted=[with_inf], ground_truth=gt, matcher=matcher)

    assert result.summary_score == pytest.approx(expected.summary_score)
    assert result.summary_score == pytest.approx(1.0)
    assert result.mean_curve_distance == pytest.approx(expected.mean_curve_distance)


def test_a_single_non_finite_y_value_is_ignored_too(matcher):
    gt = [_line([1, 2, 3])]
    pred = _xy([(0, 1), (1, INF), (2, 3)])

    result = evaluate_figure(predicted=[pred], ground_truth=gt, matcher=matcher)
    expected = evaluate_figure(
        predicted=[_xy([(0, 1), (2, 3)])], ground_truth=gt, matcher=matcher
    )

    assert result.summary_score == pytest.approx(expected.summary_score)


def test_nan_points_are_ignored(matcher):
    gt = [_line([1, 2, 3])]
    pred = _xy([(0, 1), (NAN, 5), (1, 2), (2, NAN), (2, 3)])

    result = evaluate_figure(predicted=[pred], ground_truth=gt, matcher=matcher)

    assert result.summary_score == pytest.approx(1.0)


def test_an_all_infinite_series_leaves_curve_scoring(matcher):
    gt = [_line([1, 2, 3])]
    good = _line([1, 2, 3])
    all_inf = _xy([(INF, INF), (-INF, INF), (INF, -INF)])

    result = evaluate_figure(predicted=[good, all_inf], ground_truth=gt, matcher=matcher)

    # dropped, not a false positive: match rate over max(1 scored, 1 gt)
    assert result.match_rate == pytest.approx(1.0)
    assert result.summary_score == pytest.approx(1.0)
    assert len(result.matches) == 1


def test_a_series_left_with_one_finite_point_leaves_curve_scoring(matcher):
    gt = [_line([1, 2, 3])]
    one_left = _xy([(1, 2), (INF, 3)])

    result = evaluate_figure(predicted=[one_left], ground_truth=gt, matcher=matcher)

    # nothing scoreable is left: the figure is as if nothing was predicted
    assert result.match_rate == pytest.approx(0.0)
    assert result.summary_score == pytest.approx(0.0)


def test_an_all_finite_single_point_series_is_still_scored(matcher):
    # the rule only touches series that held a non-finite value; an all-finite
    # series is scored as before, whatever its length
    gt = [_line([2])]
    pred = _line([2])

    result = evaluate_figure(predicted=[pred], ground_truth=gt, matcher=matcher)

    assert result.summary_score == pytest.approx(1.0)


def test_non_positive_x_on_a_log_axis_still_raises(matcher):
    # unchanged: a finite but non-positive x cannot be placed on a log axis,
    # and the curve comparison rejects it (the figure scores as a total miss)
    log_gt = Curve(x_values=(1.0, 10.0, 100.0), y_values=(1.0, 2.0, 3.0), x_scale=ScaleType.LOG)
    pred = Curve(
        x_values=(-1.0, 1.0, 10.0, 100.0), y_values=(0.0, 1.0, 2.0, 3.0), x_scale=ScaleType.LOG
    )

    with pytest.raises(ValueError):
        evaluate_figure(predicted=[pred], ground_truth=[log_gt], matcher=matcher)
