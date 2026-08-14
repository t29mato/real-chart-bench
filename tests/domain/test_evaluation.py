"""Figure-level aggregation: evaluate_figure() combines match rate, mean curve
distance and mean coverage ratio into a single summary_score (design §3.2.4,
§7.4: v0 uses equal weights across the three components).
"""

import pytest

from real_chart_bench.domain.curve import Curve
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
