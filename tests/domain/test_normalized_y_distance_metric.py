"""NormalizedYDistanceMetric: the primary curve-distance metric (design §3.2.1-2).

Predicted curve is linearly interpolated at each ground-truth x-coordinate;
error is |y_pred - y_gt| normalized by the ground-truth y-range. Distance is
in [0, 1] where 0.0 = perfect match, 1.0 = worst-case sentinel (used as
Hungarian-matching cost, so it must never be NaN/inf).

Boundary cases map to docs/design/benchmark-architecture.md §3.3.
"""

import math

import pytest

from real_chart_bench.domain.curve import Curve, ScaleType
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric


@pytest.fixture
def metric():
    return NormalizedYDistanceMetric()


def test_perfect_match_has_zero_distance(metric):
    # §3.3: 予測・GTが完全一致(スコア上限の確認)
    curve = Curve(x_values=(1.0, 2.0, 3.0), y_values=(1.0, 4.0, 9.0))

    result = metric.compare(predicted=curve, ground_truth=curve)

    assert result.mean_normalized_error == pytest.approx(0.0)
    assert result.coverage_ratio == pytest.approx(1.0)
    assert metric.distance(curve, curve) == pytest.approx(0.0)


def test_distance_is_bounded_in_zero_one_for_reasonable_curves(metric):
    predicted = Curve(x_values=(0.0, 10.0), y_values=(0.0, 10.0))
    ground_truth = Curve(x_values=(0.0, 10.0), y_values=(0.0, 5.0))

    d = metric.distance(predicted, ground_truth)

    assert 0.0 <= d <= 1.0


def test_no_x_overlap_returns_worst_case_distance_and_zero_coverage(metric):
    # §3.3: GTとpredのx範囲が全く重ならない(補間不能・カバレッジ0)
    predicted = Curve(x_values=(100.0, 200.0), y_values=(1.0, 2.0))
    ground_truth = Curve(x_values=(0.0, 10.0), y_values=(1.0, 2.0))

    result = metric.compare(predicted, ground_truth)

    assert result.coverage_ratio == pytest.approx(0.0)
    assert metric.distance(predicted, ground_truth) == pytest.approx(1.0)


def test_single_point_ground_truth_compares_at_that_x(metric):
    # §3.3: GT曲線が1点のみ(補間不可能なケース)
    predicted = Curve(x_values=(0.0, 10.0), y_values=(0.0, 10.0))
    ground_truth = Curve(x_values=(5.0,), y_values=(5.0,))

    result = metric.compare(predicted, ground_truth)

    assert result.coverage_ratio == pytest.approx(1.0)
    assert result.mean_normalized_error == pytest.approx(0.0)


def test_single_point_ground_truth_outside_predicted_range_has_zero_coverage(metric):
    predicted = Curve(x_values=(0.0, 10.0), y_values=(0.0, 10.0))
    ground_truth = Curve(x_values=(50.0,), y_values=(5.0,))

    result = metric.compare(predicted, ground_truth)

    assert result.coverage_ratio == pytest.approx(0.0)


def test_zero_y_range_ground_truth_falls_back_to_exact_match_check(metric):
    predicted_exact = Curve(x_values=(0.0, 1.0), y_values=(3.0, 3.0))
    predicted_off = Curve(x_values=(0.0, 1.0), y_values=(3.0, 4.0))
    ground_truth = Curve(x_values=(0.0, 1.0), y_values=(3.0, 3.0))

    assert metric.compare(predicted_exact, ground_truth).mean_normalized_error == pytest.approx(0.0)
    assert metric.compare(predicted_off, ground_truth).mean_normalized_error > 0.0


def test_log_scale_computes_error_in_log_x_space(metric):
    # x軸logスケール: 予測とGTが log(x) 空間で一致していれば誤差0になる
    log_curve = Curve(
        x_values=(1.0, 10.0, 100.0),
        y_values=(1.0, 2.0, 3.0),
        x_scale=ScaleType.LOG,
    )

    result = metric.compare(predicted=log_curve, ground_truth=log_curve)

    assert result.mean_normalized_error == pytest.approx(0.0)


def test_log_scale_with_non_positive_x_raises(metric):
    # §3.3: x軸が対数スケール、値が0または負(log変換不可)
    curve_with_zero_x = Curve(
        x_values=(0.0, 10.0), y_values=(1.0, 2.0), x_scale=ScaleType.LOG
    )
    valid_curve = Curve(x_values=(1.0, 10.0), y_values=(1.0, 2.0), x_scale=ScaleType.LOG)

    with pytest.raises(ValueError, match="non-positive"):
        metric.compare(curve_with_zero_x, valid_curve)

    with pytest.raises(ValueError, match="non-positive"):
        metric.compare(valid_curve, curve_with_zero_x)


def test_distance_never_returns_nan_or_inf(metric):
    predicted = Curve(x_values=(0.0,), y_values=(0.0,))
    ground_truth = Curve(x_values=(0.0,), y_values=(0.0,))

    d = metric.distance(predicted, ground_truth)

    assert math.isfinite(d)
