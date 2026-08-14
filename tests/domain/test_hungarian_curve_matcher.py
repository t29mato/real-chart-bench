"""HungarianCurveMatcher: bipartite assignment between predicted and GT series
(design §3.2.3, §4.2). Unmatched GT series are detection misses; unmatched
predicted series are false positives.

Boundary cases map to docs/design/benchmark-architecture.md §3.3.
"""

import pytest

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.matching import HungarianCurveMatcher, SeriesMatchResult
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric


@pytest.fixture
def matcher():
    return HungarianCurveMatcher(metric=NormalizedYDistanceMetric())


def _line(y_values, series_label=""):
    x = tuple(float(i) for i in range(len(y_values)))
    return Curve(x_values=x, y_values=tuple(float(v) for v in y_values), series_label=series_label)


def test_no_predicted_and_no_ground_truth_returns_empty(matcher):
    assert matcher.match(predicted=[], ground_truth=[]) == []


def test_zero_predicted_series_marks_all_ground_truth_as_missed(matcher):
    # §3.3: 予測系列が0件(全滅ケース)
    gt = [_line([1, 2, 3]), _line([4, 5, 6])]

    results = matcher.match(predicted=[], ground_truth=gt)

    assert len(results) == 2
    assert all(r.predicted is None for r in results)
    assert {r.ground_truth for r in results} == set(gt)
    assert all(r.is_detection_miss for r in results)


def test_zero_ground_truth_series_marks_all_predicted_as_false_positive(matcher):
    predicted = [_line([1, 2, 3])]

    results = matcher.match(predicted=predicted, ground_truth=[])

    assert len(results) == 1
    assert results[0].ground_truth is None
    assert results[0].is_false_positive


def test_more_predicted_than_ground_truth_marks_extras_as_false_positive(matcher):
    # §3.3: 予測系列数 > GT系列数(誤検出過多)
    gt = [_line([1, 2, 3])]
    predicted = [_line([1, 2, 3]), _line([9, 9, 9])]

    results = matcher.match(predicted=predicted, ground_truth=gt)

    matched = [r for r in results if r.comparison is not None]
    false_positives = [r for r in results if r.is_false_positive]
    assert len(results) == 2
    assert len(matched) == 1
    assert len(false_positives) == 1
    # the exact-match curve should be the one that gets matched, not the odd one out
    assert matched[0].predicted.y_values == (1.0, 2.0, 3.0)


def test_matches_pairs_that_minimize_total_distance(matcher):
    gt_a = _line([0, 0, 0], series_label="a")
    gt_b = _line([10, 10, 10], series_label="b")
    pred_close_to_a = _line([0, 0, 1], series_label="pred1")
    pred_close_to_b = _line([10, 10, 9], series_label="pred2")

    results = matcher.match(
        predicted=[pred_close_to_b, pred_close_to_a],  # deliberately out of order
        ground_truth=[gt_a, gt_b],
    )

    by_gt_label = {r.ground_truth.series_label: r.predicted.series_label for r in results}
    assert by_gt_label["a"] == "pred1"
    assert by_gt_label["b"] == "pred2"


def test_empty_series_label_does_not_break_matching(matcher):
    # §3.3: 系列ラベルが空文字列/欠損
    gt = [_line([1, 2, 3], series_label="")]
    predicted = [_line([1, 2, 3], series_label="")]

    results = matcher.match(predicted=predicted, ground_truth=gt)

    assert len(results) == 1
    assert results[0].comparison is not None


def test_fewer_predicted_than_ground_truth_marks_extras_as_detection_miss(matcher):
    gt = [_line([1, 2, 3], series_label="a"), _line([9, 9, 9], series_label="b")]
    predicted = [_line([1, 2, 3], series_label="pred_a")]

    results = matcher.match(predicted=predicted, ground_truth=gt)

    misses = [r for r in results if r.is_detection_miss]
    matched = [r for r in results if r.comparison is not None]
    assert len(results) == 2
    assert len(misses) == 1
    assert len(matched) == 1
    assert misses[0].ground_truth.series_label == "b"


def test_series_match_result_requires_at_least_one_curve():
    with pytest.raises(ValueError, match="at least one curve"):
        SeriesMatchResult(predicted=None, ground_truth=None, comparison=None)


def test_series_match_result_rejects_comparison_without_both_curves():
    curve = _line([1, 2, 3])
    with pytest.raises(ValueError, match="requires both"):
        SeriesMatchResult(predicted=curve, ground_truth=None, comparison="not-none")  # type: ignore[arg-type]


def test_matcher_falls_back_to_distance_only_metric_without_compare():
    class DistanceOnlyMetric:
        """Minimal MetricStrategy implementation exposing only .distance()."""

        def distance(self, predicted: Curve, ground_truth: Curve) -> float:
            return 0.0 if predicted.y_values == ground_truth.y_values else 1.0

    matcher = HungarianCurveMatcher(metric=DistanceOnlyMetric())
    curve = _line([1, 2, 3])

    results = matcher.match(predicted=[curve], ground_truth=[curve])

    assert len(results) == 1
    assert results[0].comparison.mean_normalized_error == pytest.approx(0.0)
    assert results[0].comparison.coverage_ratio == pytest.approx(1.0)
