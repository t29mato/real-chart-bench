"""Per-figure rows and the dataset-level point_metrics block (design §7.67):
macro (mean of per-figure values) is primary, micro (pooled points) alongside."""

import pytest

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.point_metrics import AxisFrame, evaluate_points
from real_chart_bench.usecase.evaluate_dataset import DatasetItem, evaluate_model_on_dataset
from real_chart_bench.usecase.model_runner import ExtractionTask
from real_chart_bench.usecase.result_payload import (
    aggregate_point_metrics,
    figure_result_row,
    point_row,
)

UNIT = AxisFrame(x_range=(0.0, 1.0), y_range=(0.0, 1.0))


def _curve(points):
    return Curve(x_values=tuple(p[0] for p in points), y_values=tuple(p[1] for p in points))


def _row(points_by_tau):
    return {"figure_id": "f", "point": point_row(points_by_tau)}


def test_point_row_keys_each_tau_as_a_short_string():
    gt = [_curve([(0.1, 0.1), (0.5, 0.5)])]
    evals = [evaluate_points(gt, gt, UNIT, tau=t) for t in (0.01, 0.02, 0.05)]

    row = point_row(evals)

    assert row["norm"] == "euclidean"
    assert set(row["by_tau"]) == {"0.01", "0.02", "0.05"}
    assert row["by_tau"]["0.02"] == {
        "point_recall": 1.0,
        "point_precision": 1.0,
        "point_f1": 1.0,
        "point_loc_error": 0.0,
        "n_matched": 2,
        "n_predicted": 2,
        "n_ground_truth": 2,
    }


def test_macro_is_the_mean_of_figures_and_micro_pools_points():
    gt_small = [_curve([(0.1, 0.1), (0.2, 0.2)])]  # 2 points, all found
    gt_big = [_curve([(0.1 * i, 0.5) for i in range(1, 9)])]  # 8 points, 2 found
    pred_big = [_curve([(0.1, 0.5), (0.2, 0.5)])]
    rows = [
        _row([evaluate_points(gt_small, gt_small, UNIT, tau=0.02)]),
        _row([evaluate_points(pred_big, gt_big, UNIT, tau=0.02)]),
    ]

    block = aggregate_point_metrics(rows, primary_tau=0.02)

    assert block["primary_tau"] == 0.02
    assert block["norm"] == "euclidean"
    assert block["n_figures"] == 2
    macro = block["by_tau"]["0.02"]["macro"]
    micro = block["by_tau"]["0.02"]["micro"]
    assert macro["point_recall"] == pytest.approx((1.0 + 0.25) / 2)
    assert macro["point_precision"] == pytest.approx(1.0)
    assert micro["point_recall"] == pytest.approx(4 / 10)
    assert micro["point_precision"] == pytest.approx(1.0)
    assert micro["point_f1"] == pytest.approx(2 * 0.4 / 1.4)
    assert macro["point_f1"] == pytest.approx((1.0 + 2 * 0.25 / 1.25) / 2)


def test_location_error_skips_figures_without_matches_and_micro_weights_by_matches():
    gt = [_curve([(0.5, 0.5)])]
    gt2 = [_curve([(0.1, 0.1), (0.3, 0.1)])]
    rows = [
        _row([evaluate_points([_curve([(0.51, 0.5)])], gt, UNIT, tau=0.02)]),  # err 0.01, 1 match
        _row([evaluate_points(gt2, gt2, UNIT, tau=0.02)]),  # err 0, 2 matches
        _row([evaluate_points([], gt, UNIT, tau=0.02)]),  # no match: None
    ]

    block = aggregate_point_metrics(rows, primary_tau=0.02)["by_tau"]["0.02"]

    assert block["macro"]["point_loc_error"] == pytest.approx(0.005)
    assert block["macro"]["n_figures_with_matches"] == 2
    assert block["micro"]["point_loc_error"] == pytest.approx(0.01 / 3)


def test_location_error_is_none_when_nothing_matched_anywhere():
    gt = [_curve([(0.5, 0.5)])]
    rows = [_row([evaluate_points([], gt, UNIT, tau=0.02)])]

    block = aggregate_point_metrics(rows, primary_tau=0.02)["by_tau"]["0.02"]

    assert block["macro"]["point_loc_error"] is None
    assert block["micro"]["point_loc_error"] is None
    assert block["micro"]["point_precision"] == 0.0


def test_aggregate_is_none_when_any_row_lacks_point_metrics():
    gt = [_curve([(0.5, 0.5)])]
    rows = [_row([evaluate_points(gt, gt, UNIT, tau=0.02)]), {"figure_id": "old"}]

    assert aggregate_point_metrics(rows, primary_tau=0.02) is None


def test_aggregate_requires_the_primary_tau_to_be_present():
    gt = [_curve([(0.5, 0.5)])]
    rows = [_row([evaluate_points(gt, gt, UNIT, tau=0.05)])]

    with pytest.raises(ValueError):
        aggregate_point_metrics(rows, primary_tau=0.02)


def test_aggregate_rejects_mixed_norms():
    gt = [_curve([(0.5, 0.5)])]
    rows = [
        _row([evaluate_points(gt, gt, UNIT, tau=0.02)]),
        _row([evaluate_points(gt, gt, UNIT, tau=0.02, norm="chebyshev")]),
    ]

    with pytest.raises(ValueError):
        aggregate_point_metrics(rows, primary_tau=0.02)


class _Replay:
    def __init__(self, answer):
        self._answer = answer

    def extract(self, task):
        if isinstance(self._answer, Exception):
            raise self._answer
        return self._answer


def test_figure_result_row_keeps_the_existing_keys_and_adds_point():
    gt = [_curve([(0.2, 0.2), (0.6, 0.6)])]
    task = ExtractionTask(image_bytes=b"i", x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    (result,) = evaluate_model_on_dataset(
        _Replay(gt), [DatasetItem("f1", task, gt)], matcher_for=_matcher_for
    )

    row = figure_result_row(result)

    assert list(row)[:6] == [
        "figure_id",
        "summary_score",
        "match_rate",
        "mean_curve_distance",
        "mean_coverage_ratio",
        "error",
    ]
    assert row["summary_score"] == 1.0
    assert row["point"]["by_tau"]["0.02"]["point_f1"] == 1.0


def _matcher_for(task):
    from real_chart_bench.usecase.evaluate_dataset import matcher_for_task

    return matcher_for_task(task)
