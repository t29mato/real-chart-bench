"""Per-figure rows and the dataset-level point_metrics block (design §7.67):
macro (mean of per-figure values) is primary, micro (pooled points) alongside.
The secondary table_metrics block (design §7.85) is built the same way."""

import pytest

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.point_metrics import AxisFrame, evaluate_points
from real_chart_bench.domain.table_metrics import TABLE_METRIC_LABEL, evaluate_chart_table
from real_chart_bench.usecase.evaluate_dataset import DatasetItem, evaluate_model_on_dataset
from real_chart_bench.usecase.model_runner import ExtractionTask
from real_chart_bench.usecase.result_payload import (
    DENSE_MARKER_CRITERION,
    aggregate_dense_marker_metrics,
    aggregate_point_metrics,
    aggregate_table_metrics,
    figure_result_row,
    point_row,
    table_row,
)

UNIT = AxisFrame(x_range=(0.0, 1.0), y_range=(0.0, 1.0))


def _curve(points):
    return Curve(x_values=tuple(p[0] for p in points), y_values=tuple(p[1] for p in points))


def _row(points_by_tau, dense=False, spacing=0.1, **curve):
    return {
        "figure_id": "f",
        "summary_score": curve.get("summary_score", 1.0),
        "match_rate": curve.get("match_rate", 1.0),
        "mean_curve_distance": curve.get("mean_curve_distance", 0.0),
        "mean_coverage_ratio": curve.get("mean_coverage_ratio", 1.0),
        "error": None,
        "point": point_row(points_by_tau),
        "marker_density": {"median_nn_spacing": spacing, "dense": dense},
    }


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


# --- dense-marker figures (design §7.72) ------------------------------------


def test_figure_result_row_records_marker_density():
    gt = [_curve([(0.2, 0.2), (0.21, 0.2), (0.22, 0.2)])]  # 0.01 apart: dense
    task = ExtractionTask(image_bytes=b"i", x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    (result,) = evaluate_model_on_dataset(
        _Replay(gt), [DatasetItem("f1", task, gt)], matcher_for=_matcher_for
    )

    row = figure_result_row(result)

    assert row["marker_density"]["median_nn_spacing"] == pytest.approx(0.01)
    assert row["marker_density"]["dense"] is True


def test_infinite_spacing_is_stored_as_null_and_not_dense():
    gt = [_curve([(0.2, 0.2)])]  # a lone point has no neighbour
    task = ExtractionTask(image_bytes=b"i", x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    (result,) = evaluate_model_on_dataset(
        _Replay(gt), [DatasetItem("f1", task, gt)], matcher_for=_matcher_for
    )

    row = figure_result_row(result)

    # JSON has no infinity
    assert row["marker_density"] == {"median_nn_spacing": None, "dense": False}


def test_point_aggregate_covers_only_the_non_dense_figures():
    gt = [_curve([(0.1, 0.1), (0.5, 0.5)])]
    found = evaluate_points(gt, gt, UNIT, tau=0.02)
    missed = evaluate_points([], gt, UNIT, tau=0.02)
    rows = [_row([found]), _row([found]), _row([missed], dense=True, spacing=0.01)]

    block = aggregate_point_metrics(rows, primary_tau=0.02)

    assert block["n_figures"] == 2
    assert block["n_dense_figures_excluded"] == 1
    assert block["dense_criterion"] == DENSE_MARKER_CRITERION
    assert block["by_tau"]["0.02"]["macro"]["point_f1"] == 1.0
    assert block["by_tau"]["0.02"]["micro"]["n_ground_truth"] == 4


def test_point_aggregate_is_none_when_every_figure_is_dense():
    gt = [_curve([(0.1, 0.1), (0.5, 0.5)])]
    rows = [_row([evaluate_points(gt, gt, UNIT, tau=0.02)], dense=True)]

    assert aggregate_point_metrics(rows, primary_tau=0.02) is None


def test_point_aggregate_is_none_when_a_row_lacks_marker_density():
    gt = [_curve([(0.1, 0.1), (0.5, 0.5)])]
    row = _row([evaluate_points(gt, gt, UNIT, tau=0.02)])
    del row["marker_density"]

    assert aggregate_point_metrics([row], primary_tau=0.02) is None


def test_dense_marker_metrics_average_the_curve_scores_of_the_dense_figures():
    gt = [_curve([(0.1, 0.1), (0.5, 0.5)])]
    p = [evaluate_points(gt, gt, UNIT, tau=0.02)]
    rows = [
        _row(p, summary_score=0.1),  # not dense: left out
        _row(p, dense=True, summary_score=0.8, match_rate=1.0, mean_curve_distance=0.1,
             mean_coverage_ratio=0.9),
        _row(p, dense=True, summary_score=0.6, match_rate=0.5, mean_curve_distance=0.3,
             mean_coverage_ratio=0.7),
    ]

    block = aggregate_dense_marker_metrics(rows)

    assert block["criterion"] == DENSE_MARKER_CRITERION
    assert block["n_figures"] == 2
    assert block["mean_summary_score"] == pytest.approx(0.7)
    assert block["mean_match_rate"] == pytest.approx(0.75)
    assert block["mean_curve_distance"] == pytest.approx(0.2)
    assert block["mean_coverage_ratio"] == pytest.approx(0.8)


def test_dense_marker_metrics_with_no_dense_figure_has_zero_figures_and_no_means():
    gt = [_curve([(0.1, 0.1), (0.5, 0.5)])]
    rows = [_row([evaluate_points(gt, gt, UNIT, tau=0.02)])]

    block = aggregate_dense_marker_metrics(rows)

    assert block["n_figures"] == 0
    assert block["mean_summary_score"] is None


def test_dense_marker_metrics_is_none_when_a_row_lacks_marker_density():
    assert aggregate_dense_marker_metrics([{"figure_id": "old", "summary_score": 1.0}]) is None


# --- chart-as-table metrics (design §7.85) -----------------------------------


def _table(predicted, ground_truth):
    return evaluate_chart_table(predicted, ground_truth, UNIT)


def _labelled(points, label):
    return Curve(
        x_values=tuple(float(p[0]) for p in points),
        y_values=tuple(float(p[1]) for p in points),
        series_label=label,
    )


def _table_figure_row(metrics):
    return {"figure_id": "f", "summary_score": 1.0, "table": table_row(metrics)}


def test_table_row_keeps_the_three_variants_and_the_shared_triple_counts():
    gt = [_labelled([(0.2, 0.4), (0.6, 0.8)], "HP680")]

    row = table_row(_table(gt, gt))

    assert set(row) == {"n_predicted", "n_ground_truth", "rms", "rms_value_only", "nms"}
    assert (row["n_predicted"], row["n_ground_truth"]) == (2, 2)
    assert row["rms"] == {"precision": 1.0, "recall": 1.0, "f1": 1.0, "score": 2.0}
    assert row["nms"]["f1"] == 1.0


def test_figure_result_row_adds_table_only_when_the_scorer_passes_it():
    gt = [_labelled([(0.2, 0.2), (0.6, 0.6)], "a")]
    task = ExtractionTask(image_bytes=b"i", x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    (result,) = evaluate_model_on_dataset(
        _Replay(gt), [DatasetItem("f1", task, gt)], matcher_for=_matcher_for
    )

    assert "table" not in figure_result_row(result)
    assert figure_result_row(result, table=_table(gt, gt))["table"]["rms"]["f1"] == 1.0


def test_table_aggregate_names_the_three_numbers_and_averages_the_figures():
    gt = [_labelled([(0.2, 0.4)], "HP680")]
    # the printed legend differs from the database label, and the value is 20%
    # of |g| off -- which only the axis-range denominator can still see
    wrong = [_labelled([(0.2, 0.48)], "printed legend")]
    rows = [_table_figure_row(_table(gt, gt)), _table_figure_row(_table(wrong, gt))]

    block = aggregate_table_metrics(rows)

    assert block["n_figures"] == 2
    assert block["definition"] == TABLE_METRIC_LABEL
    assert block["macro"]["rms_f1"] == pytest.approx(0.5)  # 1.0 and 0.0
    assert block["macro"]["rms_f1_value_only"] == pytest.approx(0.5)
    assert block["macro"]["nms_f1"] == pytest.approx((1.0 + 0.92) / 2)
    assert set(block["macro"]) == set(block["micro"])


def test_table_aggregate_pools_triples_in_the_micro_average():
    gt_small = [_labelled([(0.2, 0.4)], "a")]
    gt_big = [_labelled([(0.1 * i, 0.5) for i in range(1, 5)], "b")]
    rows = [
        _table_figure_row(_table(gt_small, gt_small)),  # 1 triple, perfect
        _table_figure_row(_table([], gt_big)),  # 4 triples, all missed
    ]

    block = aggregate_table_metrics(rows)

    assert (block["n_predicted"], block["n_ground_truth"]) == (1, 5)
    # macro averages the figures (1.0 and 0.0), micro the five triples
    assert block["macro"]["rms_f1"] == pytest.approx(0.5)
    assert block["micro"]["rms_recall"] == pytest.approx(0.2)
    assert block["micro"]["rms_precision"] == pytest.approx(1.0)


def test_table_aggregate_is_none_when_a_row_predates_the_table_metrics():
    gt = [_labelled([(0.2, 0.4)], "a")]
    rows = [_table_figure_row(_table(gt, gt)), {"figure_id": "old", "summary_score": 1.0}]

    assert aggregate_table_metrics(rows) is None
    assert aggregate_table_metrics([]) is None
