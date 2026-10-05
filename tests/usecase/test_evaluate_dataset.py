import pytest

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.matching import HungarianCurveMatcher
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric
from real_chart_bench.usecase.evaluate_dataset import DatasetItem, evaluate_model_on_dataset
from real_chart_bench.usecase.model_runner import ExtractionTask


def _curve(y_values):
    x = tuple(float(i) for i in range(len(y_values)))
    return Curve(x_values=x, y_values=tuple(float(v) for v in y_values))


class _PerfectModel:
    """Returns the ground truth curves verbatim, ignoring the image — used
    to sanity-check the harness wiring gives a perfect score end-to-end."""

    def __init__(self, answers: dict[bytes, list[Curve]]):
        self._answers = answers

    def extract(self, task: ExtractionTask) -> list[Curve]:
        return self._answers[task.image_bytes]


def _matcher():
    return HungarianCurveMatcher(metric=NormalizedYDistanceMetric())


def test_perfect_model_scores_one_across_the_dataset():
    curve = _curve([1, 2, 3])
    task = ExtractionTask(image_bytes=b"img1", x_range=(0, 2), y_range=(1, 3))
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=[curve])]
    model = _PerfectModel({b"img1": [curve]})

    results = evaluate_model_on_dataset(model, items, matcher=_matcher())

    assert len(results) == 1
    assert results[0].figure_id == "f1"
    assert results[0].evaluation.summary_score == 1.0


def test_aggregate_score_is_the_mean_of_per_figure_scores():
    curve = _curve([1, 2, 3])
    off_curve = _curve([9, 9, 9])
    task1 = ExtractionTask(image_bytes=b"img1", x_range=(0, 2), y_range=(1, 3))
    task2 = ExtractionTask(image_bytes=b"img2", x_range=(0, 2), y_range=(1, 3))
    items = [
        DatasetItem(figure_id="f1", task=task1, ground_truth=[curve]),
        DatasetItem(figure_id="f2", task=task2, ground_truth=[curve]),
    ]
    model = _PerfectModel({b"img1": [curve], b"img2": [off_curve]})

    results = evaluate_model_on_dataset(model, items, matcher=_matcher())

    scores = [r.evaluation.summary_score for r in results]
    assert scores[0] == 1.0
    assert scores[1] < 1.0


def test_model_extraction_error_is_captured_not_fatal():
    class _BrokenModel:
        def extract(self, task):
            raise RuntimeError("boom")

    curve = _curve([1, 2, 3])
    task = ExtractionTask(image_bytes=b"img1", x_range=(0, 2), y_range=(1, 3))
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=[curve])]

    results = evaluate_model_on_dataset(_BrokenModel(), items, matcher=_matcher())

    assert len(results) == 1
    assert results[0].error is not None
    assert results[0].evaluation.summary_score == 0.0


def test_empty_dataset_returns_empty_results():
    results = evaluate_model_on_dataset(_PerfectModel({}), [], matcher=_matcher())
    assert results == []


# --- per-task matcher (design §7.66) ---------------------------------------


def test_matcher_for_task_floors_the_span_at_five_percent_of_a_linear_y_axis():
    from real_chart_bench.usecase.evaluate_dataset import Y_SPAN_FLOOR_FRACTION, matcher_for_task

    task = ExtractionTask(image_bytes=b"i", x_range=(0, 1), y_range=(0, 40))

    assert Y_SPAN_FLOOR_FRACTION == 0.05
    assert matcher_for_task(task).metric.min_y_span == pytest.approx(2.0)


def test_matcher_for_task_leaves_log_y_axes_unfloored():
    from real_chart_bench.domain.curve import ScaleType
    from real_chart_bench.usecase.evaluate_dataset import matcher_for_task

    task = ExtractionTask(
        image_bytes=b"i", x_range=(0, 1), y_range=(1e-6, 1e-1), y_scale=ScaleType.LOG
    )

    assert matcher_for_task(task).metric.min_y_span == 0.0


def test_evaluate_uses_a_per_task_matcher_when_given_one():
    # flat-ish GT on a 0-40 axis, prediction 0.3 off: unfloored it is ~0.73 of
    # worst case, floored at 2.0 it is 0.15
    from real_chart_bench.usecase.evaluate_dataset import matcher_for_task

    gt = Curve(x_values=(0.0, 1.0, 2.0), y_values=(0.0, 0.41, 0.2))
    pred = Curve(x_values=(0.0, 1.0, 2.0), y_values=(0.3, 0.71, 0.5))
    task = ExtractionTask(image_bytes=b"img", x_range=(0, 2), y_range=(0, 40))
    items = [DatasetItem(figure_id="f", task=task, ground_truth=[gt])]
    model = _PerfectModel({b"img": [pred]})

    (result,) = evaluate_model_on_dataset(model, items, matcher_for=matcher_for_task)

    assert result.evaluation.mean_curve_distance == pytest.approx(0.15)


# --- point metrics (design §7.67) -------------------------------------------


def test_every_figure_gets_point_metrics_at_the_three_taus():
    curve = _curve([1, 2, 3])
    task = ExtractionTask(image_bytes=b"img1", x_range=(0, 2), y_range=(1, 3))
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=[curve])]

    (result,) = evaluate_model_on_dataset(_PerfectModel({b"img1": [curve]}), items,
                                          matcher=_matcher())

    assert [p.tau for p in result.points] == [0.01, 0.02, 0.05]
    assert all(p.point_f1 == 1.0 for p in result.points)
    assert all(p.norm == "euclidean" for p in result.points)


def test_point_metrics_normalize_by_the_task_axis_range():
    from real_chart_bench.domain.curve import ScaleType

    gt = Curve(x_values=(10.0,), y_values=(100.0,))
    pred = Curve(x_values=(10.0,), y_values=(10 ** 2.06,))  # 0.015 of a 4-decade axis
    task = ExtractionTask(
        image_bytes=b"img1", x_range=(0, 20), y_range=(1, 1e4), y_scale=ScaleType.LOG
    )
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=[gt])]

    (result,) = evaluate_model_on_dataset(_PerfectModel({b"img1": [pred]}), items,
                                          matcher=_matcher())

    by_tau = {p.tau: p for p in result.points}
    assert by_tau[0.01].n_matched == 0
    assert by_tau[0.02].n_matched == 1


def test_failed_extraction_misses_every_point():
    class _BrokenModel:
        def extract(self, task):
            raise RuntimeError("boom")

    curve = _curve([1, 2, 3])
    task = ExtractionTask(image_bytes=b"img1", x_range=(0, 2), y_range=(1, 3))
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=[curve])]

    (result,) = evaluate_model_on_dataset(_BrokenModel(), items, matcher=_matcher())

    assert result.error is not None
    assert all(p.point_recall == 0.0 and p.point_f1 == 0.0 for p in result.points)
    assert all(p.n_ground_truth == 3 for p in result.points)


def test_point_taus_and_norm_are_configurable():
    curve = _curve([1, 2, 3])
    task = ExtractionTask(image_bytes=b"img1", x_range=(0, 2), y_range=(1, 3))
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=[curve])]

    (result,) = evaluate_model_on_dataset(
        _PerfectModel({b"img1": [curve]}), items, matcher=_matcher(),
        point_taus=(0.03,), point_norm="chebyshev",
    )

    assert [(p.tau, p.norm) for p in result.points] == [(0.03, "chebyshev")]


# --- marker density (design §7.72) ------------------------------------------


def _dense_item(figure_id, step, image=b"img1"):
    # one series on a 0..1 axis, points `step` apart
    gt = Curve(x_values=tuple(step * i for i in range(5)), y_values=(0.5,) * 5)
    task = ExtractionTask(image_bytes=image, x_range=(0.0, 1.0), y_range=(0.0, 1.0))
    return DatasetItem(figure_id=figure_id, task=task, ground_truth=[gt]), gt


def test_every_figure_records_its_marker_spacing_and_density():
    sparse, gt_sparse = _dense_item("sparse", 0.1, b"a")
    dense, gt_dense = _dense_item("dense", 0.01, b"b")

    results = evaluate_model_on_dataset(
        _PerfectModel({b"a": [gt_sparse], b"b": [gt_dense]}), [sparse, dense], matcher=_matcher()
    )

    by_id = {r.figure_id: r for r in results}
    assert by_id["sparse"].marker_density.median_nn_spacing == pytest.approx(0.1)
    assert by_id["sparse"].marker_density.dense is False
    assert by_id["dense"].marker_density.median_nn_spacing == pytest.approx(0.01)
    assert by_id["dense"].marker_density.dense is True
    assert by_id["dense"].marker_density.tau == 0.02


def test_marker_density_is_recorded_for_a_failed_extraction_too():
    class _BrokenModel:
        def extract(self, task):
            raise RuntimeError("boom")

    item, _ = _dense_item("dense", 0.01)

    (result,) = evaluate_model_on_dataset(_BrokenModel(), [item], matcher=_matcher())

    assert result.error is not None
    assert result.marker_density.dense is True


def test_density_criterion_follows_the_configured_tau():
    item, gt = _dense_item("f", 0.05)  # dense at tau 0.05 (< 0.1), not at 0.02

    (default,) = evaluate_model_on_dataset(_PerfectModel({b"img1": [gt]}), [item],
                                           matcher=_matcher())
    (wide,) = evaluate_model_on_dataset(_PerfectModel({b"img1": [gt]}), [item],
                                        matcher=_matcher(), density_tau=0.05)

    assert default.marker_density.dense is False
    assert wide.marker_density.dense is True


def test_an_answer_holding_infinity_is_scored_on_its_finite_points():
    # owner decision 2026-10-05: curve scoring ignores the non-finite point;
    # the point metric keeps it as a predicted point that never matches
    gt = Curve(x_values=(0.0, 1.0, 2.0), y_values=(1.0, 2.0, 3.0))
    answer = Curve(
        x_values=(0.0, 1.0, 2.0, float("inf")), y_values=(1.0, 2.0, 3.0, float("-inf"))
    )
    task = ExtractionTask(image_bytes=b"img1", x_range=(0, 2), y_range=(1, 3))
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=[gt])]

    (result,) = evaluate_model_on_dataset(
        _PerfectModel({b"img1": [answer]}), items, matcher=_matcher()
    )

    assert result.error is None
    assert result.evaluation.summary_score == pytest.approx(1.0)
    primary = next(p for p in result.points if p.tau == 0.02)
    assert primary.n_predicted == 4
    assert primary.n_matched == 3
    assert primary.point_precision == pytest.approx(3 / 4)
    assert primary.point_recall == pytest.approx(1.0)
