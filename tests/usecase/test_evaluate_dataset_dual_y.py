"""Dual-y figures through the harness (design §7.84): a task carrying a second
y axis, and ground-truth series that say which axis they belong to.

The regression that matters is the first test: a task with no second axis must
score exactly as it did before, because none of the benchmark's current figures
is dual-y.
"""

import math

import pytest

from real_chart_bench.domain.curve import Curve, ScaleType, YAxis
from real_chart_bench.domain.matching import HungarianCurveMatcher
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric
from real_chart_bench.usecase.evaluate_dataset import (
    DatasetItem,
    axis_frame_for_task,
    evaluate_model_on_dataset,
    figure_frames_for_task,
)
from real_chart_bench.usecase.model_runner import ExtractionTask


def _curve(points, y_axis=YAxis.PRIMARY):
    return Curve(
        x_values=tuple(float(p[0]) for p in points),
        y_values=tuple(float(p[1]) for p in points),
        y_axis=y_axis,
    )


class _ReplayModel:
    def __init__(self, answer: list[Curve]):
        self._answer = answer

    def extract(self, task: ExtractionTask) -> list[Curve]:
        return self._answer


def _matcher():
    return HungarianCurveMatcher(metric=NormalizedYDistanceMetric())


def _point_cells(result, tau_index=1):
    p = result.points[tau_index]
    return (p.tau, p.n_matched, p.n_predicted, p.n_ground_truth, p.point_f1)


# --- frames from the task ------------------------------------------------------


def test_a_task_without_a_second_axis_has_no_secondary_frame():
    task = ExtractionTask(image_bytes=b"i", x_range=(0.0, 100.0), y_range=(0.0, 1.0))

    frames = figure_frames_for_task(task)

    assert frames.secondary is None
    assert frames.primary == axis_frame_for_task(task)


def test_the_second_frame_shares_the_x_axis_and_takes_its_own_y():
    task = ExtractionTask(
        image_bytes=b"i",
        x_range=(1.0, 100.0),
        y_range=(0.0, 1.0),
        y2_range=(1.0, 300.0),
        y2_scale=ScaleType.LOG,
        x_scale=ScaleType.LOG,
    )

    frames = figure_frames_for_task(task)

    assert frames.secondary.x_range == (1.0, 100.0)
    assert frames.secondary.x_scale is ScaleType.LOG
    assert frames.secondary.y_range == (1.0, 300.0)
    assert frames.secondary.y_scale is ScaleType.LOG


def test_a_second_y_scale_without_a_second_y_range_is_an_error():
    with pytest.raises(ValueError, match="y2_range"):
        ExtractionTask(
            image_bytes=b"i",
            x_range=(0.0, 100.0),
            y_range=(0.0, 1.0),
            y2_scale=ScaleType.LOG,
        )


# --- the regression -------------------------------------------------------------


def test_a_single_axis_figure_is_scored_exactly_as_before():
    gt = [_curve([(10.0, 0.2), (50.0, 0.5), (90.0, 0.8)])]
    pred = [_curve([(10.0, 0.21), (50.0, 0.51), (90.0, 0.9)])]
    task = ExtractionTask(image_bytes=b"i", x_range=(0.0, 100.0), y_range=(0.0, 1.0))
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=gt)]

    (result,) = evaluate_model_on_dataset(_ReplayModel(pred), items, matcher=_matcher())

    assert _point_cells(result) == (0.02, 2, 3, 3, 2 / 3)
    assert result.marker_density.median_nn_spacing == pytest.approx(0.5, rel=1e-9)


# --- two axes --------------------------------------------------------------------


def test_each_ground_truth_series_is_scored_against_its_own_axis():
    gt = [
        _curve([(10.0, 0.2), (50.0, 0.5), (90.0, 0.8)]),
        _curve([(10.0, 60.0), (50.0, 150.0), (90.0, 240.0)], y_axis=YAxis.SECONDARY),
    ]
    pred = [
        _curve([(10.0, 0.21), (50.0, 0.51), (90.0, 0.81)]),
        _curve([(10.0, 63.0), (50.0, 153.0), (90.0, 243.0)]),
    ]
    task = ExtractionTask(
        image_bytes=b"i", x_range=(0.0, 100.0), y_range=(0.0, 1.0), y2_range=(0.0, 300.0)
    )
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=gt)]

    (result,) = evaluate_model_on_dataset(_ReplayModel(pred), items, matcher=_matcher())

    assert _point_cells(result) == (0.02, 6, 6, 6, 1.0)


def test_a_secondary_series_on_a_task_without_a_second_axis_fails_the_figure():
    """A dataset error, not a model failure: the registry and the ground truth
    disagree about how many y axes the figure has. It must stop the run rather
    than be absorbed as this model's zero."""
    gt = [_curve([(10.0, 60.0)], y_axis=YAxis.SECONDARY)]
    task = ExtractionTask(image_bytes=b"i", x_range=(0.0, 100.0), y_range=(0.0, 1.0))
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=gt)]

    with pytest.raises(ValueError, match="secondary"):
        evaluate_model_on_dataset(_ReplayModel([]), items, matcher=_matcher())


def test_marker_density_uses_each_series_own_axis():
    gt = [
        _curve([(0.0, 0.5), (10.0, 0.5), (20.0, 0.5)]),
        _curve([(0.0, 150.0), (10.0, 150.0), (20.0, 150.0)], y_axis=YAxis.SECONDARY),
    ]
    task = ExtractionTask(
        image_bytes=b"i", x_range=(0.0, 100.0), y_range=(0.0, 1.0), y2_range=(0.0, 300.0)
    )
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=gt)]

    (result,) = evaluate_model_on_dataset(_ReplayModel([]), items, matcher=_matcher())

    assert math.isclose(result.marker_density.median_nn_spacing, 0.1, rel_tol=1e-9)
    assert result.marker_density.dense is False


def test_a_failed_extraction_on_a_dual_y_figure_misses_every_point():
    class _Broken:
        def extract(self, task):
            raise RuntimeError("boom")

    gt = [
        _curve([(10.0, 0.2), (50.0, 0.5)]),
        _curve([(10.0, 60.0), (50.0, 150.0)], y_axis=YAxis.SECONDARY),
    ]
    task = ExtractionTask(
        image_bytes=b"i", x_range=(0.0, 100.0), y_range=(0.0, 1.0), y2_range=(0.0, 300.0)
    )
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=gt)]

    (result,) = evaluate_model_on_dataset(_Broken(), items, matcher=_matcher())

    assert result.error.startswith("RuntimeError")
    assert _point_cells(result) == (0.02, 0, 0, 4, 0.0)
