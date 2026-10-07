"""The alternative normalizing frame used only for comparing against Scatteract's
criterion (paper §3.5.1). The benchmark's own metric normalizes by the axis range
(``axis_frame_for_task``); Scatteract normalizes by the span of the true points.
"""

import math

import pytest

from real_chart_bench.domain.curve import Curve, ScaleType
from real_chart_bench.usecase.evaluate_dataset import ground_truth_frame


def curve(xs, ys, scale=ScaleType.LINEAR):
    return Curve(x_values=xs, y_values=ys, x_scale=scale)


def test_frame_spans_the_bounding_box_of_every_ground_truth_curve():
    frame = ground_truth_frame(
        [curve((10.0, 20.0), (1.0, 2.0)), curve((5.0, 40.0), (0.5, 3.0))],
        x_scale=ScaleType.LINEAR,
        y_scale=ScaleType.LINEAR,
    )
    assert frame.x_range == (5.0, 40.0)
    assert frame.y_range == (0.5, 3.0)


def test_points_at_the_extremes_normalize_to_zero_and_one():
    frame = ground_truth_frame(
        [curve((100.0, 300.0), (2.0, 6.0))], x_scale=ScaleType.LINEAR, y_scale=ScaleType.LINEAR
    )
    normalized = frame.normalize(curve((100.0, 300.0), (2.0, 6.0)))
    assert normalized[0].tolist() == [0.0, 0.0]
    assert normalized[1].tolist() == [1.0, 1.0]


def test_a_log_axis_spans_the_log10_extent():
    frame = ground_truth_frame(
        [curve((1.0, 1000.0), (1.0, 2.0), scale=ScaleType.LOG)],
        x_scale=ScaleType.LOG,
        y_scale=ScaleType.LINEAR,
    )
    assert frame.x_range == (1.0, 1000.0)
    assert frame.x_scale is ScaleType.LOG
    mid = frame.normalize(curve((10.0,), (1.5,), scale=ScaleType.LOG))
    assert math.isclose(mid[0][0], 1 / 3, abs_tol=1e-12)


def test_a_flat_axis_keeps_a_usable_span_instead_of_dividing_by_zero():
    """Every y value equal: the span is 0, so the frame must fall back rather
    than produce an infinite normalized distance."""
    frame = ground_truth_frame(
        [curve((1.0, 2.0), (7.0, 7.0))],
        x_scale=ScaleType.LINEAR,
        y_scale=ScaleType.LINEAR,
        fallback_y_range=(0.0, 10.0),
    )
    assert frame.y_range == (0.0, 10.0)


def test_a_flat_axis_without_a_fallback_is_rejected():
    with pytest.raises(ValueError):
        ground_truth_frame(
            [curve((1.0, 2.0), (7.0, 7.0))],
            x_scale=ScaleType.LINEAR,
            y_scale=ScaleType.LINEAR,
        )


def test_no_curves_is_rejected():
    with pytest.raises(ValueError):
        ground_truth_frame([], x_scale=ScaleType.LINEAR, y_scale=ScaleType.LINEAR)
