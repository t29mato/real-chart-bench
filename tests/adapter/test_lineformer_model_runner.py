"""LineFormer runs in its own Python 3.10 + CUDA env (scripts/eval/lineformer/
worker.py), so the adapter under test never imports torch: it replays the
worker's raw pixel-space output. These tests pin the replay contract --
lookup by image content, the pixel->data mapping the original Colab notebook
used (full image frame), and loud failure on a missing or errored figure.
"""

import hashlib

import pytest

from real_chart_bench.adapter.lineformer_model_runner import (
    LineFormerPrediction,
    PrecomputedLineFormerModelRunner,
    image_key,
)
from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.usecase.model_runner import ExtractionTask

_IMAGE = b"fake-png-bytes"


def _runner(**prediction_kwargs) -> PrecomputedLineFormerModelRunner:
    defaults = dict(
        figure_id="1-2",
        image_key=image_key(_IMAGE),
        width=200,
        height=100,
        series=(),
        error=None,
    )
    defaults.update(prediction_kwargs)
    return PrecomputedLineFormerModelRunner([LineFormerPrediction(**defaults)])


def test_image_key_is_the_sha256_of_the_image_bytes():
    assert image_key(_IMAGE) == hashlib.sha256(_IMAGE).hexdigest()


def test_maps_pixels_over_the_full_image_frame_with_y_inverted():
    runner = _runner(series=(((0.0, 100.0), (200.0, 0.0), (100.0, 50.0)),))
    task = ExtractionTask(image_bytes=_IMAGE, x_range=(0, 10), y_range=(0, 1))

    (curve,) = runner.extract(task)

    # sorted by x; bottom-left pixel -> (x0, y0), top-right -> (x1, y1)
    assert curve.x_values == pytest.approx((0.0, 5.0, 10.0))
    assert curve.y_values == pytest.approx((0.0, 0.5, 1.0))


def test_log_axes_interpolate_in_log_space():
    runner = _runner(series=(((100.0, 50.0),),))
    task = ExtractionTask(
        image_bytes=_IMAGE,
        x_range=(1, 100),
        y_range=(10, 1000),
        x_scale=ScaleType.LOG,
        y_scale=ScaleType.LOG,
    )

    (curve,) = runner.extract(task)

    assert curve.x_values == pytest.approx((10.0,))
    assert curve.y_values == pytest.approx((100.0,))
    assert curve.x_scale is ScaleType.LOG


def test_one_curve_per_series_and_empty_series_are_dropped():
    runner = _runner(series=(((0.0, 0.0),), (), ((10.0, 10.0), (20.0, 20.0))))
    task = ExtractionTask(image_bytes=_IMAGE, x_range=(0, 1), y_range=(0, 1))

    curves = runner.extract(task)

    assert [len(c.x_values) for c in curves] == [1, 2]
    assert [c.series_label for c in curves] == ["series_0", "series_2"]


def test_worker_error_is_raised_so_the_harness_scores_the_figure_zero():
    runner = _runner(error="CUDA out of memory")
    task = ExtractionTask(image_bytes=_IMAGE, x_range=(0, 1), y_range=(0, 1))

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        runner.extract(task)


def test_an_image_the_worker_never_saw_is_an_error_not_an_empty_answer():
    runner = _runner()
    task = ExtractionTask(image_bytes=b"other", x_range=(0, 1), y_range=(0, 1))

    with pytest.raises(KeyError):
        runner.extract(task)


def test_round_trips_through_the_worker_jsonl_record():
    record = {
        "figure_id": "1-2",
        "image_key": "abc",
        "width": 200,
        "height": 100,
        "series": [[[1, 2], [3, 4]]],
        "error": None,
        "seconds": 0.5,
    }

    prediction = LineFormerPrediction.from_record(record)

    assert prediction.series == (((1.0, 2.0), (3.0, 4.0)),)
    assert prediction.error is None
