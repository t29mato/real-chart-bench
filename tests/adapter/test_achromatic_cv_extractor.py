"""Tests build real, minimal PNGs via pymupdf (drawing black/grey lines) --
not mocks -- matching the project's convention for CV-adapter tests (see
tests/adapter/test_naive_cv_extractor.py).

This baseline is the achromatic (luminance-based) counterpart to
NaiveCvModelRunner's hue-bucket baseline -- see the module docstring of
adapter/achromatic_cv_extractor.py for the failure it targets and its
documented limitations.
"""

import pymupdf
import pytest

from real_chart_bench.adapter.achromatic_cv_extractor import AchromaticCvModelRunner
from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.usecase.model_runner import ExtractionTask


def _diagonal_line_png(color: tuple[float, float, float], size: int = 200) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=size, height=size)
    page.draw_line(
        pymupdf.Point(10, 10), pymupdf.Point(size - 10, size - 10), color=color, width=3
    )
    pix = page.get_pixmap()
    png = pix.tobytes("png")
    doc.close()
    return png


def _black_diagonal_line_png(size: int = 200) -> bytes:
    return _diagonal_line_png(color=(0, 0, 0), size=size)


@pytest.fixture
def extractor():
    return AchromaticCvModelRunner()


def test_extracts_a_curve_from_a_black_line(extractor):
    image = _black_diagonal_line_png()
    task = ExtractionTask(image_bytes=image, x_range=(0, 10), y_range=(0, 10))

    curves = extractor.extract(task)

    assert len(curves) >= 1
    curve = curves[0]
    assert len(curve.x_values) > 0


def test_diagonal_line_roughly_traces_y_equals_x(extractor):
    # top-left of the image is high-y/low-x in data space (pixel y inverted);
    # a line from top-left to bottom-right of the pixmap should roughly
    # trace a downward-sloping data curve when y_range increases upward.
    image = _black_diagonal_line_png()
    task = ExtractionTask(image_bytes=image, x_range=(0, 10), y_range=(0, 10))

    (curve,) = extractor.extract(task)[:1]

    assert curve.y_values[0] > curve.y_values[-1]


def test_extracts_a_curve_from_a_mid_grey_line(extractor):
    # not pure black -- exercises the histogram-peak clustering on a grey
    # level away from 0, not just the darkest possible pixel value.
    image = _diagonal_line_png(color=(0.5, 0.5, 0.5))
    task = ExtractionTask(image_bytes=image, x_range=(0, 10), y_range=(0, 10))

    curves = extractor.extract(task)

    assert len(curves) >= 1
    assert len(curves[0].x_values) > 0


def test_two_distinct_grey_levels_yield_two_series(extractor):
    doc = pymupdf.open()
    page = doc.new_page(width=300, height=300)
    page.draw_line(pymupdf.Point(20, 280), pymupdf.Point(280, 20), color=(0, 0, 0), width=3)
    page.draw_line(pymupdf.Point(20, 20), pymupdf.Point(280, 280), color=(0.6, 0.6, 0.6), width=3)
    png = page.get_pixmap().tobytes("png")
    doc.close()

    task = ExtractionTask(image_bytes=png, x_range=(0, 10), y_range=(0, 10))
    curves = extractor.extract(task)

    assert len(curves) >= 2


def test_blank_image_yields_no_curves(extractor):
    doc = pymupdf.open()
    doc.new_page(width=100, height=100)
    pix = doc[0].get_pixmap()
    blank_png = pix.tobytes("png")
    doc.close()

    task = ExtractionTask(image_bytes=blank_png, x_range=(0, 10), y_range=(0, 10))

    assert extractor.extract(task) == []


def test_pure_color_image_yields_no_curves(extractor):
    # the converse blind spot of NaiveCvModelRunner: a fully saturated
    # (non-achromatic) line has no low-saturation pixels at all, so this
    # baseline should gracefully see nothing rather than crash or hallucinate
    # a series from anti-aliased edge pixels.
    image = _diagonal_line_png(color=(1, 0, 0))
    task = ExtractionTask(image_bytes=image, x_range=(0, 10), y_range=(0, 10))

    assert extractor.extract(task) == []


def test_respects_log_x_scale(extractor):
    image = _black_diagonal_line_png()
    task = ExtractionTask(
        image_bytes=image, x_range=(1, 100), y_range=(0, 10), x_scale=ScaleType.LOG
    )

    curves = extractor.extract(task)

    assert len(curves) >= 1
    assert all(x > 0 for x in curves[0].x_values)


def test_respects_log_y_scale(extractor):
    image = _black_diagonal_line_png()
    task = ExtractionTask(
        image_bytes=image, x_range=(0, 10), y_range=(1, 100), y_scale=ScaleType.LOG
    )

    curves = extractor.extract(task)

    assert len(curves) >= 1
    assert all(y > 0 for y in curves[0].y_values)
    assert curves[0].y_values[0] > curves[0].y_values[-1]


def test_curve_touching_axis_frame_does_not_crash(extractor):
    # a black axis-frame border sharing the same grey level as the curve
    # (a documented limitation -- see the module docstring) shouldn't crash
    # or corrupt the output shape, even when the curve touches the frame at
    # its corners.
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    page.draw_rect(pymupdf.Rect(10, 10, 190, 190), color=(0, 0, 0), width=2)
    page.draw_line(pymupdf.Point(10, 10), pymupdf.Point(190, 190), color=(0, 0, 0), width=2)
    png = page.get_pixmap().tobytes("png")
    doc.close()

    task = ExtractionTask(image_bytes=png, x_range=(0, 10), y_range=(0, 10))

    curves = extractor.extract(task)

    assert isinstance(curves, list)
    for curve in curves:
        assert len(curve.x_values) > 0
        assert all(v == v for v in curve.x_values)  # no NaN
        assert all(v == v for v in curve.y_values)  # no NaN


def test_curve_exactly_on_frame_edge_does_not_crash(extractor):
    # a degenerate boundary case: the "curve" pixels sit exactly along one
    # edge of the plot area (pixel_bbox has zero height there) -- must not
    # divide by zero or raise.
    doc = pymupdf.open()
    page = doc.new_page(width=200, height=200)
    page.draw_line(pymupdf.Point(10, 10), pymupdf.Point(190, 10), color=(0, 0, 0), width=2)
    png = page.get_pixmap().tobytes("png")
    doc.close()

    task = ExtractionTask(image_bytes=png, x_range=(0, 10), y_range=(0, 10))

    curves = extractor.extract(task)

    assert isinstance(curves, list)
