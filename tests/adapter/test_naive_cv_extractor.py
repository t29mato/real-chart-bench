"""Tests build real, minimal PNGs via pymupdf (drawing a straight colored
line) — not mocks — so the extractor is exercised against actual pixel
decoding, matching the project's convention for CV-adapter tests.
"""

import pymupdf
import pytest

from real_chart_bench.adapter.naive_cv_extractor import NaiveCvModelRunner
from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.usecase.model_runner import ExtractionTask


def _red_diagonal_line_png(size: int = 200) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page(width=size, height=size)
    page.draw_line(
        pymupdf.Point(10, 10), pymupdf.Point(size - 10, size - 10), color=(1, 0, 0), width=3
    )
    pix = page.get_pixmap()
    png = pix.tobytes("png")
    doc.close()
    return png


@pytest.fixture
def extractor():
    return NaiveCvModelRunner()


def test_extracts_a_curve_from_a_colored_line(extractor):
    image = _red_diagonal_line_png()
    task = ExtractionTask(image_bytes=image, x_range=(0, 10), y_range=(0, 10))

    curves = extractor.extract(task)

    assert len(curves) >= 1
    curve = curves[0]
    assert len(curve.x_values) > 0


def test_diagonal_line_roughly_traces_y_equals_x(extractor):
    # top-left of the image is high-y/low-x in data space (pixel y inverted);
    # a line from top-left to bottom-right of the pixmap should roughly
    # trace a downward-sloping data curve when y_range increases upward.
    image = _red_diagonal_line_png()
    task = ExtractionTask(image_bytes=image, x_range=(0, 10), y_range=(0, 10))

    (curve,) = extractor.extract(task)[:1]

    # first point (smallest x) should have a larger y than the last point
    # (this is a red line running from top-left to bottom-right of the
    # pixel image, i.e. NW->SE, which is a downward-sloping data curve)
    assert curve.y_values[0] > curve.y_values[-1]


def test_blank_image_yields_no_curves(extractor):
    doc = pymupdf.open()
    doc.new_page(width=100, height=100)
    pix = doc[0].get_pixmap()
    blank_png = pix.tobytes("png")
    doc.close()

    task = ExtractionTask(image_bytes=blank_png, x_range=(0, 10), y_range=(0, 10))

    assert extractor.extract(task) == []


def test_respects_log_x_scale():
    extractor = NaiveCvModelRunner()
    image = _red_diagonal_line_png()
    task = ExtractionTask(
        image_bytes=image, x_range=(1, 100), y_range=(0, 10), x_scale=ScaleType.LOG
    )

    curves = extractor.extract(task)

    assert len(curves) >= 1
    assert all(x > 0 for x in curves[0].x_values)


def test_respects_log_y_scale():
    # design §7.25: a colored line (unlike the black-line synthetic fixture
    # used elsewhere) lets naive-cv actually see the series, so this
    # exercises PixelCalibration's log-y pixel->data conversion end-to-end
    # rather than just documenting a known-blind-spot.
    extractor = NaiveCvModelRunner()
    image = _red_diagonal_line_png()
    task = ExtractionTask(
        image_bytes=image, x_range=(0, 10), y_range=(1, 100), y_scale=ScaleType.LOG
    )

    curves = extractor.extract(task)

    assert len(curves) >= 1
    assert all(y > 0 for y in curves[0].y_values)
    # NW->SE colored line -> downward-sloping data curve, same as the
    # linear-y case (test_diagonal_line_roughly_traces_y_equals_x).
    assert curves[0].y_values[0] > curves[0].y_values[-1]
