"""calibrate_image(normalize=True) (docs/design/local-model.md「自動校正の改善」):
a white-on-black figure is read inverted, and a figure larger than max_side
is read shrunk with the result mapped back to the original pixels. Off by
default, so every existing caller is unchanged."""

from __future__ import annotations

import numpy as np

from real_chart_bench.adapter import auto_axis_calibration as A
from real_chart_bench.domain.tick_calibration import AxisFit

from .test_auto_axis_calibration_dual_y import Y_TICKS, _ocr, _rgb


def test_inverted_figure_reads_like_the_original(monkeypatch):
    monkeypatch.setattr(A, "ocr_words", _ocr)
    plain = A.calibrate_image(_rgb())
    assert plain is not None and plain.ok
    inverted = 255 - _rgb()
    assert not (A.calibrate_image(inverted) or A.ImageCalibration((0, 0, 0, 0), None, None, 0)).ok
    c = A.calibrate_image(inverted, normalize=True)
    assert c is not None and c.ok
    assert c.frame == plain.frame
    assert abs(c.y_fit.px_to_value(Y_TICKS[1]) - plain.y_fit.px_to_value(Y_TICKS[1])) < 1e-6


def test_white_figure_is_untouched_by_normalize(monkeypatch):
    monkeypatch.setattr(A, "ocr_words", _ocr)
    a, b = A.calibrate_image(_rgb()), A.calibrate_image(_rgb(), normalize=True)
    assert a.frame == b.frame and a.x_fit == b.x_fit and a.y_fit == b.y_fit


def test_large_figure_is_read_shrunk_and_mapped_back(monkeypatch):
    seen = {}

    def fake(rgb, **kw):
        seen["shape"] = rgb.shape
        fit = AxisFit("linear", 1.0, 0.0, ((10.0, 10.0), (90.0, 90.0)), 0.5)
        return A.ImageCalibration((10.0, 5.0, 90.0, 80.0), fit, fit, 1)

    monkeypatch.setattr(A, "_calibrate_unnormalized", fake)
    big = np.full((1000, 4000, 3), 255, np.uint8)
    c = A.calibrate_image(big, normalize=True, max_side=2000)
    assert seen["shape"][:2] == (500, 2000)
    assert c.frame == (20.0, 10.0, 180.0, 160.0)
    assert c.x_fit.value_to_px(50.0) == 100.0
    small = np.full((100, 400, 3), 255, np.uint8)
    A.calibrate_image(small, normalize=True, max_side=2000)
    assert seen["shape"][:2] == (100, 400)  # never enlarged


def test_small_figure_is_read_enlarged_when_min_side_is_set(monkeypatch):
    seen = {}

    def fake(rgb, **kw):
        seen["shape"] = rgb.shape
        fit = AxisFit("linear", 1.0, 0.0, ((10.0, 10.0), (90.0, 90.0)), 0.5)
        return A.ImageCalibration((10.0, 5.0, 90.0, 80.0), fit, fit, 1)

    monkeypatch.setattr(A, "_calibrate_unnormalized", fake)
    small = np.full((150, 300, 3), 255, np.uint8)
    c = A.calibrate_image(small, normalize=True, max_side=900, min_side=600)
    assert seen["shape"][:2] == (300, 600)
    assert c.frame == (5.0, 2.5, 45.0, 40.0)
    assert c.x_fit.value_to_px(50.0) == 25.0
