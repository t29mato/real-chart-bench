"""A plot with a y axis on each side (two quantities) keeps both calibrations as
separate candidate frames when asked (design pairing-automation.md 12.8)."""

from __future__ import annotations

import numpy as np

from real_chart_bench.adapter import auto_axis_calibration as A

W, H = 520, 300
X_TICKS = [40 + 33 * k for k in range(11)]
Y_TICKS = [250 - 46 * k for k in range(6)]


def _rgb() -> np.ndarray:
    g = np.full((H, W), 255, np.uint8)
    g[250:252, 40:371] = 0  # x axis
    g[20:251, 40:42] = 0  # left y axis
    g[20:251, 369:371] = 0  # right y axis
    g[20:22, 40:371] = 0  # top
    for x in X_TICKS:
        g[252:256, x] = 0
    for y in Y_TICKS:
        g[y, 36:40] = 0
    return np.stack([g] * 3, axis=2)


def _ocr(strip, psm=11, **_):
    if strip.shape[1] > strip.shape[0] and strip.shape[1] > 200:  # x strip
        sx0 = 7
        return [(str(300 + 20 * k), X_TICKS[k] - sx0 - 9.0, 5.0, X_TICKS[k] - sx0 + 9.0, 15.0,
                 90.0) for k in range(11)]
    scale = 50 if strip.shape[1] < 60 else 10  # left strip is narrow
    return [(str(scale * k), 5.0, y - 5.0, 28.0, y + 5.0, 90.0) for k, y in enumerate(Y_TICKS)]


def test_default_keeps_only_the_first_calibrating_side(monkeypatch):
    monkeypatch.setattr(A, "ocr_words", _ocr)
    cals = A.calibrate_frames(_rgb())
    assert [c.y_side for c in cals] == ["left"]


def test_both_y_sides_gives_a_frame_per_calibrating_side(monkeypatch):
    monkeypatch.setattr(A, "ocr_words", _ocr)
    cals = A.calibrate_frames(_rgb(), both_y_sides=True)
    assert [c.y_side for c in cals] == ["left", "right"]
    left, right = cals
    assert left.frame == right.frame
    assert abs(left.y_fit.px_to_value(Y_TICKS[1]) - 50) < 1
    assert abs(right.y_fit.px_to_value(Y_TICKS[1]) - 10) < 1
