"""v3 drops tick words that touch the image border (a cut label reads as another
number). When that leaves an axis unreadable, the words are kept for a second
read (design pairing-automation.md 12.7): 28331's x labels "300...500" come as
one OCR word whose box reaches the image's right edge."""

from __future__ import annotations

import numpy as np

from real_chart_bench.adapter import auto_axis_calibration as A

W, H = 380, 300
FRAME = (40.0, 20.0, 370.0, 250.0)
X_TICKS = [40 + 33 * k for k in range(11)]
Y_TICKS = [250 - 46 * k for k in range(6)]


def _image() -> np.ndarray:
    g = np.full((H, W), 255, np.uint8)
    g[250:252, 40:371] = 0  # x axis
    g[20:251, 40:42] = 0  # y axis
    for x in X_TICKS:
        g[252:256, x] = 0
        g[262:270, x - 9 : x + 9] = 0  # label ink
    for y in Y_TICKS:
        g[y, 36:40] = 0
    return g


def _fake_ocr(strip, psm=11, **_):
    if strip.shape[1] > strip.shape[0]:  # x strip
        return [("300320340360380400420440460480500", 13.0, 5.0, float(strip.shape[1]), 15.0,
                 90.0)]
    return [(str(50 * k), 5.0, y - 5.0, 28.0, y + 5.0, 90.0) for k, y in enumerate(Y_TICKS)]


def test_x_axis_with_a_border_touching_merged_word_is_read(monkeypatch):
    monkeypatch.setattr(A, "ocr_words", _fake_ocr)
    g = _image()
    dark = g < A.DARK
    cal = A.calibrate_frame(g, dark, FRAME, v3=True, split_merged=True)
    assert cal["x_fit"] is not None
    assert len(cal["x_fit"].ticks) >= 9
    assert abs(cal["x_fit"].px_to_value(40) - 300) < 1


def test_without_v3_nothing_changes(monkeypatch):
    monkeypatch.setattr(A, "ocr_words", _fake_ocr)
    g = _image()
    cal = A.calibrate_frame(g, g < A.DARK, FRAME, v3=False, split_merged=True)
    assert cal["x_fit"] is not None  # v3-off never dropped it
