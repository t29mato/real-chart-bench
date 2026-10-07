"""Automatic tick reading, v3 (方式D「v3: 検証の誤検知と道具の追加」): the
failure types of the v2 runs -- labels printed as a x 10^n, implausible log
fits over misread labels, reversed axes, minus signs and decimal points the
OCR dropped (recovered from the label's own ink), labels cut by the image
border (unreadable, not guessed)."""

from __future__ import annotations

import numpy as np
import pytest

from real_chart_bench.domain.axis_frame import label_glyphs
from real_chart_bench.domain.tick_calibration import (
    TickReading,
    apply_glyphs,
    fit_axis,
    parse_tick_label,
    touches_border,
)

V3 = {"allow_reversed": True, "plausible_log": True}


def plain(px_vals):
    return [(px, parse_tick_label(t)) for px, t in px_vals]


@pytest.mark.parametrize("text, value", [
    ("3.5x10^4", 35000.0), ("3.5×10^4", 35000.0), ("2.0x104", 20000.0),
    ("5.0x10-3", 0.005), ("1.5E4", 15000.0), ("2e-3", 0.002), ("1.0×10⁴", 10000.0),
])
def test_mantissa_times_power_of_ten(text, value):
    assert any(r.value == pytest.approx(value) for r in parse_tick_label(text))


def test_sci_labels_fit_a_linear_axis():
    labels = [(400, "0.0"), (330, "5.0x103"), (260, "1.0x104"), (190, "1.5x104"),
              (120, "2.0x104")]
    fit = fit_axis(plain(labels), direction=-1, **V3)
    assert fit.scale == "linear"
    assert fit.px_to_value(260) == pytest.approx(10000, rel=1e-6)


def test_no_log_fit_over_a_fraction_of_a_decade_with_odd_labels():
    # "3.0 2.5 2.0 1.5" read without decimal points ("14" a misread 15):
    # four labels fit a log line loosely, but nobody prints 30, 25, 20, 14
    # on a log axis spanning a third of a decade
    labels = [(109, "30"), (159.5, "25"), (210.5, "20"), (311.5, "14")]
    fit = fit_axis(plain(labels), direction=-1, **V3)
    assert fit is None or fit.scale == "linear"
    labels = [(240.5, "400"), (286.5, "340"), (378.5, "250"), (423.7, "206")]
    fit = fit_axis(plain(labels), direction=-1, **V3)
    assert fit is None or fit.scale == "linear"


def test_log_fit_still_found_for_decades_and_1_2_5_labels():
    dec = [(400, "1"), (300, "10"), (200, "100"), (100, "1000")]
    assert fit_axis(plain(dec), direction=-1).scale == "log"
    sub = [(400, "100"), (370, "200"), (330, "500")]
    sub = [(400, 100.0), (400 - 100 * np.log10(2), 200.0), (400 - 100 * np.log10(5), 500.0)]
    fit = fit_axis([(px, [TickReading("plain", v)]) for px, v in sub], direction=-1, **V3)
    assert fit.scale == "log"


def test_reversed_axis_is_read_when_nothing_else_fits():
    # an x axis printed 900 ... 300 left to right
    labels = [(100, "900"), (200, "800"), (300, "700"), (400, "600")]
    fit = fit_axis(plain(labels), direction=+1, **V3)
    assert fit is not None and fit.slope < 0
    assert fit.px_to_value(150) == pytest.approx(850)


def test_normal_direction_wins_a_tie():
    labels = [(100, "1"), (200, "2"), (300, "3")]
    assert fit_axis(plain(labels), direction=+1).slope > 0


def test_label_glyphs_minus_and_decimal_point():
    # "-1.5": a dash, a 1, a dot on the baseline, a 5 (20 px tall glyphs)
    ink = np.zeros((24, 60), bool)
    ink[11:13, 1:9] = True  # minus: short, thin, mid-height
    ink[2:22, 13:16] = True  # "1"
    ink[19:22, 20:23] = True  # "."
    ink[2:22, 27:37] = True  # "5" (a block will do)
    g = label_glyphs(ink)
    assert g["minus"] is True
    assert g["dot_after"] == 1  # after one digit glyph
    ink2 = np.zeros((24, 40), bool)
    ink2[2:22, 2:12] = True
    ink2[2:22, 16:26] = True  # "25", no dot, no minus
    g = label_glyphs(ink2)
    assert g["minus"] is False and g["dot_after"] is None


def test_apply_glyphs_restores_what_the_ocr_dropped():
    assert apply_glyphs("15", {"minus": True, "dot_after": 1}) == "-1.5"
    assert apply_glyphs("-1.5", {"minus": True, "dot_after": 1}) == "-1.5"
    assert apply_glyphs("30", {"minus": False, "dot_after": 1}) == "3.0"
    assert apply_glyphs("300", {"minus": False, "dot_after": None}) == "300"
    assert apply_glyphs("1.5", {"minus": False, "dot_after": 1}) == "1.5"
    assert apply_glyphs("abc", {"minus": True, "dot_after": None}) == "abc"


def test_dropped_minus_signs_recovered_fit_the_normal_direction():
    # y labels -30 ... -110 (top to bottom) read without their minus signs
    texts = [(16 + 66.5 * k, str(30 + 10 * k)) for k in range(9)]
    fixed = [(px, apply_glyphs(t, {"minus": True, "dot_after": None})) for px, t in texts]
    fit = fit_axis(plain(fixed), direction=-1, **V3)
    assert fit.slope < 0 and fit.px_to_value(16) == pytest.approx(-30)


def test_touches_border():
    assert touches_border((0, 10, 30, 20), (200, 100))
    assert touches_border((10, 85, 30, 99), (200, 100))
    assert not touches_border((10, 10, 30, 20), (200, 100))
