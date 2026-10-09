"""Image normalisation around the automatic calibration (docs/design/local-model.md
「自動校正の改善」): white-on-black figures are read inverted, and very large
figures are read shrunk -- the fit found on the shrunk image is mapped back to
the original pixels."""

import numpy as np
import pytest

from real_chart_bench.domain.tick_calibration import (
    AxisFit,
    dark_background,
    scaled_fit,
    scaled_frame,
)


def test_white_page_is_not_dark_and_black_page_is():
    assert not dark_background(np.full((50, 80), 250, np.uint8))
    assert dark_background(np.full((50, 80), 10, np.uint8))


def test_dark_background_looks_at_the_majority_not_the_ink():
    g = np.full((100, 100), 255, np.uint8)
    g[:40] = 0  # a lot of black ink (thick bars) is still a white page
    assert not dark_background(g)
    assert dark_background(255 - g)


def test_scaled_fit_maps_shrunk_pixels_back():
    small = AxisFit("linear", slope=2.0, intercept=10.0, ticks=((10.0, 0.0), (30.0, 10.0)),
                    residual_px=0.5)
    big = scaled_fit(small, 4.0)  # the image was shrunk 4x
    assert big.value_to_px(0.0) == pytest.approx(40.0)
    assert big.value_to_px(10.0) == pytest.approx(120.0)
    assert big.ticks == ((40.0, 0.0), (120.0, 10.0))
    assert big.residual_px == pytest.approx(2.0)
    assert big.scale == "linear" and big.family == small.family


def test_scaled_fit_keeps_log_axes_and_none():
    fit = AxisFit("log", slope=-50.0, intercept=200.0, ticks=((200.0, 1.0), (150.0, 10.0)),
                  residual_px=0.1, family="pow10")
    big = scaled_fit(fit, 2.0)
    assert big.px_to_value(300.0) == pytest.approx(fit.px_to_value(150.0))
    assert big.family == "pow10"
    assert scaled_fit(None, 2.0) is None


def test_scaled_frame():
    assert scaled_frame((10, 20, 30, 40), 2.5) == (25.0, 50.0, 75.0, 100.0)
