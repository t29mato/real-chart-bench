from __future__ import annotations

import numpy as np

from real_chart_bench.adapter.auto_axis_calibration import ImageCalibration
from real_chart_bench.adapter.pairing_scorer import ink_mask, score_figure_on_frame
from real_chart_bench.adapter.starrydata_figure_gt import CurveGt, FigureGt
from real_chart_bench.domain.tick_calibration import AxisFit

FRAME = (50.0, 50.0, 250.0, 250.0)


def _cal() -> ImageCalibration:
    # x: value 0..100 -> px 50..250; y: value 0..100 -> px 250..50
    x_fit = AxisFit("linear", 2.0, 50.0, (), 0.0)
    y_fit = AxisFit("linear", -2.0, 250.0, (), 0.0)
    return ImageCalibration(FRAME, x_fit, y_fit, 1)


def _figure(xs, ys, unit_x="1", unit_y="1") -> FigureGt:
    curve = CurveGt("1-1-0", "A", tuple(xs), tuple(ys))
    return FigureGt("1", "1", "1", "X", unit_x, "Y", unit_y, (curve,), 0)


def _image_with_markers(points) -> np.ndarray:
    rgb = np.full((300, 300, 3), 255, np.uint8)
    for x, y in points:
        rgb[int(y) - 3 : int(y) + 4, int(x) - 3 : int(x) + 4] = 0
    return rgb


XS = [20, 35, 50, 65, 80]
YS = [30, 60, 40, 80, 55]
DRAWN = [(50 + 2 * x, 250 - 2 * y) for x, y in zip(XS, YS, strict=True)]


def test_ground_truth_drawn_in_the_frame_scores_high_with_low_null():
    ink = ink_mask(_image_with_markers(DRAWN))
    scored = score_figure_on_frame("f", _figure(XS, YS), _cal(), ink)
    assert scored is not None
    assert scored.pair.hit >= 0.99
    assert scored.pair.null < 0.3
    assert len(scored.points_px[0]) == len(XS)


def test_other_figures_values_land_on_blank_paper():
    other_ys = [90, 20, 85, 15, 95]
    scored = score_figure_on_frame(
        "f", _figure(XS, other_ys), _cal(), ink_mask(_image_with_markers(DRAWN))
    )
    assert scored is None or scored.pair.hit < 0.5


def test_values_far_outside_the_frame_in_every_known_form_give_none():
    scored = score_figure_on_frame(
        "f", _figure([1e9, 2e9], [1e9, 2e9]), _cal(), ink_mask(_image_with_markers(DRAWN))
    )
    assert scored is None


def test_ink_mask_ignores_white_and_keeps_dark_and_coloured_pixels():
    rgb = np.full((4, 4, 3), 255, np.uint8)
    rgb[0, 0] = (0, 0, 0)
    rgb[1, 1] = (255, 0, 0)
    rgb[2, 2] = (240, 240, 240)
    ink = ink_mask(rgb)
    assert ink[0, 0] and ink[1, 1]
    assert not ink[2, 2] and not ink[3, 3]
