"""Plot frames and tick marks found on a binarised figure image (pure numpy):
the pixel half of the no-LLM axis calibration (docs/design/local-model.md)."""

import numpy as np
import pytest

from real_chart_bench.domain.axis_frame import detect_axis_frames, detect_ticks


def _axes(img, x0, y0, x1, y1, *, tick_every=50, tick_len=5, thickness=1, box=False):
    img[y1 : y1 + thickness, x0 : x1 + 1] = True  # x axis
    img[y0 : y1 + thickness, x0 : x0 + thickness] = True  # y axis
    if box:
        img[y0 : y0 + thickness, x0 : x1 + 1] = True
        img[y0 : y1 + 1, x1 : x1 + thickness] = True
    for x in range(x0, x1 + 1, tick_every):
        img[y1 + thickness : y1 + thickness + tick_len, x] = True  # outward, below
    for y in range(y1, y0 - 1, -tick_every):
        img[y, x0 - tick_len : x0] = True  # outward, left


class TestDetectAxisFrames:
    def test_single_frame(self):
        img = np.zeros((300, 400), dtype=bool)
        _axes(img, 50, 20, 380, 250)
        (f,) = detect_axis_frames(img)
        assert f == pytest.approx((50, 20, 380, 250), abs=1.5)

    def test_thick_boxed_frame(self):
        img = np.zeros((300, 400), dtype=bool)
        _axes(img, 50, 20, 380, 250, thickness=3, box=True)
        (f,) = detect_axis_frames(img)
        assert f == pytest.approx((51, 20, 380, 251), abs=2)

    def test_two_panels_side_by_side(self):
        img = np.zeros((300, 800), dtype=bool)
        _axes(img, 50, 20, 370, 250)
        _axes(img, 450, 20, 770, 250)
        frames = sorted(detect_axis_frames(img))
        assert len(frames) == 2
        assert frames[0][0] == pytest.approx(50, abs=1.5)
        assert frames[1][0] == pytest.approx(450, abs=1.5)

    def test_text_lines_are_not_frames(self):
        img = np.zeros((300, 400), dtype=bool)
        for y in range(20, 280, 12):
            for x in range(10, 390, 7):
                img[y : y + 6, x : x + 4] = True
        assert detect_axis_frames(img) == []

    def test_small_frames_are_ignored(self):
        img = np.zeros((300, 400), dtype=bool)
        _axes(img, 50, 200, 90, 240, tick_every=20)
        assert detect_axis_frames(img) == []


class TestDetectTicks:
    def test_outward_ticks(self):
        img = np.zeros((300, 400), dtype=bool)
        _axes(img, 50, 20, 380, 250)
        xt, yt = detect_ticks(img, (50, 20, 380, 250))
        assert xt == pytest.approx([100, 150, 200, 250, 300, 350], abs=1)
        assert yt == pytest.approx([50, 100, 150, 200], abs=1)

    def test_inward_ticks(self):
        img = np.zeros((300, 400), dtype=bool)
        _axes(img, 50, 20, 380, 250, tick_len=0)
        for x in range(100, 381, 50):
            img[244:250, x] = True
        for y in range(200, 19, -50):
            img[y, 51:57] = True
        xt, yt = detect_ticks(img, (50, 20, 380, 250))
        assert xt == pytest.approx([100, 150, 200, 250, 300, 350], abs=1)
        assert yt == pytest.approx([50, 100, 150, 200], abs=1)

    def test_a_curve_crossing_the_axis_is_not_a_tick(self):
        img = np.zeros((300, 400), dtype=bool)
        _axes(img, 50, 20, 380, 250, tick_len=0)
        img[150:250, 200:240] = True  # a filled bar standing on the axis
        xt, _ = detect_ticks(img, (50, 20, 380, 250))
        assert xt == []


class TestOutwardTickExtent:
    def test_extent_of_outward_ticks(self):
        from real_chart_bench.domain.axis_frame import outward_tick_extent

        img = np.zeros((300, 400), dtype=bool)
        _axes(img, 50, 20, 380, 250, tick_len=7)
        assert outward_tick_extent(img, (50, 20, 380, 250)) == (7, 7)

    def test_inward_ticks_have_no_outward_extent(self):
        from real_chart_bench.domain.axis_frame import outward_tick_extent

        img = np.zeros((300, 400), dtype=bool)
        _axes(img, 50, 20, 380, 250, tick_len=0)
        assert outward_tick_extent(img, (50, 20, 380, 250)) == (0, 0)


class TestPhotographsAreNotPlots:
    def test_dark_filled_region_is_not_a_frame(self):
        img = np.zeros((300, 400), dtype=bool)
        _axes(img, 50, 20, 380, 250)
        img[30:240, 60:370] = True  # a micrograph's dark interior
        assert detect_axis_frames(img) == []

    def test_thick_bars_are_not_axis_lines(self):
        img = np.zeros((300, 400), dtype=bool)
        img[240:260, 50:380] = True
        img[20:260, 50:70] = True
        assert detect_axis_frames(img) == []
