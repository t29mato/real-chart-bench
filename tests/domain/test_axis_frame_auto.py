"""方式C (docs/design/local-model.md): frames whose y axis is drawn on the
right, and splitting a strip of tick labels into one box per label (Tesseract
merges tightly spaced labels such as "300320340" into one word)."""

import numpy as np
import pytest

from real_chart_bench.domain.axis_frame import (
    detect_right_axis_frames,
    mirror_frame,
)


class TestRightAxisFrames:
    def test_frame_with_y_axis_on_the_right(self):
        img = np.zeros((300, 400), dtype=bool)
        img[250, 20:351] = True  # x axis
        img[20:251, 350] = True  # y axis on the right
        (f,) = detect_right_axis_frames(img)
        # (x0, y0, x1, y1) in the image: x1 is the right-hand y axis
        assert f == pytest.approx((20, 20, 350, 250), abs=1.5)

    def test_mirror_frame_is_an_involution(self):
        f = (20.0, 5.0, 350.0, 250.0)
        assert mirror_frame(mirror_frame(f, 400), 400) == f
        assert mirror_frame(f, 400) == (49.0, 5.0, 379.0, 250.0)


class TestSplitAtWidestGaps:
    def test_k_labels_from_the_k_minus_1_widest_gaps(self):
        from real_chart_bench.domain.axis_frame import split_at_widest_gaps

        img = np.zeros((12, 120), dtype=bool)
        for start in (5, 45, 85):
            for k in range(3):
                img[2:10, start + 5 * k : start + 5 * k + 4] = True
        assert split_at_widest_gaps(img, 3) == [(5, 19), (45, 59), (85, 99)]

    def test_none_when_the_cut_is_not_clear(self):
        from real_chart_bench.domain.axis_frame import split_at_widest_gaps

        # four labels on the line, but only three expected: the 3rd-widest
        # gap is as wide as the cut ones, so the line holds more labels
        img = np.zeros((12, 200), dtype=bool)
        for start in (5, 45, 85, 125):
            img[2:10, start : start + 14] = True
        assert split_at_widest_gaps(img, 3, min_ratio=1.5) is None
        assert split_at_widest_gaps(img, 4, min_ratio=1.5) is not None

    def test_none_when_too_few_gaps(self):
        from real_chart_bench.domain.axis_frame import split_at_widest_gaps

        img = np.zeros((12, 40), dtype=bool)
        img[2:10, 5:15] = True
        assert split_at_widest_gaps(img, 2) is None


class TestExponentSpan:
    def test_raised_glyphs_right_of_the_base(self):
        from real_chart_bench.domain.axis_frame import exponent_span

        ink = np.zeros((30, 40), dtype=bool)
        ink[10:28, 2:6] = True  # "1"
        ink[10:28, 8:14] = True  # "0"
        ink[3:15, 16:19] = True  # raised "-" / digit
        ink[3:15, 21:25] = True
        assert exponent_span(ink) == (16, 3, 25, 15)

    def test_no_raised_glyphs(self):
        from real_chart_bench.domain.axis_frame import exponent_span

        ink = np.zeros((30, 40), dtype=bool)
        ink[10:28, 2:6] = True
        ink[10:28, 8:14] = True
        assert exponent_span(ink) is None

    def test_empty(self):
        from real_chart_bench.domain.axis_frame import exponent_span

        assert exponent_span(np.zeros((5, 5), dtype=bool)) is None
