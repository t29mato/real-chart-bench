"""verify v3 (方式D「v3: 検証の誤検知と道具の追加」): the false alarms seen in
the v2 runs -- line segments, error-bar caps, fit-curve fragments and
markers split by a white cross counted as missed markers; minor ticks
with any number of subdivisions, uneven label steps and stray tick marks
counted against the calibration -- must not raise a redo, while true
misses still do."""

from __future__ import annotations

import numpy as np
import pytest

from real_chart_bench.domain.verification import (
    blob_shape,
    calibration_signals,
    group_pieces,
    is_stroke,
    verdict,
    verify,
)

FRAME = [20.0, 20.0, 380.0, 280.0]
RED = (220, 30, 30)
BLUE = (30, 30, 220)
WHITE = (255, 255, 255)
CENTRES = [(60, 200), (110, 150), (160, 170), (210, 110), (260, 130), (310, 80), (350, 100)]


def canvas(w=400, h=300):
    img = np.full((h, w, 3), 255, np.uint8)
    x0, y0, x1, y1 = (int(v) for v in FRAME)
    img[y0, x0:x1 + 1] = 0
    img[y1, x0:x1 + 1] = 0
    img[y0:y1 + 1, x0] = 0
    img[y0:y1 + 1, x1] = 0
    return img


def disc(img, cx, cy, r, color):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    img[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = color


def ring(img, cx, cy, r, color, width=1.5):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    d = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    img[np.abs(d - r) <= width / 2] = color


def segment(img, a, b, width, color):
    h, w = img.shape[:2]
    yy, xx = np.mgrid[:h, :w]
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    t = np.clip(((xx - ax) * vx + (yy - ay) * vy) / (vx * vx + vy * vy), 0, 1)
    d = np.hypot(xx - (ax + t * vx), yy - (ay + t * vy))
    img[d <= width / 2] = color


def series(pts, label="s"):
    return {"label": label, "x": [p[0] for p in pts], "y": [p[1] for p in pts]}


def gapline(r=6, lw=3.0, gap=4.0):
    """Origin style: thick segments that stop `gap` px short of each marker."""
    img = canvas()
    for (ax, ay), (bx, by) in zip(CENTRES, CENTRES[1:], strict=False):
        d = np.hypot(bx - ax, by - ay)
        ux, uy = (bx - ax) / d, (by - ay) / d
        s = r + gap
        segment(img, (ax + ux * s, ay + uy * s), (bx - ux * s, by - uy * s), lw, RED)
    for cx, cy in CENTRES:
        disc(img, cx, cy, r, RED)
    return img


# ------------------------------------------------------------------ shape


def test_blob_shape_of_a_disc_a_segment_and_a_plus():
    img = np.zeros((60, 60), bool)
    yy, xx = np.mgrid[:60, :60]
    img[(xx - 15) ** 2 + (yy - 15) ** 2 <= 36] = True
    s = blob_shape(img[5:26, 5:26])
    assert s["elongation"] == pytest.approx(1.0, abs=0.15)
    assert s["thickness"] >= 0.8 * s["side"]
    seg = np.zeros((10, 30), bool)
    seg[4:7, 2:28] = True
    s = blob_shape(seg)
    assert s["elongation"] > 5 and s["thickness"] <= 4
    plus = np.zeros((15, 15), bool)
    plus[6:9, :] = True
    plus[:, 6:9] = True
    s = blob_shape(plus)
    assert s["elongation"] == pytest.approx(1.0, abs=0.15)
    assert s["thickness"] < 0.4 * s["side"]


def test_is_stroke_segments_but_not_markers():
    assert is_stroke({"elongation": 6.0, "thickness": 3.0, "side": 26})
    assert is_stroke({"elongation": 2.0, "thickness": 3.0, "side": 20})  # thin and long-ish
    assert not is_stroke({"elongation": 1.0, "thickness": 3.0, "side": 15})  # a "+" marker
    assert not is_stroke({"elongation": 1.1, "thickness": 12.0, "side": 13})  # a filled disc
    assert not is_stroke({"elongation": 1.6, "thickness": 9.0, "side": 13})  # a triangle


def test_group_pieces_merges_quarters_not_neighbours():
    # four quarters of one 20 px marker split by a 3 px white cross
    quarters = [((0, 0, 8, 8), 81), ((12, 0, 20, 8), 81), ((0, 12, 8, 20), 81),
                ((12, 12, 20, 20), 81)]
    assert group_pieces(quarters, gap=3) == [[0, 1, 2, 3]]
    # two whole markers side by side, 3 px apart: an elongated union
    pair = [((0, 0, 20, 20), 441), ((24, 0, 44, 20), 441)]
    assert sorted(group_pieces(pair, gap=4)) == [[0], [1]]
    # two whole discs touching diagonally: the union is mostly empty
    diag = [((0, 0, 20, 20), 314), ((23, 23, 43, 43), 314)]
    assert sorted(group_pieces(diag, gap=4)) == [[0], [1]]


# ------------------------------------------------------------------ false alarms


def test_line_segments_between_markers_are_not_missed_markers():
    out = verify(gapline(), [series(CENTRES)], FRAME)
    assert out["unexplained"]["n"] == 0
    assert out["lookalikes"]["n"] == 0
    assert verdict(out)["accept"]


def test_thick_short_segments_with_small_markers():
    out = verify(gapline(r=4, lw=4.0, gap=3.0), [series(CENTRES)], FRAME)
    assert out["unexplained"]["n"] == 0


def test_error_bar_caps_are_not_missed_markers():
    img = canvas()
    for cx, cy in CENTRES:
        disc(img, cx, cy, 6, RED)
        segment(img, (cx, cy - 22), (cx, cy - 9), 1.5, RED)  # bar above, detached
        segment(img, (cx - 6, cy - 22), (cx + 6, cy - 22), 2.0, RED)  # its cap
        segment(img, (cx - 6, cy + 20), (cx + 6, cy + 20), 2.0, RED)  # a lone cap below
    out = verify(img, [series(CENTRES)], FRAME)
    assert out["unexplained"]["n"] == 0
    assert verdict(out)["accept"]


def test_fit_curve_fragments_are_not_missed_markers():
    img = canvas()
    for cx, cy in CENTRES:
        disc(img, cx, cy, 6, RED)
    # a dashed fit curve (parabola) passing near the points
    xs = np.linspace(40, 370, 400)
    ys = 220 - 0.9 * (xs - 40) + 0.0012 * (xs - 40) ** 2
    for k in range(0, 400, 20):
        for j in range(k, min(k + 12, 399)):
            segment(img, (xs[j], ys[j]), (xs[j + 1], ys[j + 1]), 2.0, BLUE)
    out = verify(img, [series(CENTRES)], FRAME)
    assert out["unexplained"]["n"] == 0


def test_marker_split_by_a_white_cross_is_one_marker():
    img = canvas()
    for cx, cy in CENTRES:
        disc(img, cx, cy, 8, BLUE)
        img[cy - 1:cy + 2, cx - 9:cx + 10] = WHITE
        img[cy - 9:cy + 10, cx - 1:cx + 2] = WHITE
    out = verify(img, [series(CENTRES)], FRAME)
    assert out["unexplained"]["n"] == 0
    assert out["marker_diameter_px"] >= 14  # the whole marker, not a quarter
    assert verdict(out)["accept"]
    # one left out is counted once, not four times
    out = verify(img, [series(CENTRES[:-1])], FRAME)
    assert out["unexplained"]["n"] == 1


# ------------------------------------------------------------------ true misses kept


def test_missed_markers_among_line_segments_are_still_counted():
    out = verify(gapline(), [series(CENTRES[:4])], FRAME)
    assert out["unexplained"]["n"] == 3
    assert not verdict(out)["accept"]


def test_missed_hollow_and_plus_markers_are_still_counted():
    img = canvas()
    for cx, cy in CENTRES[:4]:
        ring(img, cx, cy, 6, RED)
    for cx, cy in CENTRES[4:]:
        img[cy - 1:cy + 2, cx - 7:cx + 8] = BLUE
        img[cy - 7:cy + 8, cx - 1:cx + 2] = BLUE
    out = verify(img, [series(CENTRES[:2])], FRAME)
    assert out["unexplained"]["n"] == 5


# ------------------------------------------------------------------ calibration


def axis(ticks, scale="linear"):
    return {"scale": scale, "ticks": [list(t) for t in ticks]}


def test_minor_ticks_with_any_number_of_subdivisions_are_on_the_grid():
    # labels 300..550 every 138 px, Origin's "10 minor ticks" = 11 intervals
    ticks = [(20 + 138 * k, 300 + 50 * k) for k in range(3)]
    marks = [20 + 138 / 11 * j for j in range(23)]
    s = calibration_signals({"x": axis(ticks), "y": axis([(280, 0), (150, 1), (20, 2)])},
                            {"x": marks, "y": [280, 150, 20]})
    assert s["x"]["marks_on_grid"] >= 0.95


def test_uneven_label_steps_are_on_one_grid():
    # 500 and 550 not read: steps 50, 150, 100, 100, 50
    vals = [650, 600, 450, 350, 250, 200]
    ticks = [(20 + (650 - v) * 0.5, v) for v in vals]
    s = calibration_signals({"y": axis(ticks), "x": axis([(20, 0), (200, 1), (380, 2)])},
                            None)
    assert s["y"]["grid_dev"] == pytest.approx(0, abs=1e-9)
    # a misread label still is off the grid
    bad = [(20 + (650 - v) * 0.5, 470 if v == 450 else v) for v in vals]
    s = calibration_signals({"y": axis(bad), "x": axis([(20, 0), (200, 1), (380, 2)])}, None)
    assert s["y"]["grid_dev"] > 0.2


def test_missing_label_between_read_ones_keeps_the_marks_on_the_grid():
    # labels 3, 2, 0 (the 1 not read), a mark at every unit
    ticks = [(102.5, 3), (172.5, 2), (313, 0)]
    marks = [102.5, 137.5, 172.5, 208, 243, 278, 313, 348, 383.5]
    s = calibration_signals({"y": axis(ticks), "x": axis([(20, 0), (200, 1), (380, 2)])},
                            {"y": marks, "x": []})
    assert s["y"]["grid_dev"] == pytest.approx(0, abs=1e-9)
    assert s["y"]["marks_on_grid"] >= 0.9


def test_log_marks_with_labels_a_pixel_or_two_off():
    # decades every 116 px, labels centred 2 px off their ticks
    dec = 116.0
    ticks = [(20 + 2 + dec * k, 10.0 ** (5 - k)) for k in range(5)]
    marks = [20 + dec * (k + 1 - np.log10(m)) for k in range(4) for m in range(1, 10)]
    s = calibration_signals({"y": axis(ticks, "log"), "x": axis([(20, 0), (200, 1), (380, 2)])},
                            {"y": sorted(marks), "x": []})
    assert s["y"]["marks_on_grid"] >= 0.9


def test_stray_marks_away_from_the_labels_are_not_used():
    ticks = [(111, 2), (301, 0), (497, -2), (681, -4), (1061, -8)]
    s = calibration_signals({"y": axis(ticks), "x": axis([(20, 0), (200, 1), (380, 2)])},
                            {"y": [314, 327, 372, 378.5], "x": []})
    assert s["y"]["marks_on_grid"] is None


def test_log_axis_read_as_linear_is_still_flagged():
    # exponent labels 10^0, 10^1, 10^2 read as 0, 1, 2 on a linear scale;
    # the minor marks follow the log pattern
    marks = sorted(280 - 130 * np.log10(m) - 130 * k for k in range(2) for m in range(1, 10))
    s = calibration_signals({"y": axis([(280, 0), (150, 1), (20, 2)]),
                             "x": axis([(20, 0), (200, 1), (380, 2)])},
                            {"y": marks + [20], "x": []})
    assert s["y"]["marks_on_grid"] < 0.5
