"""verify (方式D「検証とやり直し」): objective signals about an extraction,
measured on the image, never on the ground truth."""

from __future__ import annotations

import math

import numpy as np
import pytest

from real_chart_bench.domain.verification import (
    Thresholds,
    calibration_signals,
    estimate_background,
    ink_mask,
    score,
    verdict,
    verify,
)

FRAME = [20.0, 20.0, 380.0, 280.0]  # short side 260 -> ink radius 3, null shift 16


def canvas(w=400, h=300):
    img = np.full((h, w, 3), 255, np.uint8)
    # the frame lines (black), as in a plot
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


RED = (220, 30, 30)
BLUE = (30, 30, 220)
CENTRES = [(80, 100), (140, 180), (200, 120), (260, 200), (320, 90)]


def red_markers():
    img = canvas()
    for cx, cy in CENTRES:
        disc(img, cx, cy, 5, RED)
    return img


def series(pts, label="s"):
    return {"label": label, "x": [p[0] for p in pts], "y": [p[1] for p in pts]}


# ------------------------------------------------------------------ ink


def test_background_and_ink():
    img = red_markers()
    assert estimate_background(img) == (255, 255, 255)
    ink = ink_mask(img)
    assert ink[100, 80] and not ink[50, 50]
    # light grey grid lines are not ink
    img[150, 30:370] = (225, 225, 225)
    assert not ink_mask(img)[150, 100]


def test_background_of_a_dark_figure():
    img = np.full((50, 50, 3), 20, np.uint8)
    img[10:15, 10:15] = (250, 250, 0)
    assert estimate_background(img) == (20, 20, 20)
    assert ink_mask(img)[12, 12] and not ink_mask(img)[40, 40]


# ------------------------------------------------------------------ series


def test_points_on_markers_hit_and_contrast():
    out = verify(red_markers(), [series(CENTRES)], FRAME)
    s = out["series"][0]
    assert s["n"] == 5
    assert s["hit"] == 1.0
    assert s["null"] == 0.0
    assert s["color_consistency"] == 1.0
    assert s["color"] == "#dc1e1e"
    assert s["duplicates"] == 0
    assert s["outside"] == 0
    assert out["unexplained"]["n"] == 0
    assert out["marker_diameter_px"] == pytest.approx(11, abs=1)


def test_points_off_the_markers_do_not_hit():
    pts = [(x + 25, y + 25) for x, y in CENTRES]
    s = verify(red_markers(), [series(pts)], FRAME)["series"][0]
    assert s["hit"] == 0.0
    assert s["contrast"] <= 0.0


def test_colour_consistency_splits_two_colours():
    img = red_markers()
    extra = [(110, 240), (170, 60), (230, 250), (290, 40)]
    for cx, cy in extra[:2]:
        disc(img, cx, cy, 5, BLUE)
    s = verify(img, [series(CENTRES[:2] + extra[:2])], FRAME)["series"][0]
    assert s["color_consistency"] == 0.5


def test_duplicates_and_outside():
    pts = [*CENTRES, (81, 101), (390, 295)]
    s = verify(red_markers(), [series(pts)], FRAME)["series"][0]
    assert s["duplicates"] == 1  # (81,101) sits on (80,100)'s marker
    assert s["outside"] == 1


def test_points_on_frame_lines_do_not_testify():
    # a projection squashed onto the x axis: ink there is the axis line
    pts = [(x, FRAME[3]) for x, _ in CENTRES]
    s = verify(red_markers(), [series(pts)], FRAME)["series"][0]
    assert s["testifying"] == 0
    assert s["hit"] is None


def test_cross_series_duplicates():
    out = verify(red_markers(), [series(CENTRES, "a"), series(CENTRES[:2], "b")], FRAME)
    assert out["cross_duplicates"] == 2


# ------------------------------------------------------------------ unexplained blobs


def test_unexplained_marker_blobs_are_counted_with_colour_match():
    img = red_markers()
    disc(img, 300, 240, 5, BLUE)
    out = verify(img, [series(CENTRES[:3])], FRAME)
    u = out["unexplained"]
    assert u["n"] == 3
    assert u["same_color"] == 2  # the two red markers left out
    assert u["share"] == pytest.approx(3 / (3 + 3))
    centres = sorted((round(b["x"]), round(b["y"])) for b in u["blobs"])
    assert centres == [(260, 200), (300, 240), (320, 90)]


def test_blobs_under_mask_exclusion_and_text_boxes_do_not_count():
    img = red_markers()
    out = verify(img, [series(CENTRES[:3])], FRAME,
                 mask={"exclude": [[250, 190, 270, 210]]},
                 text_boxes=[(310, 80, 330, 100)])
    assert out["unexplained"]["n"] == 0


def test_legend_symbol_left_of_text_does_not_count():
    img = red_markers()
    disc(img, 300, 240, 5, BLUE)  # a legend symbol ...
    out = verify(img, [series(CENTRES)], FRAME,
                 text_boxes=[(312, 234, 360, 246)])  # ... with its text to the right
    assert out["unexplained"]["n"] == 0


def test_long_lines_are_not_marker_blobs():
    img = red_markers()
    img[250:253, 40:360] = BLUE  # a fit line
    out = verify(img, [series(CENTRES)], FRAME)
    assert out["unexplained"]["n"] == 0


def joined_markers():
    """Red discs joined by a red line: one big component, no blob to count."""
    img = canvas()
    pts = sorted(CENTRES)
    for (ax, ay), (bx, by) in zip(pts, pts[1:], strict=False):
        for t in np.linspace(0, 1, 400):
            img[int(round(ay + t * (by - ay))), int(round(ax + t * (bx - ax)))] = RED
    for cx, cy in pts:
        disc(img, cx, cy, 5, RED)
    return img


def test_lookalikes_find_missed_markers_joined_by_a_line():
    out = verify(joined_markers(), [series(sorted(CENTRES)[:3])], FRAME)
    assert out["unexplained"]["n"] == 0  # the blob count cannot see them
    found = sorted((round(p["x"]), round(p["y"])) for p in out["lookalikes"]["points"])
    assert found == [(260, 200), (320, 90)]
    assert out["series"][0]["self_match"] == 1.0
    assert out["missed"]["n"] == 2
    assert not verdict(out)["accept"]


def test_all_markers_answered_on_joined_markers_is_clean():
    out = verify(joined_markers(), [series(sorted(CENTRES))], FRAME)
    assert out["lookalikes"]["n"] == 0
    assert verdict(out)["accept"]


def test_points_along_a_line_have_many_lookalikes():
    img = canvas()
    img[150:153, 40:360] = RED
    pts = [(60, 151), (120, 151), (180, 151)]
    out = verify(img, [series(pts)], FRAME)
    assert out["lookalikes"]["n"] >= 5


def test_detector_agreement_support_and_coverage():
    dets = [(x, y, 0.9) for x, y in CENTRES] + [(100, 260, 0.3), (60, 60, 0.05)]
    pts = [*CENTRES[:3], (100, 260), (60, 60)]
    out = verify(red_markers(), [series(pts)], FRAME, detections=dets)
    det = out["detector"]
    assert det["support"] == pytest.approx(4 / 5)  # (60,60) has only a 0.05 peak
    assert det["n_confident"] == 5
    assert det["coverage"] == pytest.approx(3 / 5)
    assert sorted((p["x"], p["y"]) for p in det["uncovered"]) == [(260, 200), (320, 90)]
    assert det["f1"] == pytest.approx(2 * 0.8 * 0.6 / 1.4, abs=1e-4)


def test_detector_agreement_merges_double_peaks_and_skips_masked():
    dets = [(80, 100, 0.9), (81, 101, 0.8), (320, 90, 0.9)]
    out = verify(red_markers(), [series([(80, 100)])], FRAME, detections=dets,
                 mask={"exclude": [[300, 70, 340, 110]]})
    assert out["detector"]["n_confident"] == 1
    assert out["detector"]["coverage"] == 1.0


# ------------------------------------------------------------------ calibration


def lin_axis(ticks, scale="linear"):
    return {"scale": scale, "ticks": [list(t) for t in ticks]}


def test_calibration_signals_consistent_axis():
    cal = {"x": lin_axis([(20, 0), (110, 1), (200, 2), (290, 3), (380, 4)]),
           "y": lin_axis([(280, 0), (150, 50), (20, 100)]), "frame": FRAME}
    marks = {"x": [20, 65, 110, 155, 200, 245, 290, 335, 380], "y": [280, 150, 20]}
    s = calibration_signals(cal, marks)
    assert s["x"]["residual"] == pytest.approx(0, abs=1e-9)
    assert s["x"]["direction_ok"] and s["y"]["direction_ok"]
    assert s["x"]["grid_dev"] == pytest.approx(0, abs=1e-9)
    assert s["x"]["marks_on_grid"] == 1.0  # half steps are minor ticks on the grid
    assert s["x"]["labels_on_marks"] == 1.0


def test_calibration_signals_flag_a_misread_label():
    # "3" read as "8": the fit no longer passes through the labels
    cal = {"x": lin_axis([(20, 0), (110, 1), (200, 2), (290, 8), (380, 4)]),
           "y": lin_axis([(280, 0), (20, 100)])}
    s = calibration_signals(cal, {"x": [20, 110, 200, 290, 380], "y": []})
    assert s["x"]["residual"] > 0.05
    assert s["x"]["grid_dev"] > 0.2
    assert s["y"]["marks_on_grid"] is None  # too few marks to say


def test_calibration_signals_direction_and_log_axis():
    cal = {"x": lin_axis([(20, 4), (380, 0)]),  # values fall to the right
           "y": lin_axis([(280, 1), (150, 10), (20, 100)], "log")}
    marks = {"x": [], "y": [280, 241, 150, 111, 20]}  # 1, 2, 10, 20, 100
    s = calibration_signals(cal, marks)
    assert not s["x"]["direction_ok"]
    assert s["y"]["direction_ok"]
    assert s["y"]["residual"] == pytest.approx(0, abs=1e-9)
    assert s["y"]["marks_on_grid"] == 1.0


def test_log_axis_labels_must_be_one_digit_times_a_power_of_ten():
    good = lin_axis([(280, 0.3), (230, 0.6), (190, 1), (110, 2), (60, 3)], "log")
    bad = lin_axis([(339, 3107), (304, 4107), (254, 6107)], "log")  # "6x10^-1" misread
    s = calibration_signals({"x": lin_axis([(20, 0), (380, 4)]), "y": good}, None)
    assert s["y"]["grid_dev"] == 0
    s = calibration_signals({"x": lin_axis([(20, 0), (380, 4)]), "y": bad}, None)
    assert s["y"]["grid_dev"] == 1.0


def test_calibration_signals_with_fit_override():
    # the auto calibration keeps its own robust fit; residual is against it
    ax = {"scale": "linear", "ticks": [[20, 0], [200, 2], [380, 4]], "fit": [90.0, 20.0]}
    s = calibration_signals({"x": ax, "y": lin_axis([(280, 0), (20, 1)])}, None)
    assert s["x"]["residual"] == pytest.approx(0, abs=1e-9)
    assert s["x"]["marks_on_grid"] is None


# ------------------------------------------------------------------ verdict


def test_verdict_accepts_a_clean_extraction():
    out = verify(red_markers(), [series(CENTRES)], FRAME)
    v = verdict(out)
    assert v["accept"] and v["reasons"] == []
    assert 0.9 <= v["score"] <= 1.0


def test_verdict_rejects_missed_markers_and_off_ink_points():
    img = red_markers()
    off = [(x + 25, y + 25) for x, y in CENTRES]
    v = verdict(verify(img, [series(off)], FRAME))
    assert not v["accept"] and v["score"] == 0.0
    assert any("score" in r for r in v["reasons"])
    assert any("not on markers" in h for h in v["hints"])
    v2 = verdict(verify(img, [series(CENTRES[:2])], FRAME))
    assert not v2["accept"]
    assert v2["score"] == pytest.approx(2 * 1 * 0.4 / 1.4, abs=1e-3)  # 2 of 5 found
    assert any("unexplained" in h for h in v2["hints"])


def test_verdict_uses_given_thresholds():
    out = verify(red_markers(), [series(CENTRES[:4])], FRAME)  # score 0.889
    assert verdict(out, Thresholds(min_score_image=0.85))["accept"]
    assert not verdict(out, Thresholds(min_score_image=0.95))["accept"]


def test_score_with_detector_peaks_is_their_f1_times_explained_share():
    dets = [(x, y, 0.9) for x, y in CENTRES]
    out = verify(red_markers(), [series(CENTRES[:3])], FRAME, detections=dets)
    assert out["detector"]["f1"] == pytest.approx(2 * 0.6 / 1.6, abs=1e-4)
    assert out["missed"]["share"] == pytest.approx(0.4)
    assert score(out) == pytest.approx(0.75 * 0.6, abs=1e-4)
    v = verdict(out, Thresholds(min_score=0.4))
    assert v["accept"] and any("detector peaks have no point" in h for h in v["hints"])
    assert not verdict(out, Thresholds(min_score=0.5))["accept"]


def test_verdict_flags_a_bad_calibration():
    out = verify(red_markers(), [series(CENTRES)], FRAME,
                 calibration={"x": lin_axis([(20, 0), (110, 1), (200, 2), (290, 8), (380, 4)]),
                              "y": lin_axis([(280, 0), (20, 100)])},
                 tick_marks={"x": [20, 110, 200, 290, 380], "y": []})
    v = verdict(out)
    assert not v["accept"]
    assert any("calibration" in r for r in v["reasons"])


def test_verify_with_no_points():
    out = verify(red_markers(), [], FRAME)
    assert out["n_points"] == 0
    v = verdict(out)
    assert not v["accept"] and v["score"] == 0.0


def test_verify_without_frame_uses_the_whole_image():
    out = verify(red_markers(), [series(CENTRES)], None)
    assert out["series"][0]["hit"] == 1.0
    assert math.isfinite(out["marker_diameter_px"])
