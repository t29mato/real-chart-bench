from __future__ import annotations

import math

import numpy as np
import pytest

from real_chart_bench.domain.digitizer_tools import (
    build_mask,
    calibration_from_person,
    dominant_colors,
    filter_series_by_mask,
    fit_ticks,
    parse_color,
    points_result,
    px_to_value,
    series_to_values,
    summarize_points,
    to_hex,
    value_to_px,
)


def test_parse_color_forms():
    assert parse_color("#ff8000") == (255, 128, 0)
    assert parse_color("FF8000") == (255, 128, 0)
    assert parse_color([1, 2, 3]) == (1, 2, 3)
    assert to_hex((255, 128, 0)) == "#ff8000"
    for bad in ("#fff", "red", [1, 2], [0, 0, 300]):
        with pytest.raises(ValueError):
            parse_color(bad)


def test_dominant_colors_skips_white_and_orders_by_pixels():
    img = np.full((20, 20, 3), 255, np.uint8)
    img[:10, :5] = (200, 0, 0)  # 50 px red
    img[:3, 10:13] = (0, 0, 200)  # 9 px blue
    img[15:, :] = (0, 0, 0)  # 100 px black
    out = dominant_colors(img, k=5)
    assert [c["color"] for c in out] == ["#000000", "#c80000", "#0000c8"]
    assert out[0]["achromatic"] and not out[1]["achromatic"]
    assert out[0]["n_pixels"] == 100
    assert math.isclose(sum(c["fraction"] for c in out), 1.0, abs_tol=1e-3)


def test_dominant_colors_merges_antialiased_shades():
    img = np.full((10, 10, 3), 255, np.uint8)
    img[:5, :5] = (200, 0, 0)
    img[5:, :5] = (205, 4, 3)  # a JPEG shade of the same red, another bin
    out = dominant_colors(img, k=5)
    assert len(out) == 1 and out[0]["n_pixels"] == 50


def test_dominant_colors_treats_a_majority_dark_background_as_background():
    img = np.zeros((10, 10, 3), np.uint8)  # black background
    img[:2, :] = (255, 255, 255)
    img[5, :3] = (0, 200, 0)
    out = dominant_colors(img, k=5)
    assert [c["color"] for c in out] == ["#00c800"]


def test_dominant_colors_with_mask_and_empty():
    img = np.full((10, 10, 3), 255, np.uint8)
    img[0, 0] = (0, 0, 0)
    m = np.zeros((10, 10), bool)
    m[5:, 5:] = True
    assert dominant_colors(img, m) == []
    assert dominant_colors(img, np.zeros((10, 10), bool)) == []


def test_build_mask_include_exclude_frame():
    assert build_mask(10, 10, None) is None
    assert build_mask(10, 10, {}) is None
    m = build_mask(10, 20, {"include": [[0, 0, 9, 9]], "exclude": [[2, 2, 3, 3]]})
    assert m[:, :10].sum() == 100 - 4 and not m[:, 10:].any()
    f = build_mask(20, 20, {"frame": [2, 2, 17, 17], "frame_margin": 2})
    assert f[4, 4] and f[15, 15] and not f[3, 10] and not f[16, 10]
    # reversed corners and out-of-image rectangles are clipped, not errors
    r = build_mask(5, 5, {"include": [[10, 10, -3, 2]]})
    assert r[2:, :].all() and not r[:2, :].any()
    with pytest.raises(ValueError):
        build_mask(5, 5, {"includes": []})
    with pytest.raises(ValueError):
        build_mask(5, 5, {"exclude": [[1, 2, 3]]})


def test_filter_series_by_mask_drops_outside_points_and_empty_series():
    m = np.zeros((10, 10), bool)
    m[:, :5] = True
    s = [{"label": "a", "x": [1, 7, 2.5], "y": [1, 1, 9.9]}, {"label": "b", "x": [8], "y": [2]}]
    assert filter_series_by_mask(s, m) == [{"label": "a", "x": [1, 2.5], "y": [1, 9.9]}]
    assert filter_series_by_mask(s, None) == s


def test_fit_and_convert_linear_and_log():
    lin = {"scale": "linear", "ticks": [[100, 0], [300, 10]]}
    assert px_to_value(200, lin) == pytest.approx(5)
    assert value_to_px(5, lin) == pytest.approx(200)
    # image y grows downward: a y axis has a negative slope
    logy = {"scale": "log", "ticks": [[400, 1e-3], [100, 1e0]]}
    assert px_to_value(300, logy) == pytest.approx(1e-2)
    assert value_to_px(-1, logy) is None
    # least squares over three ticks
    slope, icpt = fit_ticks([[0, 0], [10, 1], [20, 2]], "linear")
    assert (slope, icpt) == pytest.approx((10, 0))


def test_fit_rejects_degenerate_ticks():
    with pytest.raises(ValueError):
        fit_ticks([[1, 5], [2, 5]], "linear")
    with pytest.raises(ValueError):
        fit_ticks([[1, 0], [2, 10]], "log")
    with pytest.raises(ValueError):
        fit_ticks([[1, 1], [2, 10]], "symlog")
    with pytest.raises(ValueError):
        fit_ticks([[1, 1]], "linear")


def test_axis_with_its_own_fit_uses_it():
    # the automatic calibration's robust line wins over refitting its ticks
    ax = {"scale": "linear", "ticks": [[0, 0], [10, 1]], "fit": [20.0, 5.0]}
    assert px_to_value(25, ax) == pytest.approx(1.0)
    assert value_to_px(1, ax) == pytest.approx(25)
    with pytest.raises(ValueError):
        px_to_value(1, {"scale": "linear", "ticks": [], "fit": [0, 1]})


def test_person_calibration_and_series_to_values():
    task = {"x_scale": "linear", "y_scale": "log",
            "x_ticks": [{"pixel_x": 100, "value": 0}, {"pixel_x": 300, "value": 10}],
            "y_ticks": [{"pixel_y": 400, "value": 1e-3}, {"pixel_y": 100, "value": 1}]}
    cal = calibration_from_person(task)
    out = series_to_values([{"label": "a", "x": [200], "y": [300]}], cal)
    assert out[0]["x"][0] == pytest.approx(5) and out[0]["y"][0] == pytest.approx(1e-2)
    with pytest.raises(ValueError):
        series_to_values([], {"x": cal["x"]})


def test_points_result_and_summary_scale_to_view():
    r = points_result("symbol_extract", {"color": "#000000"},
                      [{"label": "s", "x": [10, 20, 30], "y": [5, 6, 7]},
                       {"label": "t", "x": [], "y": []}])
    assert r["n_points"] == 3 and r["n_series"] == 2
    s = summarize_points(r, scale=0.5, max_points=2)
    assert s["series"][0] == {"index": 0, "label": "s", "n": 3, "bbox": [5, 2, 15, 4],
                              "first": [[5, 2], [10, 3]]}
    assert s["series"][1] == {"index": 1, "label": "t", "n": 0}
