from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from real_chart_bench.adapter.orchestrator_tools import (
    TOOLS,
    ToolBox,
    ToolError,
    image_sha256,
)


@pytest.fixture
def chart(tmp_path):
    """A white 200x120 image: a black L frame, three red discs, two blue squares."""
    img = np.full((120, 200, 3), 255, np.uint8)
    img[100, 20:190] = 0  # x axis
    img[10:101, 20] = 0  # y axis
    yy, xx = np.mgrid[:120, :200]
    for cx, cy in ((50, 80), (90, 60), (130, 40)):
        img[(xx - cx) ** 2 + (yy - cy) ** 2 <= 9] = (220, 20, 20)
    for cx, cy in ((60, 30), (150, 70)):
        img[cy - 3:cy + 4, cx - 3:cx + 4] = (20, 20, 220)
    path = tmp_path / "fig.png"
    Image.fromarray(img).save(path)
    return path


def _store():
    results = {}

    def resolve(ref):
        if ref not in results:
            raise ToolError(f"no result {ref}")
        return results[ref]

    return results, resolve


def test_tools_listed():
    assert TOOLS == ("dominant_colors", "mask", "symbol_extract", "line_extract",
                     "marker_detector", "tick_calibration", "to_values", "render_overlay")


def test_colors_then_symbol_extract(chart):
    tb = ToolBox(chart)
    results, resolve = _store()
    cols = tb.run("dominant_colors", {"k": 4}, resolve)["colors"]
    red = next(c["color"] for c in cols if c["rgb"][0] > 150 and c["rgb"][2] < 100)
    r = tb.run("symbol_extract", {"color": red, "distance_pct": 5}, resolve)
    assert r["kind"] == "points" and r["n_points"] == 3
    assert r["series"][0]["x"] == [50.5, 90.5, 130.5]  # sorted by x, centre + 0.5
    # exclude one disc with a rectangle
    r2 = tb.run("symbol_extract", {"color": red, "distance_pct": 5,
                                   "mask": {"exclude": [[80, 50, 100, 70]]}}, resolve)
    assert r2["n_points"] == 2


def test_line_extract_and_bad_params(chart):
    tb = ToolBox(chart)
    _, resolve = _store()
    r = tb.run("line_extract", {"color": "#000000", "dx_px": 20, "dy_px": 20}, resolve)
    assert r["n_points"] > 5
    with pytest.raises(ToolError, match="unknown parameters"):
        tb.run("symbol_extract", {"color": "#000000", "radius": 3}, resolve)
    with pytest.raises(ToolError, match="color is required"):
        tb.run("symbol_extract", {}, resolve)
    with pytest.raises(ToolError, match="outside"):
        tb.run("line_extract", {"color": "#000000", "dx_px": 0}, resolve)
    with pytest.raises(ToolError, match="unknown tool"):
        tb.run("python", {}, resolve)


def test_mask_by_reference_and_frame_of_calibration(chart):
    tb = ToolBox(chart, given_calibration={
        "x": {"scale": "linear", "ticks": [[20, 0], [190, 17]]},
        "y": {"scale": "linear", "ticks": [[100, 0], [10, 9]]}, "frame": None,
        "source": "person"})
    results, resolve = _store()
    results["c"] = tb.run("tick_calibration", {}, resolve)
    assert results["c"]["ok"] and results["c"]["frame"] is not None  # frame rules
    results["m"] = tb.run("mask", {"frame": "c", "exclude": [[40, 70, 60, 90]]}, resolve)
    assert results["m"]["kind"] == "mask" and 0 < results["m"]["fraction"] < 1
    r = tb.run("symbol_extract", {"color": "#dc1414", "distance_pct": 5, "mask": "m"}, resolve)
    assert r["n_points"] == 2
    results["p"] = r
    v = tb.run("to_values", {"result": "p", "calibration": "c"}, resolve)
    assert v["series"][0]["x"][0] == pytest.approx((90.5 - 20) / 10)
    with pytest.raises(ToolError, match="not a mask"):
        tb.run("symbol_extract", {"color": "#000000", "mask": "p"}, resolve)


def test_calibration_modes(chart):
    tb = ToolBox(chart, allow_auto_calibration=False)
    _, resolve = _store()
    with pytest.raises(ToolError, match="no calibration is given"):
        tb.run("tick_calibration", {"mode": "given"}, resolve)
    with pytest.raises(ToolError, match="not used"):
        tb.run("tick_calibration", {"mode": "auto"}, resolve)
    ok = tb.run("tick_calibration", {"mode": "manual",
                                     "x": {"scale": "linear", "ticks": [[20, 0], [190, 17]]},
                                     "y": {"scale": "log", "ticks": [[100, 1], [10, 100]]}},
                resolve)
    assert ok["ok"] and ok["calibration"]["source"] == "manual"
    with pytest.raises(ToolError, match="manual calibration"):
        tb.run("tick_calibration", {"mode": "manual", "x": {"scale": "linear", "ticks": []}},
               resolve)


def test_marker_detector_reads_cache_and_masks(chart, tmp_path):
    dets = tmp_path / "dets"
    dets.mkdir()
    emb_a, emb_b = [0.0] * 8, [5.0] * 8
    rows = [[50.0, 80.0, 0.9, "circle", emb_a], [90.0, 60.0, 0.8, "circle", emb_a],
            [130.0, 40.0, 0.7, "circle", emb_a], [60.0, 30.0, 0.9, "square", emb_b],
            [150.0, 70.0, 0.9, "square", emb_b], [10.0, 10.0, 0.2, "other", emb_b]]
    (dets / f"{image_sha256(chart)}.json").write_text(
        json.dumps({"size": [200, 120], "dets": {"768": rows}}))
    tb = ToolBox(chart, dets_dir=dets)
    _, resolve = _store()
    r = tb.run("marker_detector", {}, resolve)
    assert r["n_series"] == 2 and r["n_points"] == 5
    r = tb.run("marker_detector", {"mask": {"exclude": [[120, 0, 199, 119]]}}, resolve)
    # the square series keeps one point and falls under min_points 2
    assert r["n_series"] == 1 and r["n_points"] == 2
    with pytest.raises(ToolError, match="long_side"):
        tb.run("marker_detector", {"long_side": 1024}, resolve)
    with pytest.raises(ToolError, match="not available"):
        ToolBox(chart).run("marker_detector", {}, resolve)


def test_render_overlay_writes_png(chart, tmp_path):
    tb = ToolBox(chart, out_dir=tmp_path / "ov")
    results, resolve = _store()
    results["p"] = tb.run("symbol_extract", {"color": "#dc1414", "distance_pct": 5}, resolve)
    results["c"] = tb.run("tick_calibration", {"mode": "manual",
                                               "x": {"scale": "linear",
                                                     "ticks": [[20, 0], [190, 17]]},
                                               "y": {"scale": "linear",
                                                     "ticks": [[100, 0], [10, 9]]}}, resolve)
    o = tb.run("render_overlay", {"results": ["p"], "calibration": "c", "max_side": 100}, resolve)
    im = Image.open(o["path"])
    assert im.size == (100, 60)
    assert o["series"] == [{"result": "p", "index": 0, "overlay_color": "#e6194b", "n": 3}]
