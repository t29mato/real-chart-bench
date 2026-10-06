from __future__ import annotations

import json

import pytest

from real_chart_bench.domain.orchestration import (
    MAX_STEPS,
    Action,
    action_schema,
    assemble_final,
    fallback_final,
    params_to_image,
    parse_action,
    result_to_view,
)

TOOLS = ["symbol_extract", "marker_detector", "tick_calibration"]


def _pts(*series):
    return {"kind": "points", "tool": "t", "series": list(series),
            "n_points": sum(len(s["x"]) for s in series), "n_series": len(series)}


def _cal(ok=True):
    return {"kind": "calibration", "ok": ok,
            "calibration": {"x": {"scale": "linear", "ticks": [[0, 0], [10, 1]]},
                            "y": {"scale": "linear", "ticks": [[10, 0], [0, 1]]}}}


A = {"label": "a", "x": [1, 2], "y": [3, 4]}
B = {"label": "b", "x": [5], "y": [6]}
EMPTY = {"label": "e", "x": [], "y": []}


def test_schema_lists_tools_and_requires_every_key():
    s = action_schema(TOOLS)
    assert s["properties"]["tool"]["enum"] == [*TOOLS, ""]
    assert set(s["required"]) == set(s["properties"])
    assert MAX_STEPS == 12


def test_parse_call():
    a = parse_action(json.dumps({"thought": "t", "action": "call", "tool": "symbol_extract",
                                 "params": {"color": "#000000"}, "calibration": "",
                                 "series": []}), TOOLS)
    assert a == Action("call", "t", "symbol_extract", {"color": "#000000"})


@pytest.mark.parametrize(
    ("text", "msg"),
    [("not json", "not JSON"), ("[1]", "one JSON object"),
     ('{"action": "call", "tool": "rm"}', "unknown tool"),
     ('{"action": "call", "tool": "marker_detector", "params": [1]}', "params"),
     ('{"action": "final", "series": []}', "at least one series"),
     ('{"action": "final", "series": [{"from": "r1"}]}', "needs 'from'"),
     ('{"action": "final", "series": [{"from": "r1", "index": true}]}', "needs 'from'"),
     ('{"action": "stop"}', "'call' or 'final'")],
)
def test_parse_errors_explain(text, msg):
    with pytest.raises(ValueError, match=msg):
        parse_action(text, TOOLS)


def test_parse_final():
    a = parse_action(json.dumps({"thought": "", "action": "final", "tool": "", "params": {},
                                 "calibration": "r1",
                                 "series": [{"from": "r2", "index": -1, "label": ""}]}), TOOLS)
    assert a.kind == "final" and a.calibration == "r1"
    assert a.series == ({"from": "r2", "index": -1, "label": ""},)


def test_assemble_final_selects_dedupes_and_labels():
    results = {"r1": _cal(), "r2": _pts(A, B, EMPTY), "r3": _pts(B)}
    act = Action("final", calibration="r1", series=(
        {"from": "r2", "index": 1, "label": "x=0.1"},
        {"from": "r2", "index": -1, "label": "ignored"},
        {"from": "r3", "index": 0, "label": ""},
    ))
    series, cal = assemble_final(act, results, need_calibration=True)
    assert [s["label"] for s in series] == ["x=0.1", "a", "b"]
    assert series[0]["x"] == [5] and cal == results["r1"]["calibration"]
    series, cal = assemble_final(act, results, need_calibration=False)
    assert cal is None and len(series) == 3


@pytest.mark.parametrize(
    ("act", "msg"),
    [(Action("final", calibration="r1", series=({"from": "r9", "index": 0, "label": ""},)),
      "not a points result"),
     (Action("final", calibration="r1", series=({"from": "r2", "index": 5, "label": ""},)),
      "no series 5"),
     (Action("final", calibration="r1", series=({"from": "r2", "index": 2, "label": ""},)),
      "no points"),
     (Action("final", calibration="r4", series=({"from": "r2", "index": 0, "label": ""},)),
      "calibration")],
)
def test_assemble_final_errors(act, msg):
    results = {"r1": _cal(), "r2": _pts(A, B, EMPTY), "r4": _cal(ok=False)}
    with pytest.raises(ValueError, match=msg):
        assemble_final(act, results, need_calibration=True)


def test_params_to_image_scales_view_coordinates():
    p = params_to_image("symbol_extract", {
        "color": "#000000", "min_diameter_px": 4, "max_diameter_px": 50,
        "mask": {"include": [[0, 0, 50, 50]], "exclude": [[10, 10, 20, 20]],
                 "frame": [5, 5, 45, 45], "frame_margin": 1}}, 0.5)
    assert p["min_diameter_px"] == 8 and p["max_diameter_px"] == 100
    assert p["mask"] == {"include": [[0, 0, 100, 100]], "exclude": [[20, 20, 40, 40]],
                         "frame": [10, 10, 90, 90], "frame_margin": 2}
    assert params_to_image("line_extract", {"dx_px": 3, "dy_px": 0.2}, 0.5) == {
        "dx_px": 6, "dy_px": 1}
    # a mask given by result id, and a frame given by calibration id, pass through
    assert params_to_image("mask", {"frame": "r1", "exclude": [[1, 1, 2, 2]]}, 0.5) == {
        "frame": "r1", "exclude": [[2, 2, 4, 4]]}
    m = params_to_image("tick_calibration", {"mode": "manual",
                                             "x": {"scale": "linear", "ticks": [[10, 0], [20, 5]]},
                                             "y": {"scale": "log", "ticks": [[30, 1]]}}, 0.5)
    assert m["x"]["ticks"] == [[20, 0], [40, 5]] and m["y"]["ticks"] == [[60, 1]]
    # nothing to scale, input untouched
    q = {"threshold": 0.4}
    assert params_to_image("marker_detector", q, 0.5) == q


def test_result_to_view_is_compact_and_scaled():
    pts = _pts(A, B)
    assert result_to_view(pts, 0.5) == {"n_series": 2, "n_points": 3, "series": [
        {"index": 0, "label": "a", "n": 2, "bbox": [0, 2, 1, 2]},
        {"index": 1, "label": "b", "n": 1, "bbox": [2, 3, 2, 3]}]}
    cal = {"kind": "calibration", "ok": False, "calibration": None, "frame": [10, 10, 100, 80],
           "tick_marks": {"x": [10, 40], "y": [80]}, "message": "read them"}
    v = result_to_view(cal, 0.5)
    assert v == {"ok": False, "frame": [5, 5, 50, 40], "tick_marks": {"x": [5, 20], "y": [40]},
                 "message": "read them"}
    good = _cal()
    good["calibration"]["frame"] = [0, 0, 10, 10]
    v = result_to_view(good, 1.0)
    assert v["ok"] and v["x"] == {"scale": "linear", "ticks": [[0, 0], [10, 1]]}
    assert "tick_marks" not in v
    ov = {"kind": "overlay", "path": "/x.png",
          "series": [{"result": "r2", "index": 0, "overlay_color": "#e6194b", "n": 2}]}
    assert result_to_view(ov, 1.0) == {"series_drawn": [{"id": "r2:0", "drawn_in": "#e6194b",
                                                         "n": 2}]}


def test_fallback_takes_last_points_and_last_good_calibration():
    results = {"r1": _cal(), "r2": _pts(A), "r3": _pts(B, EMPTY), "r4": _pts(EMPTY),
               "r5": _cal(ok=False)}
    order = ["r1", "r2", "r3", "r4", "r5"]
    series, cal = fallback_final(results, order, need_calibration=True)
    assert [s["label"] for s in series] == ["b"] and cal == results["r1"]["calibration"]
    assert fallback_final({"r2": _pts(A)}, ["r2"], need_calibration=True) is None
    assert fallback_final({"r1": _cal()}, ["r1"], need_calibration=False) is None
    assert fallback_final({"r2": _pts(A)}, ["r2"], need_calibration=False)[1] is None
