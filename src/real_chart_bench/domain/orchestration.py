"""The orchestrator loop of 方式D (docs/design/local-model.md「方式D:
司令塔 + 道具」), the parts that are plain logic.

The model never writes a number of the answer. Each turn it returns one JSON
action: call a tool with parameters, or finish by naming which tool results
make the answer -- series of an extraction result, and (condition 1) the
calibration result that converts them. Values come only from the tools.

- ``action_schema``: the JSON schema the model's output is constrained to;
- ``parse_action``: one turn's text -> a validated action;
- ``assemble_final``: the final action + the stored results -> pixel series
  and the calibration;
- ``fallback_final``: what is answered when the model does not finish within
  the step cap or its final action is unusable: every series of the last
  extraction result, with the last successful calibration. Fixed in advance,
  never chosen on the benchmark.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

MAX_STEPS = 12


def action_schema(tools: list[str]) -> dict:
    return {
        "type": "object",
        "properties": {
            "thought": {"type": "string"},
            "action": {"enum": ["call", "final"]},
            "tool": {"enum": [*tools, ""]},
            "params": {"type": "object"},
            "calibration": {"type": "string"},
            "series": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "from": {"type": "string"},
                        "index": {"type": "integer"},
                        "label": {"type": "string"},
                    },
                    "required": ["from", "index", "label"],
                },
            },
        },
        "required": ["thought", "action", "tool", "params", "calibration", "series"],
    }


@dataclass(frozen=True)
class Action:
    kind: str  # "call" | "final"
    thought: str = ""
    tool: str = ""
    params: dict = field(default_factory=dict)
    calibration: str = ""
    series: tuple = ()


def parse_action(text: str, tools: list[str]) -> Action:
    """One turn's output -> Action. ValueError says what is wrong, in words
    the model can act on (it is shown the message as the tool result)."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"output is not JSON ({exc.msg})") from exc
    if not isinstance(obj, dict):
        raise ValueError("output must be one JSON object")
    kind = obj.get("action")
    thought = str(obj.get("thought", ""))
    if kind == "call":
        tool = obj.get("tool")
        if tool not in tools:
            raise ValueError(f"unknown tool {tool!r}; tools: {', '.join(tools)}")
        params = obj.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError("params must be an object")
        return Action("call", thought, tool, params)
    if kind == "final":
        series = obj.get("series") or []
        if not isinstance(series, list) or not series:
            raise ValueError("a final action names at least one series: "
                             '[{"from": "r2", "index": 0, "label": "..."}]')
        items = []
        for s in series:
            if not isinstance(s, dict) or not isinstance(s.get("from"), str) \
                    or not isinstance(s.get("index"), int) or isinstance(s.get("index"), bool):
                raise ValueError(f"series item {s!r} needs 'from' (result id) and 'index' (int)")
            items.append({"from": s["from"], "index": s["index"],
                          "label": str(s.get("label") or "")})
        return Action("final", thought, calibration=str(obj.get("calibration") or ""),
                      series=tuple(items))
    raise ValueError("action must be 'call' or 'final'")


def assemble_final(action: Action, results: dict[str, dict], need_calibration: bool
                   ) -> tuple[list[dict], dict | None]:
    """The series a final action names (index -1 = every series of that
    result), in the order named, and the calibration it names. A series
    named twice is taken once."""
    out, seen = [], set()
    for item in action.series:
        r = results.get(item["from"])
        if r is None or r.get("kind") != "points":
            raise ValueError(f"{item['from']!r} is not a points result")
        idxs = range(len(r["series"])) if item["index"] == -1 else [item["index"]]
        for i in idxs:
            if not 0 <= i < len(r["series"]):
                raise ValueError(f"{item['from']} has no series {i}")
            if (item["from"], i) in seen:
                continue
            seen.add((item["from"], i))
            s = r["series"][i]
            if not s["x"]:
                continue
            label = item["label"] if item["index"] != -1 and item["label"] else s.get("label")
            out.append({"label": label or f"series_{len(out) + 1}", "x": list(s["x"]),
                        "y": list(s["y"])})
    if not out:
        raise ValueError("the named series have no points")
    cal = None
    if need_calibration:
        c = results.get(action.calibration)
        if c is None or c.get("kind") != "calibration" or not c.get("ok"):
            raise ValueError(f"{action.calibration!r} is not a successful calibration result")
        cal = c["calibration"]
    return out, cal


def _scale_rect(r, f):
    return [v * f for v in r] if isinstance(r, list) else r


def _scale_mask(m, f):
    if not isinstance(m, dict):
        return m  # a mask result id
    out = dict(m)
    for k in ("include", "exclude"):
        if isinstance(out.get(k), list):
            out[k] = [_scale_rect(r, f) for r in out[k]]
    if isinstance(out.get("frame"), list):
        out["frame"] = _scale_rect(out["frame"], f)
    if isinstance(out.get("frame_margin"), int | float):
        out["frame_margin"] = out["frame_margin"] * f
    return out


def params_to_image(tool: str, params: dict, view_scale: float) -> dict:
    """The model sees the image resized by view_scale (<= 1) and gives
    coordinates and sizes in that view; tools work in the file's pixels."""
    f = 1.0 / view_scale
    p = dict(params)
    if "mask" in p:
        p["mask"] = _scale_mask(p["mask"], f)
    if tool == "mask":
        p = _scale_mask(p, f)
    for k in ("min_diameter_px", "max_diameter_px"):
        if isinstance(p.get(k), int | float):
            p[k] = p[k] * f
    for k in ("dx_px", "dy_px"):
        if isinstance(p.get(k), int | float):
            p[k] = max(1, round(p[k] * f))
    if tool == "tick_calibration":
        for ax in ("x", "y"):
            a = p.get(ax)
            if isinstance(a, dict) and isinstance(a.get("ticks"), list):
                p[ax] = {**a, "ticks": [[t[0] * f, *t[1:]] if isinstance(t, list) and t else t
                                        for t in a["ticks"]]}
    return p


def result_to_view(result: dict, view_scale: float, max_items: int = 20) -> dict:
    """What the model reads of a tool result: compact, in view pixels."""
    s = view_scale
    kind = result.get("kind")
    if kind == "points":
        out = {"n_series": result["n_series"], "n_points": result["n_points"], "series": []}
        for i, ser in enumerate(result["series"][:max_items]):
            xs = [v * s for v in ser["x"]]
            ys = [v * s for v in ser["y"]]
            item = {"index": i, "label": ser.get("label"), "n": len(xs)}
            if xs:
                item["bbox"] = [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]
            out["series"].append(item)
        return out
    if kind == "calibration":
        cal = result.get("calibration") or {}
        frame = cal.get("frame") or result.get("frame")
        out = {"ok": result.get("ok"),
               "frame": [round(v * s) for v in frame] if frame else None}
        for ax in ("x", "y"):
            a = cal.get(ax)
            if a:
                out[ax] = {"scale": a["scale"],
                           "ticks": [[round(px * s), v] for px, v in a["ticks"][:max_items]]}
        marks = result.get("tick_marks")
        if marks and not result.get("ok"):
            out["tick_marks"] = {ax: [round(v * s) for v in marks[ax][:max_items]]
                                 for ax in ("x", "y")}
        if result.get("message"):
            out["message"] = result["message"]
        return out
    if kind == "colors":
        return {"colors": [{"color": c["color"], "fraction": c["fraction"],
                            "achromatic": c["achromatic"]} for c in result["colors"]]}
    if kind == "mask":
        return {"fraction_of_image": result["fraction"]}
    if kind == "overlay":
        return {"series_drawn": [{"id": f"{x['result']}:{x['index']}", "drawn_in":
                                  x["overlay_color"], "n": x["n"]} for x in result["series"]]}
    return {k: v for k, v in result.items() if k != "series"}


def fallback_final(results: dict[str, dict], order: list[str], need_calibration: bool
                   ) -> tuple[list[dict], dict | None] | None:
    """Every series of the last points result with points, and the last
    successful calibration; None when there is no such result."""
    pts = [r for r in order if results[r].get("kind") == "points" and results[r]["n_points"]]
    if not pts:
        return None
    cal = None
    if need_calibration:
        cals = [r for r in order
                if results[r].get("kind") == "calibration" and results[r].get("ok")]
        if not cals:
            return None
        cal = results[cals[-1]]["calibration"]
    series = [{"label": s.get("label") or f"series_{i + 1}", "x": list(s["x"]), "y": list(s["y"])}
              for i, s in enumerate(results[pts[-1]]["series"]) if s["x"]]
    return series, cal
