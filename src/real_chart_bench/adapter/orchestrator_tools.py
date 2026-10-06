"""The fixed tools of 方式D (docs/design/local-model.md「方式D: 司令塔 + 道具」).

An orchestrator (a local VLM, or Claude through scripts/tools/rcb_tool.py)
only chooses tools and their settings; every number of the answer comes from
one of these:

| tool             | what it does                                                  |
|------------------|---------------------------------------------------------------|
| dominant_colors  | the image's main non-white colours (candidate series colours) |
| mask             | a mask spec checked and counted (plot frame / rectangles /    |
|                  | excluded legend box); extraction tools take the same spec     |
| symbol_extract   | starry-digitizer's Symbol Extract (domain/starry_extract.py)  |
| line_extract     | starry-digitizer's Line Extract                               |
| marker_detector  | 方式A's MarkerNet detections + its post-processing            |
| tick_calibration | 方式C's automatic calibration (frame + Tesseract tick OCR),   |
|                  | the person's calibration in condition 2, or ticks the         |
|                  | orchestrator read itself ("manual")                           |
| to_values        | pixel points -> values through a calibration                  |
| render_overlay   | a PNG of the figure with chosen results drawn over it         |

Results are JSON-able dicts. Parameters may refer to earlier results by an
id (a string); ``resolve`` turns the id into the result -- the local loop
keeps them in memory, the CLI reads saved JSON files.

The detector runs once per image elsewhere (it needs torch and the
weights); its raw detections (low threshold, several input sizes) are read
from a cache keyed by the image's sha256 (scripts/tools/export_detector_cache.py)
and post-processed here with the requested settings.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from real_chart_bench.adapter.auto_axis_calibration import DARK, calibrate_image, load_rgb, to_gray
from real_chart_bench.domain.axis_frame import detect_axis_frames, detect_ticks
from real_chart_bench.domain.digitizer_tools import (
    build_mask,
    check_calibration,
    dominant_colors,
    parse_color,
    points_result,
    series_to_values,
    to_hex,
    value_to_px,
)
from real_chart_bench.domain.marker_detection import Detection, PostConfig, postprocess
from real_chart_bench.domain.starry_extract import line_extract, symbol_extract

TOOLS = (
    "dominant_colors",
    "mask",
    "symbol_extract",
    "line_extract",
    "marker_detector",
    "tick_calibration",
    "to_values",
    "render_overlay",
)

# 方式A (b): the setting chosen on the validation split (docs/design/local-model.md)
DETECTOR_DEFAULTS = {"threshold": 0.4, "group_threshold": 0.5, "long_side": 768,
                     "dup_frac": 0.004, "min_points": 2}
OVERLAY_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#00a0a0",
                  "#f032e6", "#9a6324", "#808000", "#000075"]


class ToolError(ValueError):
    """A tool call that cannot run as asked (bad parameters, missing input)."""


def image_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _check_keys(params: dict, allowed: set[str], tool: str) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise ToolError(f"{tool}: unknown parameters {sorted(unknown)}; "
                        f"allowed: {sorted(allowed)}")


def _num(params: dict, key: str, default, lo=None, hi=None, cast=float):
    v = params.get(key, default)
    try:
        v = cast(v)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"{key} must be a number") from exc
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ToolError(f"{key}={v} outside [{lo}, {hi}]")
    return v


class ToolBox:
    def __init__(
        self,
        image_path: Path,
        *,
        dets_dir: Path | None = None,
        given_calibration: dict | None = None,
        allow_auto_calibration: bool = True,
        out_dir: Path | None = None,
    ) -> None:
        self.image_path = Path(image_path)
        self.dets_dir = Path(dets_dir) if dets_dir else None
        self.given_calibration = given_calibration
        self.allow_auto_calibration = allow_auto_calibration
        self.out_dir = Path(out_dir) if out_dir else None
        self._rgb: np.ndarray | None = None

    @property
    def rgb(self) -> np.ndarray:
        if self._rgb is None:
            rgb = load_rgb(self.image_path)
            if rgb is None:
                raise ToolError(f"cannot read image {self.image_path.name}")
            self._rgb = rgb
        return self._rgb

    @property
    def size(self) -> tuple[int, int]:
        h, w = self.rgb.shape[:2]
        return w, h

    # ------------------------------------------------------------ dispatch

    def run(self, tool: str, params: dict | None, resolve: Callable[[str], dict]) -> dict:
        params = dict(params or {})
        fn = getattr(self, f"_t_{tool}", None) if tool in TOOLS else None
        if fn is None:
            raise ToolError(f"unknown tool {tool!r}; tools: {', '.join(TOOLS)}")
        return fn(params, resolve)

    def _mask(self, spec, resolve) -> tuple[np.ndarray | None, dict | None]:
        if spec is None:
            return None, None
        if isinstance(spec, str):  # a mask result's id
            r = resolve(spec)
            if r.get("kind") != "mask":
                raise ToolError(f"{spec} is not a mask result")
            spec = r["spec"]
        if not isinstance(spec, dict):
            raise ToolError("mask must be an object or a mask result id")
        spec = dict(spec)
        if isinstance(spec.get("frame"), str):  # the frame of a calibration result
            r = resolve(spec["frame"])
            frame = (r.get("calibration") or {}).get("frame") or r.get("frame")
            if not frame:
                raise ToolError(f"{spec['frame']} has no plot frame")
            spec["frame"] = frame
        w, h = self.size
        try:
            return build_mask(h, w, spec), spec
        except ValueError as exc:
            raise ToolError(f"mask: {exc}") from exc

    # ------------------------------------------------------------ tools

    def _t_dominant_colors(self, p, resolve):
        _check_keys(p, {"k", "mask"}, "dominant_colors")
        m, _ = self._mask(p.get("mask"), resolve)
        k = _num(p, "k", 8, 1, 20, int)
        return {"kind": "colors", "colors": dominant_colors(self.rgb, m, k)}

    def _t_mask(self, p, resolve):
        _check_keys(p, {"include", "exclude", "frame", "frame_margin"}, "mask")
        m, spec = self._mask(p, resolve)
        w, h = self.size
        n = int(m.sum()) if m is not None else w * h
        return {"kind": "mask", "spec": spec, "n_pixels": n, "fraction": round(n / (w * h), 4)}

    def _extract_common(self, p, resolve):
        if "color" not in p:
            raise ToolError("color is required (e.g. from dominant_colors)")
        try:
            target = parse_color(p["color"])
        except ValueError as exc:
            raise ToolError(str(exc)) from exc
        pct = _num(p, "distance_pct", 1.0, 0.0, 100.0)
        m, spec = self._mask(p.get("mask"), resolve)
        return target, pct, m, spec

    def _t_symbol_extract(self, p, resolve):
        _check_keys(p, {"color", "distance_pct", "min_diameter_px", "max_diameter_px", "mask"},
                    "symbol_extract")
        target, pct, m, spec = self._extract_common(p, resolve)
        lo = _num(p, "min_diameter_px", 5, 0)
        hi = _num(p, "max_diameter_px", 100, 0)
        pts = symbol_extract(self.rgb, target, pct, m, lo, hi)
        params = {"color": to_hex(target), "distance_pct": pct, "min_diameter_px": lo,
                  "max_diameter_px": hi, "mask": spec}
        return points_result("symbol_extract", params, [_series(to_hex(target), pts)])

    def _t_line_extract(self, p, resolve):
        _check_keys(p, {"color", "distance_pct", "dx_px", "dy_px", "mask"}, "line_extract")
        target, pct, m, spec = self._extract_common(p, resolve)
        dx = _num(p, "dx_px", 10, 1, 500, int)
        dy = _num(p, "dy_px", 10, 1, 500, int)
        pts = line_extract(self.rgb, target, pct, m, dx, dy)
        params = {"color": to_hex(target), "distance_pct": pct, "dx_px": dx, "dy_px": dy,
                  "mask": spec}
        return points_result("line_extract", params, [_series(to_hex(target), pts)])

    def _raw_detections(self, long_side: int) -> list[Detection]:
        if self.dets_dir is None:
            raise ToolError("marker_detector is not available here (no detection cache)")
        path = self.dets_dir / f"{image_sha256(self.image_path)}.json"
        if not path.exists():
            raise ToolError("marker_detector has no detections for this image")
        rec = json.loads(path.read_text())
        rows = rec["dets"].get(str(long_side))
        if rows is None:
            raise ToolError(f"long_side must be one of {sorted(int(k) for k in rec['dets'])}")
        return [Detection(x, y, s, mk, tuple(e)) for x, y, s, mk, e in rows]

    def _t_marker_detector(self, p, resolve):
        _check_keys(p, {*DETECTOR_DEFAULTS, "mask"}, "marker_detector")
        cfg = {
            "threshold": _num(p, "threshold", DETECTOR_DEFAULTS["threshold"], 0.1, 1.0),
            "group_threshold": _num(p, "group_threshold", DETECTOR_DEFAULTS["group_threshold"],
                                    0.0, 10.0),
            "dup_frac": _num(p, "dup_frac", DETECTOR_DEFAULTS["dup_frac"], 0.0, 0.1),
            "min_points": _num(p, "min_points", DETECTOR_DEFAULTS["min_points"], 1, 1000, int),
        }
        ls = _num(p, "long_side", DETECTOR_DEFAULTS["long_side"], 1, 10000, int)
        m, spec = self._mask(p.get("mask"), resolve)
        dets = self._raw_detections(ls)
        if m is not None:
            h, w = m.shape
            dets = [d for d in dets
                    if 0 <= int(d.y) < h and 0 <= int(d.x) < w and m[int(d.y), int(d.x)]]
        series = postprocess(dets, self.size, None, PostConfig(**cfg))
        return points_result("marker_detector", {**cfg, "long_side": ls, "mask": spec}, series)

    def _t_tick_calibration(self, p, resolve):
        _check_keys(p, {"mode", "x", "y"}, "tick_calibration")
        mode = p.get("mode", "given" if self.given_calibration else "auto")
        if mode == "given":
            if not self.given_calibration:
                raise ToolError("no calibration is given in this condition; use mode auto")
            cal = dict(self.given_calibration)
            if not cal.get("frame"):  # the plot frame by the frame rules alone
                frames = detect_axis_frames(to_gray(self.rgb) < DARK)
                if frames:
                    f = max(frames, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
                    cal["frame"] = [round(v, 1) for v in f]
            return _cal_result(cal, "the person's calibration", None)
        if mode == "manual":
            frame = None
            cal = {"x": p.get("x"), "y": p.get("y"), "frame": frame, "source": "manual"}
            try:
                check_calibration(cal)
            except (ValueError, TypeError, KeyError) as exc:
                raise ToolError(f"manual calibration: {exc}; give x and y as "
                                '{"scale": "linear|log", "ticks": [[px, value], ...]}') from exc
            return _cal_result(cal, "ticks given by the orchestrator", None)
        if mode != "auto":
            raise ToolError("mode must be auto, given or manual")
        if not self.allow_auto_calibration:
            raise ToolError("automatic calibration is not used in this condition")
        cal = calibrate_image(self.rgb)
        if cal is None:
            return {"kind": "calibration", "ok": False, "calibration": None, "frame": None,
                    "message": "no plot frame (axis lines) found"}
        dark = to_gray(self.rgb) < DARK
        xt, yt = detect_ticks(dark, cal.frame)
        frame = [round(v, 1) for v in cal.frame]
        marks = {"x": [round(v, 1) for v in xt], "y": [round(v, 1) for v in yt]}
        if not cal.ok:
            return {"kind": "calibration", "ok": False, "calibration": None, "frame": frame,
                    "tick_marks": marks,
                    "read": {ax: _axis_dict(f) for ax, f in (("x", cal.x_fit), ("y", cal.y_fit))},
                    "message": "tick labels could not be read on "
                    + " and ".join(ax for ax, f in (("x", cal.x_fit), ("y", cal.y_fit))
                                   if f is None)
                    + "; read them yourself and call tick_calibration with mode manual"}
        out = {"x": _axis_dict(cal.x_fit), "y": _axis_dict(cal.y_fit), "frame": frame,
               "source": "auto", "y_side": cal.y_side}
        return {**_cal_result(out, "frame rules + Tesseract tick OCR", marks)}

    def _t_to_values(self, p, resolve):
        _check_keys(p, {"result", "calibration"}, "to_values")
        r = resolve(_req(p, "result"))
        c = resolve(_req(p, "calibration"))
        if r.get("kind") != "points":
            raise ToolError("result must be a points result")
        if c.get("kind") != "calibration" or not c.get("ok"):
            raise ToolError("calibration must be a successful calibration result")
        return {"kind": "values", "series": series_to_values(r["series"], c["calibration"])}

    def _t_render_overlay(self, p, resolve):
        _check_keys(p, {"results", "calibration", "out", "max_side"}, "render_overlay")
        refs = p.get("results") or []
        if isinstance(refs, str):
            refs = [refs]
        im = Image.fromarray(self.rgb).convert("RGB")
        draw = ImageDraw.Draw(im)
        font = _font(max(12, round(max(im.size) / 60)))
        w, h = im.size
        r = max(3, round(max(w, h) / 220))
        legend = []
        k = 0
        for ref in refs:
            res = resolve(ref)
            if res.get("kind") != "points":
                raise ToolError(f"{ref} is not a points result")
            for i, s in enumerate(res["series"]):
                col = OVERLAY_COLORS[k % len(OVERLAY_COLORS)]
                k += 1
                for x, y in zip(s["x"], s["y"], strict=True):
                    draw.ellipse([x - r, y - r, x + r, y + r], outline=col, width=2)
                    draw.line([x - r - 2, y, x + r + 2, y], fill=col, width=1)
                    draw.line([x, y - r - 2, x, y + r + 2], fill=col, width=1)
                if s["x"]:
                    draw.text((s["x"][0] + r + 2, s["y"][0] - 2 * r), f"{ref}:{i}", fill=col,
                              font=font)
                legend.append({"result": ref, "index": i, "overlay_color": col,
                               "n": len(s["x"])})
        if p.get("calibration"):
            c = resolve(p["calibration"])
            cal = c.get("calibration") or {}
            frame = cal.get("frame") or c.get("frame")
            if frame:
                draw.rectangle(frame, outline="#00c000", width=1)
            for ax in ("x", "y"):
                axis = cal.get(ax)
                if not axis:
                    continue
                for _px, v in axis["ticks"]:
                    q = value_to_px(v, axis)
                    if q is None:
                        continue
                    if ax == "x":
                        draw.line([q, 0, q, h], fill="#00c000", width=1)
                        draw.text((q + 2, 2), f"{v:g}", fill="#008000", font=font)
                    else:
                        draw.line([0, q, w, q], fill="#00c000", width=1)
                        draw.text((2, q + 1), f"{v:g}", fill="#008000", font=font)
        max_side = _num(p, "max_side", 0, 0, 10000, int)
        if max_side and max(w, h) > max_side:
            s = max_side / max(w, h)
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        out = p.get("out")
        if out is None:
            if self.out_dir is None:
                raise ToolError("render_overlay needs out (a .png path)")
            self.out_dir.mkdir(parents=True, exist_ok=True)
            out = self.out_dir / f"overlay_{len(list(self.out_dir.glob('overlay_*.png'))) + 1}.png"
        out = Path(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        im.save(out)
        return {"kind": "overlay", "path": str(out), "series": legend}


def _font(size: int):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow < 10.1: the fixed bitmap font
        return ImageFont.load_default()


def _req(p: dict, key: str):
    if key not in p:
        raise ToolError(f"{key} is required")
    return p[key]


def _series(label: str, pts: list[tuple[float, float]]) -> dict:
    pts = sorted(pts)
    return {"label": label, "x": [x for x, _ in pts], "y": [y for _, y in pts]}


def _axis_dict(f) -> dict | None:
    if f is None:
        return None
    return {"scale": f.scale, "ticks": [[round(px, 2), v] for px, v in f.ticks],
            "fit": [f.slope, f.intercept]}


def _cal_result(cal: dict, how: str, marks: dict | None) -> dict:
    out = {"kind": "calibration", "ok": True, "calibration": cal, "frame": cal.get("frame"),
           "how": how}
    if marks is not None:
        out["tick_marks"] = marks
    return out
