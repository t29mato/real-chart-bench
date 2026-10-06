"""Pure parts of the fixed tools of 方式D (docs/design/local-model.md,
「方式D: 司令塔 + 道具」): candidate series colours, masks, axis calibration
from ticks, pixels -> values, and the result shapes the orchestrator sees.

The tools themselves (image loading, the detector, Tesseract, drawing) are
in adapter/orchestrator_tools.py; everything here works on arrays and dicts.

Shapes shared by every tool:

- a point series: ``{"label": str, "x": [px...], "y": [px...]}`` in the
  image file's pixels (origin top-left, y down) -- the scorer's pixel answer;
- a calibration: ``{"x": {"scale": "linear|log", "ticks": [[px, value], ...]},
  "y": {...}, "frame": [x0, y0, x1, y1] | None, "source": str}``, the values
  as printed on the ticks;
- a mask spec: ``{"include": [[x0, y0, x1, y1], ...], "exclude": [...],
  "frame": [x0, y0, x1, y1], "frame_margin": px}``, every key optional.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

Rect = Sequence[float]


# ---------------------------------------------------------------- colours


def parse_color(color) -> tuple[int, int, int]:
    """'#rrggbb', 'rrggbb' or [r, g, b] -> (r, g, b)."""
    if isinstance(color, str):
        s = color.strip().lstrip("#")
        if len(s) != 6:
            raise ValueError(f"colour {color!r} is not #rrggbb")
        return int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16)
    if isinstance(color, Sequence) and len(color) == 3:
        r, g, b = (int(v) for v in color)
        if not all(0 <= v <= 255 for v in (r, g, b)):
            raise ValueError(f"colour {color!r} out of 0-255")
        return r, g, b
    raise ValueError(f"colour {color!r} is not #rrggbb or [r, g, b]")


def to_hex(rgb: Sequence[int]) -> str:
    return "#" + "".join(f"{int(v):02x}" for v in rgb[:3])


def _dist_pct(a: Sequence[float], b: Sequence[float]) -> float:
    """starry-digitizer's colour distance, in % (matchColor's left side)."""
    return sum((float(x) - float(y)) ** 2 for x, y in zip(a, b, strict=True)) / (255.0**2 * 3) * 100


def dominant_colors(
    rgb: np.ndarray,
    mask: np.ndarray | None = None,
    k: int = 8,
    *,
    white_level: int = 230,
    merge_pct: float = 2.0,
    step: int = 16,
) -> list[dict]:
    """The image's main non-background colours, most pixels first: candidate
    series colours for symbol_extract / line_extract.

    Pixels are binned on a `step` grid per channel; near-white bins (every
    channel >= white_level) are background and left out. Bins are merged,
    largest first, into the first colour within `merge_pct` (starry-digitizer's
    distance); a colour's RGB is the pixel mean of its bins. Black and grey
    (the axes, the text) are kept and flagged ``achromatic``. A bin holding
    more than half of the pixels is background too (a dark background)."""
    px = rgb[..., :3].reshape(-1, 3)
    if mask is not None:
        px = px[mask.reshape(-1).astype(bool)]
    if len(px) == 0:
        return []
    px = px.astype(np.int64)
    keep = ~(px >= white_level).all(axis=1)
    # a figure saved on a dark or coloured background (a flattened
    # transparent PNG): the one bin covering most of the image is background
    q_all = px // step
    k_all = q_all[:, 0] * 4096 + q_all[:, 1] * 64 + q_all[:, 2]
    vals, cnt = np.unique(k_all, return_counts=True)
    if cnt.max() > 0.5 * len(px):
        keep &= k_all != vals[cnt.argmax()]
    px = px[keep]
    total = len(px)
    if total == 0:
        return []
    q = px // step
    key = (q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]).astype(np.int64)
    uniq, inv, counts = np.unique(key, return_inverse=True, return_counts=True)
    sums = np.zeros((len(uniq), 3), dtype=np.float64)
    np.add.at(sums, inv, px)
    means = sums / counts[:, None]
    clusters: list[dict] = []
    for i in np.argsort(-counts, kind="stable"):
        c = means[i]
        for cl in clusters:
            if _dist_pct(cl["sum"] / cl["n"], c) < merge_pct:
                cl["sum"] += sums[i]
                cl["n"] += int(counts[i])
                break
        else:
            clusters.append({"sum": sums[i].copy(), "n": int(counts[i])})
    clusters.sort(key=lambda cl: -cl["n"])
    out = []
    for cl in clusters[:k]:
        mean = cl["sum"] / cl["n"]
        rgb_i = [int(round(v)) for v in mean]
        out.append({
            "color": to_hex(rgb_i),
            "rgb": rgb_i,
            "n_pixels": cl["n"],
            "fraction": round(cl["n"] / total, 4),
            "achromatic": max(rgb_i) - min(rgb_i) < 40,
        })
    return out


# ---------------------------------------------------------------- masks


def _clip_rect(r: Rect, w: int, h: int) -> tuple[int, int, int, int]:
    if len(r) != 4:
        raise ValueError(f"rectangle {r!r} is not [x0, y0, x1, y1]")
    x0, y0, x1, y1 = (float(v) for v in r)
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return (max(0, math.floor(x0)), max(0, math.floor(y0)),
            min(w, math.ceil(x1) + 1), min(h, math.ceil(y1) + 1))


def build_mask(height: int, width: int, spec: dict | None) -> np.ndarray | None:
    """A boolean mask (True = examined) from a mask spec; None = no mask.

    include: union of rectangles (default: the whole image); frame: the plot
    frame, shrunk inwards by frame_margin px (default 2, so the axis lines
    themselves are left out); exclude: rectangles removed (a legend box).
    Rectangles are [x0, y0, x1, y1] in image pixels, both ends inclusive."""
    if not spec:
        return None
    unknown = set(spec) - {"include", "exclude", "frame", "frame_margin"}
    if unknown:
        raise ValueError(f"unknown mask keys {sorted(unknown)}")
    m = np.zeros((height, width), dtype=bool)
    includes = spec.get("include") or []
    if includes:
        for r in includes:
            x0, y0, x1, y1 = _clip_rect(r, width, height)
            m[y0:y1, x0:x1] = True
    else:
        m[:, :] = True
    if spec.get("frame"):
        margin = float(spec.get("frame_margin", 2))
        fx0, fy0, fx1, fy1 = (float(v) for v in spec["frame"])
        inner = [fx0 + margin, fy0 + margin, fx1 - margin, fy1 - margin]
        f = np.zeros_like(m)
        if inner[2] >= inner[0] and inner[3] >= inner[1]:
            x0, y0, x1, y1 = _clip_rect(inner, width, height)
            f[y0:y1, x0:x1] = True
        m &= f
    for r in spec.get("exclude") or []:
        x0, y0, x1, y1 = _clip_rect(r, width, height)
        m[y0:y1, x0:x1] = False
    return m


def filter_series_by_mask(series: list[dict], mask: np.ndarray | None) -> list[dict]:
    """Drop points outside the mask (for tools that find points without
    looking at pixels, i.e. the detector). Empty series are dropped."""
    if mask is None:
        return series
    h, w = mask.shape
    out = []
    for s in series:
        keep = [
            (x, y) for x, y in zip(s["x"], s["y"], strict=True)
            if 0 <= int(y) < h and 0 <= int(x) < w and mask[int(y), int(x)]
        ]
        if keep:
            out.append({**s, "x": [p[0] for p in keep], "y": [p[1] for p in keep]})
    return out


# ---------------------------------------------------------------- calibration


def fit_ticks(ticks: Sequence[Sequence[float]], scale: str) -> tuple[float, float]:
    """px = slope * t(value) + intercept by least squares over (px, value)
    ticks, t = log10 on a log axis. Needs two ticks with different values."""
    if scale not in ("linear", "log"):
        raise ValueError(f"scale {scale!r} is not linear or log")
    pts = []
    for px, v in ticks:
        if scale == "log":
            if v <= 0:
                raise ValueError(f"log axis tick value {v} is not positive")
            v = math.log10(v)
        pts.append((float(px), float(v)))
    if len({v for _, v in pts}) < 2:
        raise ValueError("an axis needs two ticks with different values")
    n = len(pts)
    mv = sum(v for _, v in pts) / n
    mp = sum(p for p, _ in pts) / n
    svv = sum((v - mv) ** 2 for _, v in pts)
    spv = sum((p - mp) * (v - mv) for p, v in pts)
    slope = spv / svv
    if slope == 0:
        raise ValueError("ticks at the same pixel")
    return slope, mp - slope * mv


def _line(axis: dict) -> tuple[float, float]:
    """An axis's (slope, intercept): its own "fit" when the tool that made it
    fitted robustly (the automatic calibration), else least squares."""
    if axis.get("fit"):
        slope, intercept = (float(v) for v in axis["fit"])
        if slope == 0:
            raise ValueError("axis fit has zero slope")
        return slope, intercept
    return fit_ticks(axis["ticks"], axis["scale"])


def px_to_value(px: float, axis: dict) -> float:
    slope, intercept = _line(axis)
    t = (px - intercept) / slope
    return 10**t if axis["scale"] == "log" else t


def value_to_px(value: float, axis: dict) -> float | None:
    slope, intercept = _line(axis)
    if axis["scale"] == "log":
        if value <= 0:
            return None
        value = math.log10(value)
    return slope * value + intercept


def calibration_from_person(task: dict) -> dict:
    """The person's calibration as handed to the agents in the pixcal tasks
    (x_ticks: pixel_x / value, y_ticks: pixel_y / value, scales)."""
    return {
        "x": {"scale": task["x_scale"],
              "ticks": [[t["pixel_x"], t["value"]] for t in task["x_ticks"]]},
        "y": {"scale": task["y_scale"],
              "ticks": [[t["pixel_y"], t["value"]] for t in task["y_ticks"]]},
        "frame": None,
        "source": "person",
    }


def check_calibration(cal: dict) -> None:
    for ax in ("x", "y"):
        if ax not in cal or not isinstance(cal[ax], dict):
            raise ValueError(f"calibration has no {ax} axis")
        if cal[ax].get("scale") not in ("linear", "log"):
            raise ValueError(f"{ax} scale {cal[ax].get('scale')!r} is not linear or log")
        _line(cal[ax])


def series_to_values(series: list[dict], cal: dict) -> list[dict]:
    """Pixel series -> value series through a calibration."""
    check_calibration(cal)
    return [
        {**s,
         "x": [px_to_value(v, cal["x"]) for v in s["x"]],
         "y": [px_to_value(v, cal["y"]) for v in s["y"]]}
        for s in series
    ]


# ---------------------------------------------------------------- results


def points_result(tool: str, params: dict, series: list[dict]) -> dict:
    """An extraction tool's result: its series and counts."""
    return {
        "kind": "points",
        "tool": tool,
        "params": params,
        "series": series,
        "n_series": len(series),
        "n_points": sum(len(s["x"]) for s in series),
    }


def summarize_points(result: dict, scale: float = 1.0, max_points: int = 6) -> dict:
    """What the orchestrator model reads of a points result: per series the
    count, the bounding box and the first points, in its view's pixels
    (image pixels x scale)."""
    out = []
    for i, s in enumerate(result["series"]):
        xs = [v * scale for v in s["x"]]
        ys = [v * scale for v in s["y"]]
        item = {"index": i, "label": s.get("label"), "n": len(xs)}
        if xs:
            item["bbox"] = [round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys))]
            item["first"] = [[round(x), round(y)] for x, y in list(zip(xs, ys, strict=True))[
                :max_points]]
        out.append(item)
    return {"tool": result["tool"], "n_series": result["n_series"],
            "n_points": result["n_points"], "series": out}
