"""Plot areas from the owner-reviewed tick-mark pixel positions
(data/verified_pairs/axis_pixel_candidates.json), for methods that detect
curves but not axes -- LineFormer's tick-calibrated row (design §7.66).

An entry is used only while it still describes the figure as scored: same
image file, tick labels equal to the registry's x_range/y_range, every label
present, and no `pixel_coords_stale_since` (set when an image was re-cropped
or rotated). Anything else is left out rather than guessed.

Returned boxes are PixelCalibration.pixel_bbox tuples (x0, y0, x1, y1): x0/x1
the x_range ticks, y0 the y_range[1] tick (top), y1 the y_range[0] tick.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_LABELS = ("x_min_label", "x_max_label", "y_min_label", "y_max_label")


def _same(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-9 * max(1.0, abs(b))


def load_tick_plot_areas(
    axis_path: Path, pairings: Iterable[Any]
) -> dict[tuple[str, str], tuple[float, float, float, float]]:
    entries = {
        (e["paper_id"], e["figure_id"]): e
        for e in json.loads(axis_path.read_text())
        if "paper_id" in e
    }
    areas = {}
    for p in pairings:
        key = (p.paper_id, p.figure_id)
        e = entries.get(key)
        if e is None or e.get("pixel_coords_stale_since") or e.get("image_path") != p.image_path:
            continue
        labels = [e.get(k) for k in _LABELS]
        box = e.get("pixel_bbox_mean")
        if None in labels or not box:
            continue
        ranges = (*p.x_range, *p.y_range)
        if not all(_same(float(a), float(b)) for a, b in zip(labels, ranges, strict=True)):
            continue
        areas[key] = (
            float(box["x_min_px"]),
            float(box["y_max_px"]),
            float(box["x_max_px"]),
            float(box["y_min_px"]),
        )
    return areas


def plot_area_from_ticks(
    cal: dict, x_range, y_range, x_scale: str, y_scale: str
) -> tuple[float, float, float, float]:
    """PixelCalibration.pixel_bbox for the task's ranges, from any two ticks
    per axis (data/verified_pairs/tick_calibration.json, design 7.77): each
    range end is placed by the line through the two ticks (log10 on a log
    axis). Returns (x0, y0, x1, y1) with y0 at y_range[1] (top)."""

    def at(ticks, value, scale):
        (a, b) = ticks
        t = math.log10 if scale == "log" else (lambda v: v)
        frac = (t(value) - t(a["value"])) / (t(b["value"]) - t(a["value"]))
        return a["px"] + frac * (b["px"] - a["px"])

    return (
        at(cal["x"], x_range[0], x_scale),
        at(cal["y"], y_range[1], y_scale),
        at(cal["x"], x_range[1], x_scale),
        at(cal["y"], y_range[0], y_scale),
    )


def load_tick_calibration(path: Path) -> dict[tuple[str, str], dict]:
    """(paper_id, figure_id) -> calibration record of tick_calibration.json."""
    return {(c["paper_id"], c["figure_id"]): c for c in json.loads(path.read_text())["figures"]}
