"""Training data for the local extractor (docs/design/local-model.md).

One label per image, in one shape (labels.jsonl), whatever its source:
an existing dataset, the synthetic generator, or Starrydata x a real figure.
Pixel points are in the image file's grid (origin top-left, y down); axis
ticks are (pixel, value) pairs with values as printed (design 7.82). Unknown
fields are null, never guessed.

Every path that builds training data passes assert_no_benchmark_leak: no
figure of a paper that real-chart-bench evaluates on may be trained on.
"""

from __future__ import annotations

import math
from collections.abc import Iterable

SOURCES_WITH_PAPERS = {"starrydata"}
SCALES = {"linear", "log"}
MARKERS = {"circle", "square", "triangle", "diamond", "cross", "other"}


class BenchmarkLeakError(ValueError):
    pass


def _num(v) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _point_list(points, name: str, errors: list[str]) -> int | None:
    if points is None:
        return None
    if not isinstance(points, list) or not all(
        isinstance(p, list | tuple) and len(p) == 2 and all(_num(c) for c in p) for p in points
    ):
        errors.append(f"{name} must be a list of [x, y] numbers")
        return None
    return len(points)


def validate_label(label: dict) -> list[str]:
    """Problems with one label line; an empty list means it is usable."""
    errors: list[str] = []
    for key in ("image", "width", "height", "source", "license", "series"):
        if label.get(key) is None:
            errors.append(f"{key} is required")
    w, h = label.get("width"), label.get("height")
    if label.get("source") in SOURCES_WITH_PAPERS and not label.get("paper_id"):
        errors.append("a real figure needs its paper_id (benchmark exclusion)")
    axes = label.get("axes")
    if axes is not None:
        for name in ("x", "y"):
            axis = axes.get(name) or {}
            if axis.get("scale") not in SCALES:
                errors.append(f"axes.{name}.scale must be one of {sorted(SCALES)}")
            ticks = axis.get("ticks") or []
            if len(ticks) < 2 or not all(_num(t.get("px")) and _num(t.get("value")) for t in ticks):
                errors.append(f"axes.{name}.ticks needs at least two (px, value) pairs")
    for i, s in enumerate(label.get("series") or []):
        if s.get("marker") is not None and s["marker"] not in MARKERS:
            errors.append(f"series[{i}].marker must be one of {sorted(MARKERS)} or null")
        n_px = _point_list(s.get("points_px"), f"series[{i}].points_px", errors)
        n_val = _point_list(s.get("points_value"), f"series[{i}].points_value", errors)
        if n_px is None and s.get("points_px") is None:
            errors.append(f"series[{i}].points_px is required")
        if n_px is not None and n_val is not None and n_px != n_val:
            errors.append(f"series[{i}].points_value has {n_val} points, points_px has {n_px}")
        if n_px and _num(w) and _num(h):
            outside = [p for p in s["points_px"] if not (0 <= p[0] <= w and 0 <= p[1] <= h)]
            if outside:
                errors.append(f"series[{i}]: {len(outside)} points_px outside the image")
    return errors


def assert_no_benchmark_leak(labels: Iterable[dict], benchmark_paper_ids: set[str]) -> None:
    """Raise if any label comes from a paper the benchmark evaluates on."""
    leaked = sorted(
        {str(lab["paper_id"]) for lab in labels if lab.get("paper_id") is not None}
        & {str(p) for p in benchmark_paper_ids}
    )
    if leaked:
        raise BenchmarkLeakError(f"benchmark papers in training data: {leaked}")


def _axis_coord(value: float, scale: str) -> float:
    if scale == "log":
        if value <= 0:
            raise ValueError(f"log axis cannot place the non-positive value {value}")
        return math.log10(value)
    return float(value)


def axis_value_to_px(value: float, axis: dict) -> float:
    """Where a value sits along one axis of a label, from that axis's ticks.

    The ticks are fitted by least squares (pixel against value, or against
    log10 value on a log axis), so one tick box a pixel off barely moves the
    axis; values past the outermost ticks extrapolate.
    """
    scale = axis["scale"]
    pts = [(_axis_coord(t["value"], scale), float(t["px"])) for t in axis["ticks"]]
    n = len(pts)
    mean_v = sum(v for v, _ in pts) / n if n else 0.0
    mean_p = sum(p for _, p in pts) / n if n else 0.0
    var = sum((v - mean_v) ** 2 for v, _ in pts)
    if n < 2 or var == 0:
        raise ValueError("an axis needs ticks with at least two distinct values")
    slope = sum((v - mean_v) * (p - mean_p) for v, p in pts) / var
    return mean_p + slope * (_axis_coord(value, scale) - mean_v)


def max_axis_residual_px(label: dict) -> float | None:
    """Worst pixel gap between a label's points_px and its points_value
    projected through its own axis ticks; None when it cannot be checked."""
    axes = label.get("axes")
    if not axes:
        return None
    worst: float | None = None
    for s in label.get("series") or []:
        values = s.get("points_value")
        if not values:
            continue
        for (px, py), (vx, vy) in zip(s["points_px"], values, strict=True):
            gap = max(
                abs(axis_value_to_px(vx, axes["x"]) - px),
                abs(axis_value_to_px(vy, axes["y"]) - py),
            )
            worst = gap if worst is None else max(worst, gap)
    return worst
