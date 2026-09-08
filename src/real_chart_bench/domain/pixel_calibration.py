"""Pixel-space <-> data-space coordinate mapping (design §7.15: evaluation
harness).

v0 scope decision: the harness evaluates curve-tracing ability *given* axis
calibration (the pixel bounding box of the plot area, and the data-space
range it corresponds to), not full end-to-end chart understanding including
reading axis tick labels. This mirrors how CHART-Infographics separates
"visual element detection" (task 6a) from "data extraction" (task 6b) —
see design §3.1. Pure math — no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from real_chart_bench.domain.curve import ScaleType


@dataclass(frozen=True)
class PixelCalibration:
    """Maps a pixel-space plot-area bounding box to a data-space range.

    Pixel y increases downward (image convention); data y increases upward
    (chart convention) — the mapping inverts y accordingly.
    """

    pixel_bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1)
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_scale: ScaleType = ScaleType.LINEAR
    # design §7.25: y_scale only matters here (pixel->data extraction) — the
    # evaluation metric (domain/metrics.py) compares curves' raw data values
    # directly and doesn't need to know how the source chart was rendered,
    # so unlike x_scale this has no counterpart on Curve/the metric layer.
    y_scale: ScaleType = ScaleType.LINEAR

    def to_data(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        px0, py0, px1, py1 = self.pixel_bbox
        x_frac = _safe_frac(pixel_x, px0, px1)
        y_frac = 1.0 - _safe_frac(pixel_y, py0, py1)  # invert: pixel-down -> data-up

        x = _scale_frac(x_frac, self.x_range, self.x_scale, axis_name="x")
        y = _scale_frac(y_frac, self.y_range, self.y_scale, axis_name="y")
        return x, y

    def to_pixel(self, x: float, y: float) -> tuple[float, float]:
        """Inverse of :meth:`to_data`: where a data point sits in the image.

        Extrapolates past the bounding box rather than clamping — ground truth
        routinely lies outside the outermost printed tick (design §7.57).
        """
        px0, py0, px1, py1 = self.pixel_bbox
        x_frac = _unscale_frac(x, self.x_range, self.x_scale, axis_name="x")
        y_frac = _unscale_frac(y, self.y_range, self.y_scale, axis_name="y")

        pixel_x = px0 + x_frac * (px1 - px0)
        pixel_y = py1 - y_frac * (py1 - py0)  # invert: data-up -> pixel-down
        return pixel_x, pixel_y


def _scale_frac(
    frac: float, value_range: tuple[float, float], scale: ScaleType, *, axis_name: str
) -> float:
    lo, hi = value_range
    if scale is ScaleType.LOG:
        if lo <= 0 or hi <= 0:
            raise ValueError(
                f"log {axis_name}_scale requires a strictly positive {axis_name}_range"
            )
        log_value = math.log10(lo) + frac * (math.log10(hi) - math.log10(lo))
        return 10**log_value
    return lo + frac * (hi - lo)


def _unscale_frac(
    value: float, value_range: tuple[float, float], scale: ScaleType, *, axis_name: str
) -> float:
    lo, hi = value_range
    if scale is ScaleType.LOG:
        if lo <= 0 or hi <= 0:
            raise ValueError(
                f"log {axis_name}_scale requires a strictly positive {axis_name}_range"
            )
        if value <= 0:
            raise ValueError(
                f"log {axis_name}_scale requires a strictly positive {axis_name} value"
            )
        return _safe_frac(math.log10(value), math.log10(lo), math.log10(hi))
    return _safe_frac(value, lo, hi)


def _safe_frac(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return (value - lo) / (hi - lo)
