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

    def to_data(self, pixel_x: float, pixel_y: float) -> tuple[float, float]:
        px0, py0, px1, py1 = self.pixel_bbox
        x_frac = _safe_frac(pixel_x, px0, px1)
        y_frac = 1.0 - _safe_frac(pixel_y, py0, py1)  # invert: pixel-down -> data-up

        x_lo, x_hi = self.x_range
        if self.x_scale is ScaleType.LOG:
            if x_lo <= 0 or x_hi <= 0:
                raise ValueError("log x_scale requires a strictly positive x_range")
            log_x = math.log10(x_lo) + x_frac * (math.log10(x_hi) - math.log10(x_lo))
            x = 10**log_x
        else:
            x = x_lo + x_frac * (x_hi - x_lo)

        y_lo, y_hi = self.y_range
        y = y_lo + y_frac * (y_hi - y_lo)
        return x, y


def _safe_frac(value: float, lo: float, hi: float) -> float:
    if hi == lo:
        return 0.0
    return (value - lo) / (hi - lo)
