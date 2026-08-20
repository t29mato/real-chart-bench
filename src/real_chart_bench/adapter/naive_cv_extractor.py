"""Naive classical-CV baseline implementing ModelRunnerPort (design §7.15).

Deliberately simple/weak reference point, not a competitive model: buckets
colored (non-gray) pixels into a fixed hue palette, takes the median pixel
y per column within each color bucket, and rescales via the task-provided
axis calibration (domain/pixel_calibration.py). Known limitations (documented,
not silently hidden): cannot see black/gray line series (indistinguishable
from axis/text), and uses the colored-pixel bounding box as a proxy for the
plot-area frame rather than true axis-tick detection.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pymupdf

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.pixel_calibration import PixelCalibration
from real_chart_bench.usecase.model_runner import ExtractionTask

# (name, hue_lo, hue_hi) in degrees; red wraps around 0/360.
_HUE_BUCKETS = [
    ("red", 340, 20),
    ("orange", 20, 45),
    ("yellow", 45, 70),
    ("green", 70, 170),
    ("cyan", 170, 200),
    ("blue", 200, 260),
    ("purple", 260, 340),
]

_MIN_PIXELS_PER_SERIES = 15
_SATURATION_THRESHOLD = 40
_BACKGROUND_LUMINANCE_THRESHOLD = 245


def _decode_rgb(image_bytes: bytes) -> np.ndarray:
    pixmap = pymupdf.Pixmap(image_bytes)
    if pixmap.alpha:
        pixmap = pymupdf.Pixmap(pixmap, 0)
    if pixmap.colorspace is None or pixmap.colorspace.name != "DeviceRGB":
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )[:, :, :3]


def _hue_degrees(rgb: np.ndarray) -> np.ndarray:
    r, g, b = (rgb[..., i].astype(np.float64) / 255.0 for i in range(3))
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc
    delta_safe = np.where(delta == 0, 1.0, delta)

    rc = (maxc - r) / delta_safe
    gc = (maxc - g) / delta_safe
    bc = (maxc - b) / delta_safe

    is_r = maxc == r
    is_g = (maxc == g) & ~is_r
    is_b = (maxc == b) & ~is_r & ~is_g

    hue = np.zeros_like(maxc)
    hue = np.where(is_r, (bc - gc) % 6.0, hue)
    hue = np.where(is_g, 2.0 + rc - bc, hue)
    hue = np.where(is_b, 4.0 + gc - rc, hue)
    return (hue / 6.0 % 1.0) * 360.0


def _hue_mask(hue: np.ndarray, lo: float, hi: float) -> np.ndarray:
    if lo < hi:
        return (hue >= lo) & (hue < hi)
    return (hue >= lo) | (hue < hi)  # wraps around 0/360 (red)


class NaiveCvModelRunner:
    def extract(self, task: ExtractionTask) -> list[Curve]:
        rgb = _decode_rgb(task.image_bytes)
        r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
        saturation = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        luminance = (r + g + b) / 3.0
        colored_mask = (saturation > _SATURATION_THRESHOLD) & (
            luminance < _BACKGROUND_LUMINANCE_THRESHOLD
        )

        if not colored_mask.any():
            return []

        ys, xs = np.nonzero(colored_mask)
        pixel_bbox = (float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1)
        calibration = PixelCalibration(
            pixel_bbox=pixel_bbox,
            x_range=task.x_range,
            y_range=task.y_range,
            x_scale=task.x_scale,
            y_scale=task.y_scale,
        )

        hue = _hue_degrees(rgb)
        curves = []
        for name, lo, hi in _HUE_BUCKETS:
            mask = colored_mask & _hue_mask(hue, lo, hi)
            if mask.sum() < _MIN_PIXELS_PER_SERIES:
                continue
            curve = self._trace_curve(mask, calibration, task, series_label=name)
            if curve is not None:
                curves.append(curve)
        return curves

    @staticmethod
    def _trace_curve(
        mask: np.ndarray,
        calibration: PixelCalibration,
        task: ExtractionTask,
        *,
        series_label: str,
    ) -> Curve | None:
        ys, xs = np.nonzero(mask)
        per_column: dict[int, list[int]] = defaultdict(list)
        for x, y in zip(xs.tolist(), ys.tolist(), strict=True):
            per_column[x].append(y)

        x_values, y_values = [], []
        for x in sorted(per_column):
            median_y = float(np.median(per_column[x]))
            data_x, data_y = calibration.to_data(float(x), median_y)
            x_values.append(data_x)
            y_values.append(data_y)

        if len(x_values) < 2:
            return None
        return Curve(
            x_values=tuple(x_values),
            y_values=tuple(y_values),
            x_scale=task.x_scale,
            series_label=series_label,
        )
