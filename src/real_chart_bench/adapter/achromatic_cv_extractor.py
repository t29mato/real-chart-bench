"""Achromatic classic-CV baseline implementing ModelRunnerPort (design §7.15).

Companion to naive_cv_extractor.py's hue-bucket baseline, built specifically
for the failure that one documents and cannot handle: black/grey line series,
which hue-based (saturated-colour) detection is structurally blind to. Its
colour mask finds zero coloured pixels on such figures and scores exactly
0.0 -- confirmed for several real figures in
docs/experiments/2026-09-02-failure-analysis.md.

Approach: works on luminance, not hue. Real papers commonly distinguish
achromatic series in print by *distinct grey levels* (solid black vs. mid
grey vs. light grey) rather than by stroke style, so this extractor:

  1. builds a luminance histogram of achromatic (low-saturation,
     non-background) pixels;
  2. finds its dominant peaks via local-maxima detection + greedy
     non-max suppression. This *adapts to whatever grey levels a given
     figure actually uses* instead of guessing fixed bands up front (unlike
     naive_cv_extractor's fixed hue bands, which is fine for hue because hue
     buckets are a small fixed wheel -- there is no equivalent fixed
     vocabulary for "how many grey levels a paper happens to use"). It also
     lets anti-aliased edge pixels (the continuous grey ramp between a
     stroke and the white background) get folded into their nearest real
     peak rather than forming spurious extra "series" of their own, since
     they rarely have enough pixels at any single intermediate level to
     form a peak themselves;
  3. assigns each achromatic pixel to its nearest peak (within a radius) to
     form one bucket per detected grey level, then traces each bucket the
     same way naive_cv_extractor traces a hue bucket: median pixel y per
     column, rescaled via the task's PixelCalibration.

Known limitations (documented, not silently hidden -- see
docs/experiments/2026-09-02-failure-analysis.md and design §7.4x for the
project convention of stating these plainly):

  - like naive_cv_extractor, uses the achromatic-pixel bounding box as a
    proxy for the plot-area frame rather than true axis-tick detection.
  - cannot see saturated colour at all -- the exact converse blind spot of
    naive_cv_extractor. Together the two baselines cover more of the colour
    space than either alone, but neither is a general-purpose extractor and
    a figure that mixes coloured and achromatic series needs both baselines
    run and merged by hand; this extractor does not do that merge itself.
  - no text/axis-frame exclusion: a black axis border, tick marks, or dense
    figure text drawn at the same (or a nearby) grey level as a real curve
    is not distinguished from curve ink and can appear in, or contaminate,
    a traced series. There is no OCR or line-vs-shape classification here.
  - like the hue baseline, cannot separate two series that land in the same
    detected grey level, or trace a single series across a self-crossing or
    a crossing with another series of the same level (no ridge tracing
    across occlusions, no connected-component splitting).
  - assumes the print convention of a handful of distinct, mostly-flat grey
    tones. A genuinely continuous grey gradient (e.g. a colormap rendered in
    greyscale) will not cleanly separate into peaks and may be under- or
    over-segmented.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pymupdf

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.pixel_calibration import PixelCalibration
from real_chart_bench.usecase.model_runner import ExtractionTask

_SATURATION_THRESHOLD = 40  # complement of naive_cv_extractor's "colored" threshold
_BACKGROUND_LUMINANCE_THRESHOLD = 245
_MIN_PIXELS_PER_SERIES = 15

_PEAK_SMOOTH_WINDOW = 5
_PEAK_MERGE_DISTANCE = 18.0  # luminance units; peaks closer than this are one series
_MAX_CLUSTER_RADIUS = 30.0  # a pixel farther than this from every peak joins none
_MAX_BUCKETS = 8
_PEAK_MIN_FRACTION = 0.01  # a peak needs >=1% of achromatic pixels, or the floor above


def _decode_rgb(image_bytes: bytes) -> np.ndarray:
    pixmap = pymupdf.Pixmap(image_bytes)
    if pixmap.alpha:
        pixmap = pymupdf.Pixmap(pixmap, 0)
    if pixmap.colorspace is None or pixmap.colorspace.name != "DeviceRGB":
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )[:, :, :3]


def _luminance_histogram_peaks(luminance_values: np.ndarray) -> list[float]:
    """Dominant grey levels among `luminance_values` (0-255 float array),
    found by smoothing a histogram, taking local maxima above a minimum
    count, and greedily keeping the strongest ones that are at least
    `_PEAK_MERGE_DISTANCE` apart. Returns levels sorted ascending
    (darkest first)."""
    counts = np.bincount(
        np.clip(luminance_values.round().astype(np.int64), 0, 255), minlength=256
    ).astype(np.float64)
    kernel = np.ones(_PEAK_SMOOTH_WINDOW) / _PEAK_SMOOTH_WINDOW
    smoothed = np.convolve(counts, kernel, mode="same")

    min_count = max(_MIN_PIXELS_PER_SERIES, luminance_values.size * _PEAK_MIN_FRACTION)
    left = np.concatenate(([-1.0], smoothed[:-1]))
    right = np.concatenate((smoothed[1:], [-1.0]))
    is_local_max = (smoothed >= min_count) & (smoothed >= left) & (smoothed >= right)
    candidates = sorted(
        ((float(smoothed[i]), float(i)) for i in np.nonzero(is_local_max)[0]),
        key=lambda c: c[0],
        reverse=True,
    )

    peaks: list[float] = []
    for _, level in candidates:
        if len(peaks) >= _MAX_BUCKETS:
            break
        if all(abs(level - p) >= _PEAK_MERGE_DISTANCE for p in peaks):
            peaks.append(level)
    return sorted(peaks)


def _assign_pixels_to_peaks(
    rgb: np.ndarray, valid_mask: np.ndarray, peaks: list[float]
) -> dict[float, np.ndarray]:
    """Buckets each True pixel in `valid_mask` by its nearest peak (within
    `_MAX_CLUSTER_RADIUS`); pixels closer to no peak than that are dropped
    as noise/ramp. Returns {peak: boolean mask}, omitting peaks that ended
    up with no pixels."""
    ys, xs = np.nonzero(valid_mask)
    r = rgb[ys, xs, 0].astype(np.float64)
    g = rgb[ys, xs, 1].astype(np.float64)
    b = rgb[ys, xs, 2].astype(np.float64)
    luminance = (r + g + b) / 3.0

    peaks_arr = np.array(peaks)
    diffs = np.abs(luminance[:, None] - peaks_arr[None, :])
    nearest_idx = np.argmin(diffs, axis=1)
    nearest_dist = diffs[np.arange(len(luminance)), nearest_idx]
    keep = nearest_dist <= _MAX_CLUSTER_RADIUS

    buckets: dict[float, np.ndarray] = {}
    for k, peak in enumerate(peaks):
        sel = keep & (nearest_idx == k)
        if not sel.any():
            continue
        bucket_mask = np.zeros(valid_mask.shape, dtype=bool)
        bucket_mask[ys[sel], xs[sel]] = True
        buckets[peak] = bucket_mask
    return buckets


class AchromaticCvModelRunner:
    def extract(self, task: ExtractionTask) -> list[Curve]:
        rgb = _decode_rgb(task.image_bytes)
        r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
        saturation = np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)
        luminance = (r + g + b) / 3.0
        achromatic_mask = (saturation <= _SATURATION_THRESHOLD) & (
            luminance < _BACKGROUND_LUMINANCE_THRESHOLD
        )

        if not achromatic_mask.any():
            return []

        peaks = _luminance_histogram_peaks(luminance[achromatic_mask])
        if not peaks:
            return []

        buckets = _assign_pixels_to_peaks(rgb, achromatic_mask, peaks)
        if not buckets:
            return []

        ys, xs = np.nonzero(achromatic_mask)
        pixel_bbox = (float(xs.min()), float(ys.min()), float(xs.max()) + 1, float(ys.max()) + 1)
        calibration = PixelCalibration(
            pixel_bbox=pixel_bbox,
            x_range=task.x_range,
            y_range=task.y_range,
            x_scale=task.x_scale,
            y_scale=task.y_scale,
        )

        curves = []
        for peak in sorted(buckets):
            mask = buckets[peak]
            if mask.sum() < _MIN_PIXELS_PER_SERIES:
                continue
            curve = self._trace_curve(
                mask, calibration, task, series_label=f"grey~{round(peak)}"
            )
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
