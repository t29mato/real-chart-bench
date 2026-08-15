"""Composite multi-panel figure layout detection (2026-08-16 技術調査,
design §7.10/§7.11: "figure_name refers to a single panel, but a PDF only
yields the whole composite figure" — the open problem shared with
deep-digitizer).

Heuristic: scientific figures are almost always laid out on a near-uniform
(typically white) background, with panels separated by a clear background
gutter. Detecting rows/columns that are almost entirely background, and
treating gutter runs above a minimum width as real panel separators, turns
out to be a robust, dependency-light way to recover a panel grid without a
trained detector — deliberately simple, since a full ML-based approach is
unwarranted investment before knowing whether basic layout analysis is
"good enough" (mirrors the automation-investment posture in design §7.10).

Pure numpy array logic only — no image codec, no I/O. Decoding raw image
bytes into a luminance array is an adapter's job (see
adapter/panel_layout.py), keeping this module trivially reusable by any
consumer (including deep-digitizer) without pulling in real-chart-bench's
own domain types.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_LABELS = "abcdefghijklmnopqrstuvwxyz"

_DEFAULT_BACKGROUND_THRESHOLD = 245
_DEFAULT_GUTTER_FRACTION = 0.98
_DEFAULT_MIN_GUTTER_PX = 3
_DEFAULT_MIN_CONTENT_FRACTION = 0.02


@dataclass(frozen=True)
class PanelRegion:
    label: str  # "a", "b", "c", ... in row-major reading order
    row: int  # 0-indexed grid row
    col: int  # 0-indexed grid column
    bbox: tuple[int, int, int, int]  # (x0, y0, x1, y1) pixel coords, half-open


def detect_panel_grid(
    luminance: np.ndarray,
    *,
    background_threshold: int = _DEFAULT_BACKGROUND_THRESHOLD,
    gutter_fraction: float = _DEFAULT_GUTTER_FRACTION,
    min_gutter_px: int = _DEFAULT_MIN_GUTTER_PX,
    min_content_fraction: float = _DEFAULT_MIN_CONTENT_FRACTION,
) -> tuple[PanelRegion, ...]:
    if luminance.ndim != 2:
        raise ValueError(f"luminance must be a 2D array, got shape {luminance.shape}")

    height, width = luminance.shape
    is_background = luminance >= background_threshold

    row_bands = _content_bands(is_background.mean(axis=1), gutter_fraction, min_gutter_px, height)
    col_bands = _content_bands(is_background.mean(axis=0), gutter_fraction, min_gutter_px, width)

    if len(row_bands) <= 1 and len(col_bands) <= 1:
        # No confidently-detected internal gutter: don't guess a crop, just
        # hand back the whole image as a single "panel".
        return (PanelRegion(label="a", row=0, col=0, bbox=(0, 0, width, height)),)

    regions = []
    for row_index, (y0, y1) in enumerate(row_bands):
        for col_index, (x0, x1) in enumerate(col_bands):
            cell = is_background[y0:y1, x0:x1]
            content_fraction = 1.0 - float(cell.mean()) if cell.size else 0.0
            if content_fraction < min_content_fraction:
                continue
            regions.append((row_index, col_index, (x0, y0, x1, y1)))

    return tuple(
        PanelRegion(label=_LABELS[i], row=row_index, col=col_index, bbox=bbox)
        for i, (row_index, col_index, bbox) in enumerate(regions)
    )


def _content_bands(
    background_fraction: np.ndarray,
    gutter_fraction: float,
    min_gutter_px: int,
    length: int,
) -> list[tuple[int, int]]:
    """Splits [0, length) into content bands, separated by runs of
    background_fraction >= gutter_fraction that are at least min_gutter_px
    long (shorter runs are noise, not real gutters, and are absorbed back
    into the surrounding content band)."""
    is_gutter_candidate = background_fraction >= gutter_fraction

    real_gutter = np.zeros(length, dtype=bool)
    run_start: int | None = None
    for i in range(length + 1):
        at_end = i == length
        if not at_end and is_gutter_candidate[i]:
            if run_start is None:
                run_start = i
            continue
        if run_start is not None:
            if i - run_start >= min_gutter_px:
                real_gutter[run_start:i] = True
            run_start = None

    bands: list[tuple[int, int]] = []
    run_start = None
    for i in range(length + 1):
        at_end = i == length
        if not at_end and not real_gutter[i]:
            if run_start is None:
                run_start = i
            continue
        if run_start is not None:
            bands.append((run_start, i))
            run_start = None

    return bands
