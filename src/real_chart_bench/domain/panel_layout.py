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
# len(_LABELS): a real scientific figure essentially never has this many
# panels, so exceeding it is itself a signal that gutter detection is
# picking up noise (JPEG artifacts etc.) rather than a real grid — see
# design §7.14, found on ~39% of real extracted images before this guard.
_DEFAULT_MAX_PANELS = len(_LABELS)
# §7.14 visual audit: axis-tick strips and rotated axis-label text form
# their own thin whitespace-separated band and were being sliced off as
# bogus extra panels. Bands narrower than this fraction of the image
# dimension are dropped rather than treated as real panel rows/columns.
_DEFAULT_MIN_PANEL_SIZE_FRACTION = 0.05
# §7.21 finding (paper 4176): splitting a *full academic page render*
# (figure + caption + body text, not a pre-cropped figure) produced a bogus
# extra "panel" that was actually a paragraph of running text. A paragraph
# has substantial non-background content -- like a real chart -- so
# min_content_fraction alone doesn't reject it. Its texture does differ: a
# cell with this many separate horizontal content "runs" (individual text
# lines) rather than one large, roughly continuous block is treated as
# prose, not a panel, and dropped. A real chart panel essentially never has
# this many evenly-repeating horizontal bands.
_DEFAULT_MIN_TEXT_LINE_RUNS = 4
# Run *count* alone isn't safe (see _looks_like_text docstring): a busy
# multi-series chart can also produce several short runs. Text lines are
# additionally nearly uniform in height; real chart content runs vary far
# more. 0.3 sits well above real text's observed CV (~0.12-0.15) and well
# below a busy real chart panel's observed CV (~0.6-0.9, see the paper
# 17037 regression this threshold was tuned against).
_DEFAULT_MAX_RUN_HEIGHT_CV = 0.3


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
    max_panels: int = _DEFAULT_MAX_PANELS,
    min_panel_size_fraction: float = _DEFAULT_MIN_PANEL_SIZE_FRACTION,
    min_text_line_runs: int = _DEFAULT_MIN_TEXT_LINE_RUNS,
    max_run_height_cv: float = _DEFAULT_MAX_RUN_HEIGHT_CV,
) -> tuple[PanelRegion, ...]:
    if luminance.ndim != 2:
        raise ValueError(f"luminance must be a 2D array, got shape {luminance.shape}")

    height, width = luminance.shape
    is_background = luminance >= background_threshold

    row_bands = _content_bands(is_background.mean(axis=1), gutter_fraction, min_gutter_px, height)
    col_bands = _content_bands(is_background.mean(axis=0), gutter_fraction, min_gutter_px, width)

    row_bands = _drop_thin_bands(row_bands, height, min_panel_size_fraction)
    col_bands = _drop_thin_bands(col_bands, width, min_panel_size_fraction)

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
            if _looks_like_text(cell, gutter_fraction, min_text_line_runs, max_run_height_cv):
                continue
            regions.append((row_index, col_index, (x0, y0, x1, y1)))

    if not regions or len(regions) > max_panels:
        # Either every cell was filtered out as blank, or we detected an
        # implausible number of "panels" (noise, not a real grid): both are
        # "not confident", so fall back rather than guess.
        return (PanelRegion(label="a", row=0, col=0, bbox=(0, 0, width, height)),)

    return tuple(
        PanelRegion(label=_LABELS[i], row=row_index, col=col_index, bbox=bbox)
        for i, (row_index, col_index, bbox) in enumerate(regions)
    )


def _looks_like_text(
    cell: np.ndarray,
    gutter_fraction: float,
    min_text_line_runs: int,
    max_run_height_cv: float,
) -> bool:
    """A body-text paragraph's per-row background-fraction profile alternates
    between many "content" runs (individual text lines) and "gap" runs
    (inter-line spacing) — but run *count* alone isn't a safe discriminator:
    a real chart panel with several overlapping curves/series can also
    produce many short runs (confirmed regression against paper 17037's
    figure 20736 panel (a), a 6-series line chart, during this fix's
    development). What's actually distinctive about text is that every run
    is nearly the same height (one line of text looks like the next) —
    real chart content runs vary wildly in height instead (a single
    continuous curve/axes block here, a thin stray mark there). Requiring
    *both* enough runs *and* low height variation (coefficient of
    variation, std/mean) is what correctly separates prose from charts;
    see tests/domain/test_panel_layout.py for the concrete numbers from
    both a real text block and the chart panel that motivated this."""
    # Only called (from detect_panel_grid) on a cell that already passed the
    # min_content_fraction check, so cell is always non-empty and has at
    # least one content row here -- no defensive empty/zero-mean guard
    # needed (an untested one would just be dead code).
    row_background_fraction = cell.mean(axis=1)
    is_content_row = row_background_fraction < gutter_fraction

    run_heights: list[int] = []
    run_start: int | None = None
    for i, content in enumerate(is_content_row):
        if content and run_start is None:
            run_start = i
        elif not content and run_start is not None:
            run_heights.append(i - run_start)
            run_start = None
    if run_start is not None:
        run_heights.append(len(is_content_row) - run_start)

    if len(run_heights) < min_text_line_runs:
        return False

    heights = np.array(run_heights, dtype=np.float64)
    coefficient_of_variation = float(heights.std()) / float(heights.mean())
    return coefficient_of_variation <= max_run_height_cv


def _drop_thin_bands(
    bands: list[tuple[int, int]], total_length: int, min_fraction: float
) -> list[tuple[int, int]]:
    min_px = min_fraction * total_length
    return [(s, e) for s, e in bands if (e - s) >= min_px]


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
