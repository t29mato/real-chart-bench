"""detect_panel_grid: splits a composite multi-panel figure into sub-panel
regions by finding near-uniform "background" bands (whitespace gutters)
between panels — the technical investigation requested by 司令塔 (2026-08-16)
to resolve the "figure_name refers to a single panel, but only the whole
composite figure can be extracted from a PDF" gap (design §7.10/§7.11,
deep-digitizer's shared open problem).

Pure numpy array logic — no image codec, no I/O (design layering: decoding
raw image bytes into an array is an adapter concern, see
adapter/panel_layout.py). Intentionally generic (row/col/bbox/label only)
so it can be reused as-is by other consumers (e.g. deep-digitizer) without
depending on real-chart-bench's own domain types.
"""

import numpy as np
import pytest

from real_chart_bench.domain.panel_layout import PanelRegion, detect_panel_grid

WHITE = 255
DARK = 40


def _canvas(height: int, width: int) -> np.ndarray:
    return np.full((height, width), WHITE, dtype=np.uint8)


def _fill(canvas: np.ndarray, y0: int, y1: int, x0: int, x1: int, value: int = DARK) -> None:
    canvas[y0:y1, x0:x1] = value


def test_single_panel_with_no_gutters_returns_one_region_covering_whole_image():
    canvas = _canvas(100, 100)
    _fill(canvas, 5, 95, 5, 95)  # content fills almost the whole image

    regions = detect_panel_grid(canvas)

    assert regions == (PanelRegion(label="a", row=0, col=0, bbox=(0, 0, 100, 100)),)


def test_two_by_two_grid_is_detected_in_row_major_order():
    canvas = _canvas(100, 100)
    # four panels with a >=6px white gutter between them, at row 48-54 / col 48-54
    _fill(canvas, 0, 46, 0, 46)  # top-left
    _fill(canvas, 0, 46, 54, 100)  # top-right
    _fill(canvas, 54, 100, 0, 46)  # bottom-left
    _fill(canvas, 54, 100, 54, 100)  # bottom-right

    regions = detect_panel_grid(canvas, min_gutter_px=4)

    labels = [r.label for r in regions]
    assert labels == ["a", "b", "c", "d"]
    # row-major: a=top-left, b=top-right, c=bottom-left, d=bottom-right
    a = next(r for r in regions if r.label == "a")
    d = next(r for r in regions if r.label == "d")
    assert a.row == 0 and a.col == 0
    assert d.row == 1 and d.col == 1


def test_one_by_three_horizontal_strip_grid():
    canvas = _canvas(60, 180)
    _fill(canvas, 5, 55, 0, 50)
    _fill(canvas, 5, 55, 65, 115)
    _fill(canvas, 5, 55, 130, 180)

    regions = detect_panel_grid(canvas, min_gutter_px=4)

    assert [r.label for r in regions] == ["a", "b", "c"]
    assert all(r.row == 0 for r in regions)
    assert [r.col for r in regions] == [0, 1, 2]


def test_mostly_blank_cell_is_excluded_as_not_a_real_panel():
    # 2x2 grid but bottom-right cell is left essentially blank (e.g. caption-only)
    canvas = _canvas(100, 100)
    _fill(canvas, 0, 46, 0, 46)
    _fill(canvas, 0, 46, 54, 100)
    _fill(canvas, 54, 100, 0, 46)
    # bottom-right stays background (blank)

    regions = detect_panel_grid(canvas, min_gutter_px=4)

    assert [r.label for r in regions] == ["a", "b", "c"]


def test_narrow_gap_below_min_gutter_px_is_not_treated_as_a_real_gutter():
    canvas = _canvas(100, 100)
    _fill(canvas, 0, 100, 0, 49)
    # only a 2px gap (49-51), narrower than min_gutter_px=4 -> not a real gutter
    _fill(canvas, 0, 100, 51, 100)

    regions = detect_panel_grid(canvas, min_gutter_px=4)

    assert len(regions) == 1


def test_bbox_coordinates_are_within_image_bounds():
    canvas = _canvas(50, 80)
    _fill(canvas, 2, 48, 2, 78)

    regions = detect_panel_grid(canvas)

    (region,) = regions
    x0, y0, x1, y1 = region.bbox
    assert 0 <= x0 < x1 <= 80
    assert 0 <= y0 < y1 <= 50


def test_rejects_non_2d_array():
    with pytest.raises(ValueError, match="2D"):
        detect_panel_grid(np.zeros((10, 10, 3), dtype=np.uint8))


def test_excessive_fragmentation_falls_back_to_whole_image_instead_of_crashing():
    # §7.14 finding: noisy real-world images (JPEG artifacts etc.) can trip
    # gutter detection into producing far more "panels" than any real
    # scientific figure has (up to 26+, exceeding the a-z label alphabet and
    # crashing). Treat "way too many" the same as "not confidently a grid",
    # even when every individual cell is a real, well-formed panel (a
    # deterministic 3x5=15-cell grid here, forced over an artificially low
    # max_panels=10 — more robust than relying on random noise to happen to
    # produce >max_panels surviving cells after the sliver-band filter).
    canvas = _canvas(70, 120)
    for row in range(3):
        for col in range(5):
            y0, x0 = row * 25, col * 25
            _fill(canvas, y0, y0 + 20, x0, x0 + 20)

    regions = detect_panel_grid(canvas, max_panels=10)

    assert regions == (PanelRegion(label="a", row=0, col=0, bbox=(0, 0, 120, 70)),)


def test_thin_sliver_band_does_not_produce_spurious_extra_panels():
    # §7.14 visual audit finding: real charts' axis-tick strips / rotated
    # axis-label text sit in their own thin whitespace-separated band and
    # were getting sliced off as bogus extra "panels". A real 2-panel
    # side-by-side figure plus a thin 3px sliver band below (isolated by a
    # real 4px gutter on both sides) should still yield exactly 2 panels.
    canvas = _canvas(101, 100)  # rows 90-93 and 97-100 stay background (gutters)
    _fill(canvas, 0, 90, 0, 46)
    _fill(canvas, 0, 90, 54, 100)
    _fill(canvas, 94, 97, 0, 46)
    _fill(canvas, 94, 97, 54, 100)

    regions = detect_panel_grid(canvas, min_gutter_px=4, min_panel_size_fraction=0.05)

    assert [r.label for r in regions] == ["a", "b"]


def test_min_panel_size_fraction_of_zero_disables_the_sliver_filter():
    canvas = _canvas(101, 100)
    _fill(canvas, 0, 90, 0, 46)
    _fill(canvas, 0, 90, 54, 100)
    _fill(canvas, 94, 97, 0, 46)
    _fill(canvas, 94, 97, 54, 100)

    regions = detect_panel_grid(canvas, min_gutter_px=4, min_panel_size_fraction=0.0)

    assert len(regions) == 4


def test_text_paragraph_band_is_excluded_even_though_it_has_substantial_content():
    # §7.21 finding (paper 4176): running PyMuPdfPanelSplitter on a *full
    # academic page render* (figure + caption + body text, not a pre-cropped
    # figure) produced a bogus extra "panel" that was actually a paragraph
    # of running text. A paragraph has plenty of non-background content
    # (like a real chart), so the existing content_fraction filter alone
    # doesn't reject it -- but its texture is distinctive: many thin,
    # evenly-spaced horizontal "lines" (individual text lines) rather than
    # one large, roughly continuous content block (an axes box + curves).
    canvas = _canvas(200, 100)
    _fill(canvas, 0, 70, 0, 100)  # panel a: real chart-like content
    # gutter 70-78 (8px, >= min_gutter_px)
    _fill(canvas, 78, 148, 0, 100)  # panel b: real chart-like content
    # gutter 148-156 (8px)
    # rows 156-192: 6 "text lines" (4px dark + 2px background gap each).
    # The 2px gaps are below min_gutter_px=4, so they don't split into
    # separate row-bands -- the whole paragraph merges into one row-band,
    # exactly reproducing the real full-page-render failure mode.
    for i in range(6):
        y0 = 156 + i * 6
        _fill(canvas, y0, y0 + 4, 0, 100)

    regions = detect_panel_grid(canvas, min_gutter_px=4)

    assert [r.label for r in regions] == ["a", "b"]


def test_min_text_line_runs_can_be_raised_to_stop_rejecting_a_real_panel():
    # A real chart with several separated content bands within one panel
    # (e.g. a legend box floating above the plotted data) could in principle
    # brush up against the text-detection heuristic. Raising
    # min_text_line_runs is the escape hatch -- mirrors how
    # min_panel_size_fraction=0.0 disables the sliver-band filter.
    canvas = _canvas(200, 100)
    _fill(canvas, 0, 70, 0, 100)
    _fill(canvas, 78, 148, 0, 100)
    for i in range(6):
        y0 = 156 + i * 6
        _fill(canvas, y0, y0 + 4, 0, 100)

    regions = detect_panel_grid(canvas, min_gutter_px=4, min_text_line_runs=100)

    assert [r.label for r in regions] == ["a", "b", "c"]


def test_busy_multi_series_chart_panel_is_not_misclassified_as_text():
    # Regression guard: an earlier version of the text-line heuristic used
    # run *count* alone, which false-positived on a real 6-series line
    # chart (paper 17037 figure 20736 panel a) and silently dropped a
    # legitimate, previously-VERIFIED panel. That panel's real per-row
    # content-run heights were [3, 71, 87, 121, 10] (highly irregular) vs a
    # real text block's [15, 20, 20, 16] (nearly uniform) -- reproduce that
    # irregular-run-height shape here directly, using is_background rows
    # that mimic several overlapping curves rather than evenly spaced
    # text lines.
    canvas = _canvas(200, 100)
    _fill(canvas, 0, 46, 0, 100)  # neighbor panel (kept simple, unrelated)
    # gutter 46-54 (real gutter, >= min_gutter_px)
    # one panel with internal content runs of wildly different heights
    # (3, 71, 87, 121, 10), separated by 2px gaps -- below min_gutter_px=4,
    # so they merge into a single row-band (exactly how the real panel
    # showed up), while still producing distinct is_content_row runs for
    # the per-cell text heuristic to (correctly) not flag as text.
    y = 54
    for height in (3, 71, 87, 121, 10):
        _fill(canvas, y, y + height, 0, 100)
        y += height + 2

    regions = detect_panel_grid(canvas, min_gutter_px=4)

    assert [r.label for r in regions] == ["a", "b"]


def test_max_panels_default_never_exceeds_the_label_alphabet():
    canvas = _canvas(30 * 40, 30 * 40)
    for row in range(30):
        for col in range(30):
            y0, x0 = row * 40, col * 40
            _fill(canvas, y0 + 2, y0 + 34, x0 + 2, x0 + 34)

    regions = detect_panel_grid(canvas, min_gutter_px=2)

    assert len(regions) <= 26
