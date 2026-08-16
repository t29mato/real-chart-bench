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


def test_max_panels_default_never_exceeds_the_label_alphabet():
    canvas = _canvas(30 * 40, 30 * 40)
    for row in range(30):
        for col in range(30):
            y0, x0 = row * 40, col * 40
            _fill(canvas, y0 + 2, y0 + 34, x0 + 2, x0 + 34)

    regions = detect_panel_grid(canvas, min_gutter_px=2)

    assert len(regions) <= 26
