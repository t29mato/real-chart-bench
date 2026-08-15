"""Tests build real PNG bytes via pymupdf (not mocks), exercising the full
decode -> detect_panel_grid -> crop -> re-encode round trip."""

import pymupdf
import pytest

from real_chart_bench.adapter.panel_layout import PyMuPdfPanelSplitter

WHITE = (255, 255, 255)
DARK = (40, 40, 40)


def _solid_rgb_png(width: int, height: int, color=WHITE) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pix.set_rect(pix.irect, color)
    return pix.tobytes("png")


def _two_by_two_grid_png() -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, 100, 100), False)
    pix.set_rect(pix.irect, WHITE)
    pix.set_rect(pymupdf.IRect(0, 0, 46, 46), DARK)
    pix.set_rect(pymupdf.IRect(54, 0, 100, 46), DARK)
    pix.set_rect(pymupdf.IRect(0, 54, 46, 100), DARK)
    pix.set_rect(pymupdf.IRect(54, 54, 100, 100), DARK)
    return pix.tobytes("png")


@pytest.fixture
def splitter():
    return PyMuPdfPanelSplitter(min_gutter_px=4)


def test_single_panel_image_returns_one_split_wrapping_original(splitter):
    image = _solid_rgb_png(80, 60, DARK)

    result = splitter.split(image)

    assert len(result) == 1
    assert result[0].label == "a"
    # decodes back to an image of the same size (not necessarily byte-identical
    # after PNG re-encoding, but dimensions must be preserved)
    pix = pymupdf.Pixmap(result[0].image_bytes)
    assert (pix.width, pix.height) == (80, 60)


def test_two_by_two_grid_produces_four_labeled_panels(splitter):
    image = _two_by_two_grid_png()

    result = splitter.split(image)

    assert [p.label for p in result] == ["a", "b", "c", "d"]
    for panel in result:
        pix = pymupdf.Pixmap(panel.image_bytes)
        assert pix.width > 0 and pix.height > 0
        # each cropped panel should be meaningfully smaller than the original
        assert pix.width < 100 and pix.height < 100
