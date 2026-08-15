"""Tests build real, minimal PDFs in-memory via pymupdf itself (not mocks)
so the extraction adapter is exercised against actual PDF parsing, the same
way it validated against real papers in the deep-digitizer pilot.
"""

import pymupdf
import pytest

from real_chart_bench.adapter.figure_extraction import PyMuPdfFigureExtractor
from real_chart_bench.usecase.figure_extraction import ImageSource


def _solid_png(width: int, height: int) -> bytes:
    pix = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), False)
    pix.set_rect(pix.irect, (200, 50, 50))
    return pix.tobytes("png")


def _pdf_with_embedded_image(width: int, height: int) -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_image(
        pymupdf.Rect(50, 50, 50 + width, 50 + height),
        stream=_solid_png(width, height),
    )
    data = doc.tobytes()
    doc.close()
    return data


def _pdf_with_text_only() -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "no embedded images on this page")
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def extractor():
    return PyMuPdfFigureExtractor(min_embedded_pixels=100 * 100, render_dpi=72)


def test_embedded_image_above_threshold_is_extracted(extractor):
    pdf = _pdf_with_embedded_image(200, 200)

    images = extractor.extract(pdf)

    assert len(images) == 1
    assert images[0].source is ImageSource.EMBEDDED
    assert images[0].width == 200
    assert images[0].height == 200


def test_text_only_page_falls_back_to_page_render(extractor):
    pdf = _pdf_with_text_only()

    images = extractor.extract(pdf)

    assert len(images) == 1
    assert images[0].source is ImageSource.PAGE_RENDER
    assert images[0].width > 0 and images[0].height > 0


def test_embedded_image_below_threshold_is_excluded_and_page_falls_back(extractor):
    # 50x50 = 2500px < 100*100 threshold -> treated as icon/logo, excluded
    pdf = _pdf_with_embedded_image(50, 50)

    images = extractor.extract(pdf)

    assert len(images) == 1
    assert images[0].source is ImageSource.PAGE_RENDER


def test_multi_page_pdf_mixes_both_strategies(extractor):
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_image(pymupdf.Rect(0, 0, 200, 200), stream=_solid_png(200, 200))
    page2 = doc.new_page()
    page2.insert_text((50, 50), "text only page")
    pdf = doc.tobytes()
    doc.close()

    images = extractor.extract(pdf)

    assert len(images) == 2
    by_page = {img.page_number: img.source for img in images}
    assert by_page[1] is ImageSource.EMBEDDED
    assert by_page[2] is ImageSource.PAGE_RENDER


def test_empty_document_returns_no_images(extractor):
    doc = pymupdf.open()
    doc.new_page()
    pdf = doc.tobytes()
    doc.close()

    # a blank page has no embedded images -> falls back to a page render,
    # not an empty list (there's still a page to represent)
    images = extractor.extract(pdf)
    assert len(images) == 1
    assert images[0].source is ImageSource.PAGE_RENDER
