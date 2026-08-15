"""PyMuPDF-backed implementation of FigureExtractionPort (design §1.2,
validated in the deep-digitizer pilot — design §7.10).

Defaults (min_embedded_pixels=150*150, render_dpi=150) match the pilot's
empirically chosen values: high enough to exclude icons/logos/journal
mastheads, low enough not to lose real embedded figure images.
"""

from __future__ import annotations

import pymupdf

from real_chart_bench.usecase.figure_extraction import ExtractedImage, ImageSource

_DEFAULT_MIN_EMBEDDED_PIXELS = 150 * 150
_DEFAULT_RENDER_DPI = 150


class PyMuPdfFigureExtractor:
    def __init__(
        self,
        *,
        min_embedded_pixels: int = _DEFAULT_MIN_EMBEDDED_PIXELS,
        render_dpi: int = _DEFAULT_RENDER_DPI,
    ) -> None:
        self._min_embedded_pixels = min_embedded_pixels
        self._render_dpi = render_dpi

    def extract(self, pdf_bytes: bytes) -> list[ExtractedImage]:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            images: list[ExtractedImage] = []
            for page_index in range(doc.page_count):
                images.extend(self._extract_page(doc, page_index))
            return images
        finally:
            doc.close()

    def _extract_page(self, doc: pymupdf.Document, page_index: int) -> list[ExtractedImage]:
        page = doc[page_index]
        page_number = page_index + 1
        embedded: list[ExtractedImage] = []

        for image_info in page.get_images(full=True):
            xref = image_info[0]
            try:
                base = doc.extract_image(xref)
            except Exception:  # noqa: BLE001 - a single malformed xref shouldn't abort the page
                continue
            width, height = base.get("width", 0), base.get("height", 0)
            if width * height < self._min_embedded_pixels:
                continue
            embedded.append(
                ExtractedImage(
                    page_number=page_number,
                    source=ImageSource.EMBEDDED,
                    image_bytes=base["image"],
                    width=width,
                    height=height,
                )
            )

        if embedded:
            return embedded

        # No (large enough) embedded images on this page: fall back to
        # rendering the whole page, so vector-drawn charts aren't missed.
        pixmap = page.get_pixmap(dpi=self._render_dpi)
        return [
            ExtractedImage(
                page_number=page_number,
                source=ImageSource.PAGE_RENDER,
                image_bytes=pixmap.tobytes("png"),
                width=pixmap.width,
                height=pixmap.height,
            )
        ]
