"""PyMuPDF-backed implementation of PanelSplitterPort (2026-08-16 技術調査,
design §7.10/§7.11). Decodes arbitrary raster image bytes (PNG/JPEG/etc.,
as produced by adapter/figure_extraction.py) to a numpy array, delegates
grid detection to the pure domain.panel_layout.detect_panel_grid, then
crops and re-encodes each detected region back to PNG bytes.
"""

from __future__ import annotations

import numpy as np
import pymupdf

from real_chart_bench.domain.panel_layout import detect_panel_grid
from real_chart_bench.usecase.panel_splitting import SplitPanel

_LUMINANCE_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def _to_rgb_array(image_bytes: bytes) -> np.ndarray:
    pixmap = pymupdf.Pixmap(image_bytes)
    if pixmap.alpha:
        pixmap = pymupdf.Pixmap(pixmap, 0)  # drop the alpha channel
    if pixmap.colorspace is None or pixmap.colorspace.name != "DeviceRGB":
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)

    array = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, pixmap.n
    )
    return array


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return (rgb[:, :, :3].astype(np.float32) @ _LUMINANCE_WEIGHTS).astype(np.uint8)


def _encode_png(rgb_crop: np.ndarray) -> bytes:
    height, width, _ = rgb_crop.shape
    contiguous = np.ascontiguousarray(rgb_crop)
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, width, height, contiguous.tobytes(), False)
    return pixmap.tobytes("png")


class PyMuPdfPanelSplitter:
    def __init__(self, **detect_panel_grid_kwargs) -> None:
        self._detect_kwargs = detect_panel_grid_kwargs

    def split(self, image_bytes: bytes) -> list[SplitPanel]:
        rgb = _to_rgb_array(image_bytes)
        luminance = _luminance(rgb)
        regions = detect_panel_grid(luminance, **self._detect_kwargs)

        panels = []
        for region in regions:
            x0, y0, x1, y1 = region.bbox
            crop = rgb[y0:y1, x0:x1]
            panels.append(SplitPanel(label=region.label, image_bytes=_encode_png(crop)))
        return panels
