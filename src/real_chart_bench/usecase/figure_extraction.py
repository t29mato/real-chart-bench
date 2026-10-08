"""Port for extracting figure-candidate images from a PDF (design §1.2
"図表抽出", validated by the deep-digitizer pilot, design §7.10).

Two extraction strategies are both required, not either/or: embedded raster
images (most figures) AND whole-page rendering as a fallback for pages with
no embedded images above the size threshold (vector-drawn charts, e.g.
Origin/matplotlib output, are otherwise missed entirely — confirmed in the
pilot audit, SID 3995 Fig.7).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class ImageSource(Enum):
    EMBEDDED = "embedded"
    PAGE_RENDER = "page_render"


@dataclass(frozen=True)
class ExtractedImage:
    page_number: int  # 1-indexed
    source: ImageSource
    image_bytes: bytes
    width: int
    height: int


class FigureExtractionPort(Protocol):
    def extract(self, pdf_bytes: bytes) -> list[ExtractedImage]: ...


def canonical_image_name(index: int, image: ExtractedImage) -> str:
    """The file name every collection script has used for an extracted image
    (``collect_v0_dataset.py``, ``fetch_verified_images.py``): registry entries
    address raw images by this name, so a re-extraction must reproduce it.
    ``index`` is the image's position in the whole extractor output, not per page.
    Embedded images keep the historical ``.jpg`` suffix whatever their codec."""
    ext = "png" if image.source is ImageSource.PAGE_RENDER else "jpg"
    return f"p{image.page_number:02d}_{image.source.value}_{index}.{ext}"
