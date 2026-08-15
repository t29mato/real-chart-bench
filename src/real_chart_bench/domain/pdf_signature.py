"""PDF content validation (design §7.10, deep-digitizer pilot finding).

Publishers frequently respond to an `is_oa=true` PDF URL with an HTML
paywall/interstitial instead of the actual PDF (observed: 12/45 in the
deep-digitizer pilot). Checking the ``%PDF`` magic bytes lets the fetch
adapter distinguish a genuine PDF from that before attempting to parse it.
Pure function — no I/O.
"""

from __future__ import annotations

_PDF_MAGIC = b"%PDF"


def is_pdf_content(data: bytes) -> bool:
    return data.lstrip()[: len(_PDF_MAGIC)] == _PDF_MAGIC
