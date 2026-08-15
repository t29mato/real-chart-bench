"""Port for fetching a paper's PDF (design §1.2 "論文PDF/HTML取得").

Status taxonomy mirrors the deep-digitizer pilot's empirical failure modes
(docs/design §7.10): OA-labeled papers frequently have no direct PDF URL, or
the URL resolves to an HTML paywall/interstitial rather than a real PDF, or
the publisher's server rejects/(times out on) bot traffic outright.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class PdfFetchStatus(Enum):
    OK = "ok"
    NO_URL = "no_pdf_url"
    NOT_A_PDF = "not_a_pdf"  # e.g. paywall/HTML interstitial
    HTTP_ERROR = "http_error"  # e.g. 403/404
    CONNECTION_ERROR = "connection_error"


@dataclass(frozen=True)
class PdfFetchResult:
    status: PdfFetchStatus
    content: bytes | None = None
    detail: str | None = None


class PdfFetchPort(Protocol):
    def fetch(self, pdf_url: str | None) -> PdfFetchResult: ...
