"""HTTP-backed implementation of PdfFetchPort (design §1.2, §7.10).

HTTP access is isolated behind an injectable ``transport`` callable so tests
never need a live network connection; the default transport is a thin
urllib wrapper. ``urllib.error.HTTPError`` (a subclass of ``OSError``) is
checked before the broader ``OSError`` catch so 403/404-style responses are
distinguished from connection-level failures (timeout, DNS, reset) — both
were observed as distinct real failure modes in the deep-digitizer pilot.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from collections.abc import Callable

from real_chart_bench.domain.pdf_signature import is_pdf_content
from real_chart_bench.usecase.pdf_fetch import PdfFetchResult, PdfFetchStatus

_USER_AGENT = "real-chart-bench/0.0.1 (https://github.com/t29mato/real-chart-bench)"

Transport = Callable[[str], bytes]


def _default_transport(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return response.read()


class HttpPdfFetchAdapter:
    def __init__(self, *, transport: Transport | None = None) -> None:
        self._transport = transport or _default_transport

    def fetch(self, pdf_url: str | None) -> PdfFetchResult:
        if not pdf_url or not pdf_url.strip():
            return PdfFetchResult(status=PdfFetchStatus.NO_URL)

        try:
            content = self._transport(pdf_url)
        except urllib.error.HTTPError as exc:
            return PdfFetchResult(status=PdfFetchStatus.HTTP_ERROR, detail=f"HTTP {exc.code}")
        except OSError as exc:
            return PdfFetchResult(status=PdfFetchStatus.CONNECTION_ERROR, detail=str(exc))

        if not is_pdf_content(content):
            return PdfFetchResult(status=PdfFetchStatus.NOT_A_PDF)

        return PdfFetchResult(status=PdfFetchStatus.OK, content=content)
