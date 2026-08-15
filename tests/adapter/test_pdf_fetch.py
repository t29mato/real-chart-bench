import urllib.error

import pytest

from real_chart_bench.adapter.pdf_fetch import HttpPdfFetchAdapter
from real_chart_bench.usecase.pdf_fetch import PdfFetchStatus


def test_no_url_short_circuits_without_calling_transport():
    def transport(url: str) -> bytes:
        raise AssertionError("should not be called")

    adapter = HttpPdfFetchAdapter(transport=transport)

    assert adapter.fetch(None).status is PdfFetchStatus.NO_URL
    assert adapter.fetch("   ").status is PdfFetchStatus.NO_URL


def test_valid_pdf_bytes_are_ok():
    adapter = HttpPdfFetchAdapter(transport=lambda url: b"%PDF-1.4\n...")

    result = adapter.fetch("https://example.org/paper.pdf")

    assert result.status is PdfFetchStatus.OK
    assert result.content == b"%PDF-1.4\n..."


def test_html_interstitial_is_not_a_pdf():
    adapter = HttpPdfFetchAdapter(transport=lambda url: b"<html>paywall</html>")

    result = adapter.fetch("https://example.org/paper.pdf")

    assert result.status is PdfFetchStatus.NOT_A_PDF
    assert result.content is None


def test_http_error_is_classified():
    def transport(url: str) -> bytes:
        raise urllib.error.HTTPError(url, 403, "Forbidden", hdrs=None, fp=None)  # type: ignore[arg-type]

    adapter = HttpPdfFetchAdapter(transport=transport)

    result = adapter.fetch("https://example.org/paper.pdf")

    assert result.status is PdfFetchStatus.HTTP_ERROR
    assert "403" in (result.detail or "")


def test_connection_error_is_classified():
    def transport(url: str) -> bytes:
        raise ConnectionResetError("reset by peer")

    adapter = HttpPdfFetchAdapter(transport=transport)

    result = adapter.fetch("https://example.org/paper.pdf")

    assert result.status is PdfFetchStatus.CONNECTION_ERROR


def test_generic_url_error_is_classified_as_connection_error():
    def transport(url: str) -> bytes:
        raise urllib.error.URLError("name resolution failed")

    adapter = HttpPdfFetchAdapter(transport=transport)

    result = adapter.fetch("https://example.org/paper.pdf")

    assert result.status is PdfFetchStatus.CONNECTION_ERROR


@pytest.mark.parametrize("url", [None, ""])
def test_missing_url_variants(url):
    adapter = HttpPdfFetchAdapter(transport=lambda u: b"%PDF")
    assert adapter.fetch(url).status is PdfFetchStatus.NO_URL
