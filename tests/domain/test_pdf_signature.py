"""is_pdf_content: validates the deep-digitizer pilot finding (docs/design
§7.10) that OA-labeled papers often return an HTML paywall/interstitial
instead of a PDF. Checking the %PDF magic bytes catches that before we waste
effort trying to parse HTML as a PDF.
"""

from real_chart_bench.domain.pdf_signature import is_pdf_content


def test_valid_pdf_header_is_recognized():
    assert is_pdf_content(b"%PDF-1.4\n...rest of file...") is True


def test_html_interstitial_is_rejected():
    assert is_pdf_content(b"<!DOCTYPE html><html><head></head></html>") is False


def test_empty_content_is_rejected():
    assert is_pdf_content(b"") is False


def test_too_short_content_is_rejected():
    assert is_pdf_content(b"%PD") is False


def test_leading_whitespace_before_pdf_header_is_still_recognized():
    # some servers prepend a BOM or stray newline before the real PDF bytes
    assert is_pdf_content(b"\n\n%PDF-1.7\n") is True
