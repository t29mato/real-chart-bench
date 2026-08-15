from real_chart_bench.usecase.pdf_fetch import PdfFetchResult, PdfFetchStatus


def test_pdf_fetch_result_is_a_plain_value_object():
    result = PdfFetchResult(status=PdfFetchStatus.OK, content=b"%PDF-1.4")
    assert result.status is PdfFetchStatus.OK
    assert result.content == b"%PDF-1.4"


def test_pdf_fetch_result_detail_defaults_to_none():
    result = PdfFetchResult(status=PdfFetchStatus.NO_URL)
    assert result.detail is None
    assert result.content is None
