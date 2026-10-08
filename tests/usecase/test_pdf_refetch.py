"""Phase 1 step 1 (scaling-verification.md 第1段, figure-fetch-distribution.md
§7.2/§9 Phase 2): re-fetch the licence-cleared papers that have no images, trying
every Unpaywall oa_location instead of the single pdf_url the v0 collection used."""

import hashlib

from real_chart_bench.usecase.figure_extraction import (
    ExtractedImage,
    ImageSource,
    canonical_image_name,
)
from real_chart_bench.usecase.oa_lookup import OaLocation, OaRecord
from real_chart_bench.usecase.pdf_fetch import PdfFetchResult, PdfFetchStatus
from real_chart_bench.usecase.pdf_refetch import RefetchStatus, refetch_paper

PDF = b"%PDF-1.7 fake"


def _record(*locations: OaLocation, license_id="cc-by", is_oa=True) -> OaRecord:
    return OaRecord(is_oa=is_oa, best_license=license_id, locations=tuple(locations))


def _loc(url_for_pdf, host_type="publisher", license_id="cc-by") -> OaLocation:
    return OaLocation(url_for_pdf=url_for_pdf, host_type=host_type, license=license_id)


class _ScriptedFetcher:
    def __init__(self, responses: dict[str, PdfFetchResult]):
        self.responses = responses
        self.calls: list[str] = []

    def fetch(self, pdf_url):
        self.calls.append(pdf_url)
        return self.responses[pdf_url]


class _FixedExtractor:
    def __init__(self, images):
        self.images = images

    def extract(self, pdf_bytes):
        return list(self.images)


class _BrokenExtractor:
    def extract(self, pdf_bytes):
        raise RuntimeError("cannot parse")


def _ok():
    return PdfFetchResult(status=PdfFetchStatus.OK, content=PDF)


def _http(code=403):
    return PdfFetchResult(status=PdfFetchStatus.HTTP_ERROR, detail=f"HTTP {code}")


def _img(page, source=ImageSource.EMBEDDED, data=b"img"):
    return ExtractedImage(page_number=page, source=source, image_bytes=data, width=10, height=10)


# --- canonical image naming (shared with collect_v0_dataset.py / fetch_verified_images.py)


def test_embedded_image_name_uses_zero_padded_page_and_jpg():
    assert canonical_image_name(4, _img(3)) == "p03_embedded_4.jpg"


def test_page_render_name_uses_png():
    assert canonical_image_name(0, _img(12, ImageSource.PAGE_RENDER)) == "p12_page_render_0.png"


# --- OaRecord.pdf_urls


def test_pdf_urls_keep_location_order_and_drop_duplicates_and_missing():
    record = _record(
        _loc("https://a/x.pdf"), _loc(None), _loc("https://b/y.pdf"), _loc("https://a/x.pdf")
    )
    assert record.pdf_urls() == ["https://a/x.pdf", "https://b/y.pdf"]


def test_pdf_urls_of_record_without_locations_is_empty():
    assert _record().pdf_urls() == []


# --- licence re-check


def test_licence_moved_to_nd_is_skipped_without_fetching():
    fetcher = _ScriptedFetcher({})
    outcome = refetch_paper(
        _record(_loc("https://a/x.pdf"), license_id="cc-by-nc-nd"), fetcher, _FixedExtractor([])
    )
    assert outcome.status is RefetchStatus.LICENCE_CHANGED
    assert outcome.licence_today == "cc-by-nc-nd"
    assert fetcher.calls == []


def test_missing_licence_today_is_treated_as_changed_not_as_cc_by():
    outcome = refetch_paper(
        _record(_loc("https://a/x.pdf"), license_id=None), _ScriptedFetcher({}), _FixedExtractor([])
    )
    assert outcome.status is RefetchStatus.LICENCE_CHANGED


def test_closed_access_today_is_licence_changed():
    outcome = refetch_paper(
        _record(_loc("https://a/x.pdf"), license_id=None, is_oa=False),
        _ScriptedFetcher({}),
        _FixedExtractor([]),
    )
    assert outcome.status is RefetchStatus.LICENCE_CHANGED


def test_lookup_failure_is_its_own_status_so_it_can_be_retried():
    outcome = refetch_paper(None, _ScriptedFetcher({}), _FixedExtractor([]))
    assert outcome.status is RefetchStatus.LOOKUP_FAILED


# --- routes


def test_no_pdf_url_today_is_no_url():
    outcome = refetch_paper(_record(_loc(None)), _ScriptedFetcher({}), _FixedExtractor([]))
    assert outcome.status is RefetchStatus.NO_URL
    assert outcome.attempts == ()


def test_falls_through_to_second_location_when_first_is_forbidden():
    fetcher = _ScriptedFetcher({"https://pub/x.pdf": _http(), "https://repo/x.pdf": _ok()})
    outcome = refetch_paper(
        _record(_loc("https://pub/x.pdf"), _loc("https://repo/x.pdf", host_type="repository")),
        fetcher,
        _FixedExtractor([_img(1)]),
    )
    assert outcome.status is RefetchStatus.OK
    assert outcome.route_url == "https://repo/x.pdf"
    assert outcome.route_host_type == "repository"
    assert [a.status for a in outcome.attempts] == [RefetchStatus.HTTP_ERROR, RefetchStatus.OK]
    assert outcome.attempts[0].detail == "HTTP 403"


def test_stops_at_first_successful_location():
    fetcher = _ScriptedFetcher({"https://a/x.pdf": _ok(), "https://b/x.pdf": _ok()})
    refetch_paper(
        _record(_loc("https://a/x.pdf"), _loc("https://b/x.pdf")), fetcher, _FixedExtractor([])
    )
    assert fetcher.calls == ["https://a/x.pdf"]


def test_pause_is_called_between_requests_not_before_the_first():
    pauses = []
    fetcher = _ScriptedFetcher({"https://a/x.pdf": _http(), "https://b/x.pdf": _http(404)})
    refetch_paper(
        _record(_loc("https://a/x.pdf"), _loc("https://b/x.pdf")),
        fetcher,
        _FixedExtractor([]),
        pause=lambda: pauses.append(1),
    )
    assert len(pauses) == 1


def test_all_locations_failing_reports_the_last_failure_and_every_attempt():
    fetcher = _ScriptedFetcher(
        {
            "https://a/x.pdf": _http(),
            "https://b/x.pdf": PdfFetchResult(status=PdfFetchStatus.NOT_A_PDF),
        }
    )
    outcome = refetch_paper(
        _record(_loc("https://a/x.pdf"), _loc("https://b/x.pdf")), fetcher, _FixedExtractor([])
    )
    assert outcome.status is RefetchStatus.NOT_PDF
    assert len(outcome.attempts) == 2
    assert outcome.route_url is None
    assert outcome.pdf_bytes is None


def test_connection_error_is_kept_distinct_from_http_error():
    fetcher = _ScriptedFetcher(
        {
            "https://a/x.pdf": PdfFetchResult(
                status=PdfFetchStatus.CONNECTION_ERROR, detail="timed out"
            )
        }
    )
    outcome = refetch_paper(_record(_loc("https://a/x.pdf")), fetcher, _FixedExtractor([]))
    assert outcome.status is RefetchStatus.CONNECTION_ERROR


# --- extraction and hashes


def test_ok_outcome_names_images_canonically_and_hashes_everything():
    images = [_img(2, data=b"a"), _img(5, ImageSource.PAGE_RENDER, data=b"b")]
    outcome = refetch_paper(
        _record(_loc("https://a/x.pdf")),
        _ScriptedFetcher({"https://a/x.pdf": _ok()}),
        _FixedExtractor(images),
    )
    assert outcome.pdf_sha256 == hashlib.sha256(PDF).hexdigest()
    assert [i.name for i in outcome.images] == ["p02_embedded_0.jpg", "p05_page_render_1.png"]
    assert outcome.images[0].sha256 == hashlib.sha256(b"a").hexdigest()
    assert outcome.n_images == 2


def test_pdf_with_zero_extractable_images_is_still_ok_with_zero_images():
    outcome = refetch_paper(
        _record(_loc("https://a/x.pdf")),
        _ScriptedFetcher({"https://a/x.pdf": _ok()}),
        _FixedExtractor([]),
    )
    assert outcome.status is RefetchStatus.OK
    assert outcome.n_images == 0


def test_extractor_crash_is_extract_failed_but_keeps_the_pdf_hash():
    outcome = refetch_paper(
        _record(_loc("https://a/x.pdf")),
        _ScriptedFetcher({"https://a/x.pdf": _ok()}),
        _BrokenExtractor(),
    )
    assert outcome.status is RefetchStatus.EXTRACT_FAILED
    assert outcome.pdf_sha256 == hashlib.sha256(PDF).hexdigest()
    assert "cannot parse" in outcome.detail


# --- log entry


def test_log_entry_is_json_ready_and_carries_no_bytes():
    outcome = refetch_paper(
        _record(_loc("https://a/x.pdf")),
        _ScriptedFetcher({"https://a/x.pdf": _ok()}),
        _FixedExtractor([_img(1, data=b"z")]),
    )
    entry = outcome.to_log_entry()
    assert entry["status"] == "ok"
    assert entry["route"] == {"url": "https://a/x.pdf", "host_type": "publisher"}
    assert entry["n_images"] == 1
    assert entry["images"] == [
        {"name": "p01_embedded_0.jpg", "sha256": hashlib.sha256(b"z").hexdigest()}
    ]
    assert entry["attempts"] == [{"url": "https://a/x.pdf", "status": "ok", "detail": None}]
    assert all(not isinstance(v, bytes) for v in entry.values())


def test_log_entry_of_failure_has_null_route_and_hashes():
    entry = refetch_paper(
        _record(_loc(None)), _ScriptedFetcher({}), _FixedExtractor([])
    ).to_log_entry()
    assert entry["status"] == "no_url"
    assert entry["route"] is None
    assert entry["pdf_sha256"] is None
    assert entry["n_images"] == 0
