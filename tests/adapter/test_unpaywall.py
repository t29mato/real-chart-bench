import json
import urllib.error

from real_chart_bench.adapter.unpaywall import UnpaywallOaLookupAdapter


def _payload(**overrides):
    data = {
        "doi": "10.1/x",
        "is_oa": True,
        "best_oa_location": {
            "url_for_pdf": "https://pub/x.pdf",
            "host_type": "publisher",
            "license": "cc-by",
        },
        "oa_locations": [
            {"url_for_pdf": "https://pub/x.pdf", "host_type": "publisher", "license": "cc-by"},
            {"url_for_pdf": "https://repo/x.pdf", "host_type": "repository", "license": None},
            {"url_for_pdf": None, "url": "https://landing", "host_type": "repository"},
        ],
    }
    data.update(overrides)
    return json.dumps(data).encode()


def test_query_url_carries_doi_and_contact_email():
    seen = []
    adapter = UnpaywallOaLookupAdapter(
        email="me@example.org", transport=lambda url: seen.append(url) or _payload()
    )
    adapter.lookup("10.1/x")
    assert seen == ["https://api.unpaywall.org/v2/10.1/x?email=me@example.org"]


def test_best_location_comes_first_and_all_oa_locations_follow():
    record = UnpaywallOaLookupAdapter(email="e", transport=lambda u: _payload()).lookup("10.1/x")
    assert record.is_oa is True
    assert record.best_license == "cc-by"
    assert record.pdf_urls() == ["https://pub/x.pdf", "https://repo/x.pdf"]
    assert [loc.host_type for loc in record.locations][:2] == ["publisher", "publisher"]


def test_no_best_location_means_no_licence():
    record = UnpaywallOaLookupAdapter(
        email="e", transport=lambda u: _payload(best_oa_location=None, oa_locations=[], is_oa=False)
    ).lookup("10.1/x")
    assert record.best_license is None
    assert record.is_oa is False
    assert record.pdf_urls() == []


def test_http_error_returns_none():
    def transport(url):
        raise urllib.error.HTTPError(url, 404, "Not Found", hdrs=None, fp=None)  # type: ignore[arg-type]

    assert UnpaywallOaLookupAdapter(email="e", transport=transport).lookup("10.1/x") is None


def test_connection_error_returns_none():
    def transport(url):
        raise OSError("timed out")

    assert UnpaywallOaLookupAdapter(email="e", transport=transport).lookup("10.1/x") is None


def test_malformed_json_returns_none():
    assert (
        UnpaywallOaLookupAdapter(email="e", transport=lambda u: b"<html>").lookup("10.1/x") is None
    )
