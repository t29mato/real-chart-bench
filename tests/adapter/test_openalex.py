import json

from real_chart_bench.adapter.openalex import OpenAlexLicenseLookupAdapter


def _fake_transport(response_by_url):
    def transport(url: str) -> bytes:
        for prefix, payload in response_by_url.items():
            if url.startswith(prefix):
                return json.dumps(payload).encode("utf-8")
        raise AssertionError(f"unexpected URL: {url}")

    return transport


def test_fetch_many_parses_license_and_oa_status():
    payload = {
        "results": [
            {
                "doi": "https://doi.org/10.1/a",
                "open_access": {"is_oa": True, "oa_status": "gold"},
                "primary_location": {"license": "cc-by"},
            },
            {
                "doi": "https://doi.org/10.1/b",
                "open_access": {"is_oa": False, "oa_status": "closed"},
                "primary_location": {"license": None},
            },
        ]
    }
    transport = _fake_transport({"https://api.openalex.org/works": payload})
    adapter = OpenAlexLicenseLookupAdapter(transport=transport)

    results = adapter.fetch_many(["10.1/a", "10.1/b"])

    assert results["10.1/a"].is_oa is True
    assert results["10.1/a"].license_id == "cc-by"
    assert results["10.1/b"].is_oa is False
    assert results["10.1/b"].license_id is None


def test_fetch_many_falls_back_to_non_primary_location_license():
    payload = {
        "results": [
            {
                "doi": "https://doi.org/10.1/a",
                "open_access": {"is_oa": True, "oa_status": "green"},
                "primary_location": {"license": None},
                "locations": [{"license": None}, {"license": "cc-by"}],
            },
        ]
    }
    transport = _fake_transport({"https://api.openalex.org/works": payload})
    adapter = OpenAlexLicenseLookupAdapter(transport=transport)

    results = adapter.fetch_many(["10.1/a"])

    assert results["10.1/a"].license_id == "cc-by"


def test_fetch_many_empty_input_makes_no_request():
    def transport(url: str) -> bytes:
        raise AssertionError("should not be called for empty input")

    adapter = OpenAlexLicenseLookupAdapter(transport=transport)

    assert adapter.fetch_many([]) == {}


def test_fetch_many_batches_large_doi_lists():
    seen_urls = []

    def transport(url: str) -> bytes:
        seen_urls.append(url)
        return json.dumps({"results": []}).encode("utf-8")

    adapter = OpenAlexLicenseLookupAdapter(transport=transport, batch_size=2)
    adapter.fetch_many(["10.1/a", "10.1/b", "10.1/c"])

    assert len(seen_urls) == 2  # batches of 2: [a,b], [c]


def test_fetch_many_unresolved_doi_is_absent_from_result():
    payload = {"results": []}
    transport = _fake_transport({"https://api.openalex.org/works": payload})
    adapter = OpenAlexLicenseLookupAdapter(transport=transport)

    results = adapter.fetch_many(["10.1/does-not-exist"])

    assert results == {}
