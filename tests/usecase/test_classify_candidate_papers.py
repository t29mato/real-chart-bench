from real_chart_bench.domain.licensing import LicenseStatus
from real_chart_bench.usecase.classify_candidate_papers import classify_candidate_papers
from real_chart_bench.usecase.license_lookup import LicenseLookupResult


class _FakeLicenseLookup:
    def __init__(self, results: dict[str, LicenseLookupResult]) -> None:
        self._results = results

    def fetch_many(self, dois):
        return {doi: self._results[doi] for doi in dois if doi in self._results}


def test_classifies_each_resolved_doi():
    lookup = _FakeLicenseLookup(
        {
            "10.1/cc-by": LicenseLookupResult(doi="10.1/cc-by", is_oa=True, license_id="cc-by"),
            "10.1/closed": LicenseLookupResult(doi="10.1/closed", is_oa=False, license_id=None),
        }
    )

    result = classify_candidate_papers(["10.1/cc-by", "10.1/closed"], license_lookup=lookup)

    assert result["10.1/cc-by"] is LicenseStatus.REDISTRIBUTABLE
    assert result["10.1/closed"] is LicenseStatus.EXCLUDED


def test_unresolved_doi_is_needs_review_not_silently_dropped():
    lookup = _FakeLicenseLookup({})

    result = classify_candidate_papers(["10.1/unknown"], license_lookup=lookup)

    assert result["10.1/unknown"] is LicenseStatus.NEEDS_REVIEW


def test_empty_doi_list_returns_empty_dict():
    lookup = _FakeLicenseLookup({})

    assert classify_candidate_papers([], license_lookup=lookup) == {}
