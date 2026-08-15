from real_chart_bench.usecase.license_lookup import LicenseLookupResult


def test_license_lookup_result_is_a_plain_value_object():
    result = LicenseLookupResult(doi="10.1000/xyz", is_oa=True, license_id="cc-by")

    assert result.doi == "10.1000/xyz"
    assert result.is_oa is True
    assert result.license_id == "cc-by"
