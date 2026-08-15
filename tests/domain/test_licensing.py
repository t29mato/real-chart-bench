"""classify_license implements the pseudocode in design §1.3 exactly:
try the OpenAlex-reported license first, fall back to Crossref's only when
OpenAlex reported nothing, and never auto-allow an unrecognized-but-present
license string. Pure function — no network I/O (callers fetch the license
strings via an adapter and pass them in).

§7.2 allowlist (RESOLVED): CC-BY / CC0 / CC-BY-SA allowed; CC-BY-NC*,
CC-BY-ND* excluded; anything else unknown falls to NEEDS_REVIEW only via the
Crossref-fallback path, otherwise EXCLUDED.
"""

import pytest

from real_chart_bench.domain.licensing import LicenseStatus, classify_license


@pytest.mark.parametrize(
    "license_id",
    ["cc-by", "CC-BY", "cc-by-4.0", "cc-by-3.0", "cc0", "CC0", "public-domain", "cc-by-sa"],
)
def test_allowlisted_licenses_are_redistributable(license_id):
    assert classify_license(license_id) is LicenseStatus.REDISTRIBUTABLE


@pytest.mark.parametrize(
    "license_id",
    ["cc-by-nc", "cc-by-nc-4.0", "cc-by-nd", "cc-by-nd-4.0", "cc-by-nc-nd", "cc-by-nc-sa"],
)
def test_nc_and_nd_variants_are_excluded(license_id):
    assert classify_license(license_id) is LicenseStatus.EXCLUDED


def test_unknown_non_empty_license_is_excluded():
    # design §1.3 pseudocode: a *present* license not on the allowlist is
    # EXCLUDED outright (no Crossref fallback attempted in that branch).
    assert classify_license("publisher-specific") is LicenseStatus.EXCLUDED


def test_missing_license_with_no_crossref_fallback_needs_review():
    assert classify_license(None) is LicenseStatus.NEEDS_REVIEW
    assert classify_license("") is LicenseStatus.NEEDS_REVIEW


def test_missing_openalex_license_falls_back_to_crossref():
    assert (
        classify_license(None, crossref_license_id="cc-by") is LicenseStatus.REDISTRIBUTABLE
    )


def test_crossref_fallback_that_is_also_not_allowlisted_stays_needs_review():
    # per pseudocode: the fallback branch only ever returns REDISTRIBUTABLE
    # or NEEDS_REVIEW, never EXCLUDED (openalex reported nothing to exclude on).
    assert (
        classify_license(None, crossref_license_id="cc-by-nc") is LicenseStatus.NEEDS_REVIEW
    )


def test_whitespace_only_license_is_treated_as_missing():
    assert classify_license("   ") is LicenseStatus.NEEDS_REVIEW


def test_license_matching_is_case_and_whitespace_insensitive():
    assert classify_license("  CC-BY  ") is LicenseStatus.REDISTRIBUTABLE


def test_closed_access_with_no_license_is_excluded_not_needs_review():
    # Pilot-driven refinement (§7.9): don't send unambiguously closed papers
    # to the review queue just because OpenAlex didn't report a license.
    assert classify_license(None, is_oa=False) is LicenseStatus.EXCLUDED


def test_is_oa_false_is_checked_before_crossref_fallback():
    assert (
        classify_license(None, crossref_license_id="cc-by", is_oa=False)
        is LicenseStatus.EXCLUDED
    )


def test_allowlisted_license_wins_even_if_is_oa_is_somehow_false():
    # license on the allowlist is a stronger, more direct signal than is_oa.
    assert classify_license("cc-by", is_oa=False) is LicenseStatus.REDISTRIBUTABLE


def test_is_oa_true_does_not_bypass_the_normal_missing_license_path():
    assert classify_license(None, is_oa=True) is LicenseStatus.NEEDS_REVIEW
