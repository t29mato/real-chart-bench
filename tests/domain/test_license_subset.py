"""Licence -> dataset subset mapping (design §7.88 / §7.88.1).

CC BY / CC BY-SA / CC0 / public domain -> ``core``; CC BY-NC / CC BY-NC-SA ->
``nc``; anything with ND -> excluded (we redistribute crops, which are
derivatives); empty or unrecognised -> the manual review queue. The mapping is
decided from the licence string alone and is insensitive to how the string is
spelled (case, spaces, underscores, version suffix, creativecommons.org URL).
"""

import pytest

from real_chart_bench.domain.dataset_subset import DatasetSubset
from real_chart_bench.domain.licensing import (
    DriftKind,
    LicenseStatus,
    classify_figure_license,
    licence_admitted_for,
    license_drift,
    license_subset,
    normalize_license_id,
)

# --- normalisation -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("cc-by", "cc-by"),
        ("CC-BY", "cc-by"),
        ("  CC BY 4.0  ", "cc-by-4.0"),
        ("CC-BY-NC-SA 4.0", "cc-by-nc-sa-4.0"),
        ("cc_by_nc", "cc-by-nc"),
        ("CC BY-NC-ND 4.0", "cc-by-nc-nd-4.0"),
        ("cc  by--nc", "cc-by-nc"),
        ("https://creativecommons.org/licenses/by-nc/4.0/", "cc-by-nc-4.0"),
        ("http://creativecommons.org/licenses/by/3.0", "cc-by-3.0"),
        ("https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode", "cc-by-nc-sa-4.0"),
        ("https://creativecommons.org/publicdomain/zero/1.0/", "cc0"),
        ("Public Domain", "public-domain"),
        (None, ""),
        ("   ", ""),
    ],
)
def test_normalize_license_id(raw, expected):
    assert normalize_license_id(raw) == expected


# --- licence string -> subset --------------------------------------------------


@pytest.mark.parametrize(
    "license_id",
    [
        "cc-by",
        "CC-BY",
        "cc-by-4.0",
        "CC BY 4.0",
        "cc-by-2.0",
        "cc-by-sa",
        "CC-BY-SA 4.0",
        "cc0",
        "CC0",
        "public-domain",
        "https://creativecommons.org/licenses/by/4.0/",
    ],
)
def test_core_licences(license_id):
    decision = license_subset(license_id)
    assert decision.status is LicenseStatus.REDISTRIBUTABLE
    assert decision.subset is DatasetSubset.CORE


@pytest.mark.parametrize(
    "license_id",
    [
        "cc-by-nc",
        "cc-by-nc-4.0",
        "CC-BY-NC",
        "CC BY-NC 4.0",
        "cc_by_nc",
        "cc-by-nc-sa",
        "CC-BY-NC-SA 4.0",
        "cc by nc sa 3.0",
        "https://creativecommons.org/licenses/by-nc-sa/4.0/",
    ],
)
def test_nc_licences(license_id):
    decision = license_subset(license_id)
    assert decision.status is LicenseStatus.REDISTRIBUTABLE
    assert decision.subset is DatasetSubset.NC


@pytest.mark.parametrize(
    "license_id",
    [
        "cc-by-nd",
        "cc-by-nd-4.0",
        "CC BY-ND 4.0",
        "cc-by-nc-nd",
        "CC-BY-NC-ND 4.0",
        "cc_by_nc_nd",
        # element order is not significant
        "cc-by-nd-nc",
        "https://creativecommons.org/licenses/by-nc-nd/4.0/",
    ],
)
def test_nd_licences_are_excluded_from_both_subsets(license_id):
    decision = license_subset(license_id)
    assert decision.status is LicenseStatus.EXCLUDED
    assert decision.subset is None


@pytest.mark.parametrize(
    "license_id",
    [
        None,
        "",
        "   ",
        "publisher-specific-oa",
        "other-oa",
        "implied-oa",
        "cc-by-foo",  # an unrecognised element: do not guess
        "ccby",
        "elsevier-specific: oa user license",
    ],
)
def test_empty_or_unknown_goes_to_manual_review(license_id):
    decision = license_subset(license_id)
    assert decision.status is LicenseStatus.NEEDS_REVIEW
    assert decision.subset is None


def test_decision_keeps_the_normalised_string_for_the_record():
    assert license_subset("CC-BY-NC-SA 4.0").normalized == "cc-by-nc-sa-4.0"


# --- the §1.3 gate with subsets --------------------------------------------------


def test_gate_tags_nc_instead_of_dropping_it():
    decision = classify_figure_license("cc-by-nc", is_oa=True)
    assert decision.status is LicenseStatus.REDISTRIBUTABLE
    assert decision.subset is DatasetSubset.NC


def test_gate_still_drops_nd():
    assert classify_figure_license("cc-by-nc-nd", is_oa=True).status is LicenseStatus.EXCLUDED


def test_gate_core_is_unchanged():
    decision = classify_figure_license("cc-by", is_oa=True)
    assert decision.status is LicenseStatus.REDISTRIBUTABLE
    assert decision.subset is DatasetSubset.CORE


def test_gate_closed_access_without_licence_is_excluded():
    assert classify_figure_license(None, is_oa=False).status is LicenseStatus.EXCLUDED


def test_gate_closed_access_with_unknown_licence_is_excluded():
    assert classify_figure_license("other-oa", is_oa=False).status is LicenseStatus.EXCLUDED


def test_gate_licence_wins_over_is_oa_false_as_before():
    decision = classify_figure_license("cc-by-nc", is_oa=False)
    assert decision.subset is DatasetSubset.NC


def test_gate_unknown_present_licence_goes_to_review():
    assert classify_figure_license("other-oa").status is LicenseStatus.NEEDS_REVIEW


def test_gate_crossref_fallback_can_tag_nc():
    decision = classify_figure_license(None, crossref_license_id="cc-by-nc-sa")
    assert decision.status is LicenseStatus.REDISTRIBUTABLE
    assert decision.subset is DatasetSubset.NC


def test_gate_crossref_fallback_never_excludes():
    # §1.3: OpenAlex reported nothing to exclude on, so a non-allowed Crossref
    # value sends the paper to review rather than out.
    decision = classify_figure_license(None, crossref_license_id="cc-by-nd")
    assert decision.status is LicenseStatus.NEEDS_REVIEW


def test_gate_crossref_is_only_consulted_when_openalex_is_empty():
    decision = classify_figure_license("cc-by-nd", crossref_license_id="cc-by")
    assert decision.status is LicenseStatus.EXCLUDED


def test_gate_missing_everything_needs_review():
    decision = classify_figure_license(None)
    assert decision.status is LicenseStatus.NEEDS_REVIEW
    assert decision.subset is None


# --- licence drift (recheck_figure_licenses.py) ------------------------------------


@pytest.mark.parametrize(
    ("recorded", "current", "kind"),
    [
        ("cc-by", "cc-by", DriftKind.UNCHANGED),
        ("cc-by", "CC BY 4.0", DriftKind.UNCHANGED),
        ("cc-by-nc", "cc-by-nc-sa", DriftKind.UNCHANGED),  # still nc
        ("cc-by", "cc-by-sa", DriftKind.UNCHANGED),  # still core
        ("cc-by", "cc-by-nc", DriftKind.SUBSET_MOVE),
        ("cc-by-nc", "cc-by", DriftKind.SUBSET_MOVE),
        ("cc-by-nc", "cc-by-nc-nd", DriftKind.EXCLUDE),
        ("cc-by", "cc-by-nc-nd", DriftKind.EXCLUDE),
        ("cc-by", "cc-by-nd", DriftKind.EXCLUDE),
        ("cc-by", "closed", DriftKind.EXCLUDE),
        ("cc-by", "unknown", DriftKind.REVIEW),
        ("cc-by-nc", "other-oa", DriftKind.REVIEW),
        ("cc-by", None, DriftKind.REVIEW),
        ("cc-by", "error:TimeoutError", DriftKind.LOOKUP_ERROR),
        ("cc-by", "?", DriftKind.LOOKUP_ERROR),
    ],
)
def test_license_drift(recorded, current, kind):
    assert license_drift(recorded, current).kind is kind


def test_drift_reports_both_subsets_on_a_move():
    drift = license_drift("cc-by", "cc-by-nc")
    assert drift.recorded_subset is DatasetSubset.CORE
    assert drift.current_subset is DatasetSubset.NC


def test_drift_of_an_unrecognised_recorded_licence_is_review():
    # nothing to compare against -- a person must look at it
    assert license_drift("publisher-specific", "cc-by").kind is DriftKind.REVIEW


# --- review follow-ups: more spellings, and versions that do not exist ---------------


@pytest.mark.parametrize(
    ("license_id", "status", "subset"),
    [
        # ND as a URL without a version is still ND
        ("https://creativecommons.org/licenses/by-nd/", LicenseStatus.EXCLUDED, None),
        ("http://creativecommons.org/licenses/by-nc-nd", LicenseStatus.EXCLUDED, None),
        # the IGO ports of 3.0
        ("cc-by-3.0-igo", LicenseStatus.REDISTRIBUTABLE, DatasetSubset.CORE),
        ("CC BY-NC-SA 3.0 IGO", LicenseStatus.REDISTRIBUTABLE, DatasetSubset.NC),
        ("https://creativecommons.org/licenses/by-nc/3.0/igo/", LicenseStatus.REDISTRIBUTABLE,
         DatasetSubset.NC),
        # slash form and typographic dashes
        ("CC BY/NC", LicenseStatus.REDISTRIBUTABLE, DatasetSubset.NC),
        ("cc-by/nc-sa/4.0", LicenseStatus.REDISTRIBUTABLE, DatasetSubset.NC),
        ("CC BY—NC 4.0", LicenseStatus.REDISTRIBUTABLE, DatasetSubset.NC),  # em dash
        ("CC BY–NC–ND", LicenseStatus.EXCLUDED, None),  # en dash
        # element order is not significant
        ("cc-by-sa-nc", LicenseStatus.REDISTRIBUTABLE, DatasetSubset.NC),
        # versions CC never published -> do not guess
        ("cc-by-99", LicenseStatus.NEEDS_REVIEW, None),
        ("cc-by-0", LicenseStatus.NEEDS_REVIEW, None),
        ("cc-by-nc-5.0", LicenseStatus.NEEDS_REVIEW, None),
        ("cc-by-4.0.1", LicenseStatus.NEEDS_REVIEW, None),
        ("cc0-2.0", LicenseStatus.NEEDS_REVIEW, None),
        # versions CC did publish
        ("cc-by-1.0", LicenseStatus.REDISTRIBUTABLE, DatasetSubset.CORE),
        ("cc-by-2.5", LicenseStatus.REDISTRIBUTABLE, DatasetSubset.CORE),
        ("cc0-1.0", LicenseStatus.REDISTRIBUTABLE, DatasetSubset.CORE),
    ],
)
def test_more_spellings_and_versions(license_id, status, subset):
    decision = license_subset(license_id)
    assert decision.status is status
    assert decision.subset is subset


@pytest.mark.parametrize(
    ("license_id", "subset", "admitted"),
    [
        ("cc-by", DatasetSubset.CORE, True),
        ("cc-by-nc", DatasetSubset.NC, True),
        ("cc-by-nd", DatasetSubset.CORE, False),
        ("cc-by-nc-nd", DatasetSubset.NC, False),
        ("other-oa", DatasetSubset.CORE, False),
        (None, DatasetSubset.CORE, False),
    ],
)
def test_licence_admitted_for_subset(license_id, subset, admitted):
    assert licence_admitted_for(license_id, subset) is admitted


@pytest.mark.parametrize(
    ("license_id", "subset"),
    [("cc-by", DatasetSubset.NC), ("cc-by-nc-sa", DatasetSubset.CORE)],
)
def test_licence_admitted_for_a_contradicting_subset_raises(license_id, subset):
    with pytest.raises(ValueError, match="subset"):
        licence_admitted_for(license_id, subset)
