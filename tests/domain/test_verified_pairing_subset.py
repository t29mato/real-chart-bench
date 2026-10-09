"""VerifiedPairing.subset (design §7.88.1): defaults to core, must agree with
the entry's licence (never hand-overridden), and a committed figure must sit
in its subset's own directory."""

import pytest

from real_chart_bench.domain.dataset_subset import DatasetSubset
from real_chart_bench.domain.verified_pairing import VerificationStatus, VerifiedPairing


def _pairing(*, license_id="cc-by", subset=None, image_path="data/verified_pairs/crops/1/a.png"):
    kwargs = {} if subset is None else {"subset": subset}
    return VerifiedPairing(
        paper_id="1",
        figure_id="10",
        image_path=image_path,
        panel_label=None,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        status=VerificationStatus.VERIFIED,
        verified_at="2026-10-10",
        evidence="test",
        license_id=license_id,
        **kwargs,
    )


def test_subset_defaults_to_core():
    assert _pairing().subset is DatasetSubset.CORE


def test_nc_entry_with_nc_licence_and_nc_directory():
    p = _pairing(
        license_id="cc-by-nc-sa",
        subset=DatasetSubset.NC,
        image_path="data/verified_pairs_nc/crops/1/a.png",
    )
    assert p.subset is DatasetSubset.NC


def test_nc_licence_left_as_core_is_rejected():
    with pytest.raises(ValueError, match="subset"):
        _pairing(license_id="cc-by-nc", image_path="data/verified_pairs_nc/crops/1/a.png")


def test_core_licence_marked_nc_is_rejected():
    with pytest.raises(ValueError, match="subset"):
        _pairing(
            license_id="cc-by",
            subset=DatasetSubset.NC,
            image_path="data/verified_pairs_nc/crops/1/a.png",
        )


def test_nc_figure_committed_under_the_core_directory_is_rejected():
    with pytest.raises(ValueError, match="verified_pairs_nc"):
        _pairing(
            license_id="cc-by-nc",
            subset=DatasetSubset.NC,
            image_path="data/verified_pairs/crops/1/a.png",
        )


def test_core_figure_committed_under_the_nc_directory_is_rejected():
    with pytest.raises(ValueError, match="verified_pairs/"):
        _pairing(image_path="data/verified_pairs_nc/crops/1/a.png")


def test_bare_raw_filename_is_not_a_committed_file_and_passes():
    # a bare name points into the gitignored data/raw/images/<paper_id>/
    p = _pairing(license_id="cc-by-nc", subset=DatasetSubset.NC, image_path="p03_embedded_4.jpg")
    assert p.subset is DatasetSubset.NC


def test_entry_without_licence_loads_but_is_not_admitted():
    # legacy-shaped entry: constructible (the registry must stay loadable), but
    # with no licence on record it is admitted to no subset
    p = _pairing(license_id=None)
    assert p.subset is DatasetSubset.CORE
    assert p.is_licence_admitted is False


@pytest.mark.parametrize("license_id", ["cc-by-nd", "cc-by-nc-nd", "other-oa", "cc-by-99"])
def test_entry_whose_licence_is_not_admitted_loads_but_is_not_admitted(license_id):
    # e.g. a licence that drifted to ND: the entry stays loadable (it is
    # excluded via excluded_reason), but no subset admits it
    assert _pairing(license_id=license_id).is_licence_admitted is False


@pytest.mark.parametrize("subset", [DatasetSubset.CORE, DatasetSubset.NC])
def test_explicit_subset_does_not_admit_an_nd_licence(subset):
    root = "verified_pairs_nc" if subset is DatasetSubset.NC else "verified_pairs"
    p = _pairing(
        license_id="cc-by-nc-nd", subset=subset, image_path=f"data/{root}/crops/1/a.png"
    )
    assert p.is_licence_admitted is False


def test_admitted_core_and_nc_entries():
    assert _pairing().is_licence_admitted is True
    nc = _pairing(
        license_id="cc-by-nc",
        subset=DatasetSubset.NC,
        image_path="data/verified_pairs_nc/crops/1/a.png",
    )
    assert nc.is_licence_admitted is True
