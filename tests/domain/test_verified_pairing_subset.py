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


def test_entry_without_licence_keeps_whatever_subset_it_was_given():
    # legacy entries have no license_id; nothing to check against
    assert _pairing(license_id=None).subset is DatasetSubset.CORE


def test_excluded_licence_does_not_constrain_subset():
    # e.g. a licence that drifted to ND: excluded via excluded_reason, its
    # recorded subset stays what it was
    assert _pairing(license_id="cc-by-nc-nd").subset is DatasetSubset.CORE
