"""Reading the subset of manifest entries and results files (design §7.88.1)."""

import pytest

from real_chart_bench.domain.dataset_subset import DatasetSubset
from real_chart_bench.usecase.dataset_subsets import (
    manifest_entry_subset,
    result_subset,
)

# --- papers.json / figures.json entries -------------------------------------------


def test_legacy_cc_by_entry_without_key_is_core():
    assert manifest_entry_subset({"paper_id": "1", "license_id": "cc-by"}) is DatasetSubset.CORE


def test_entry_with_nc_licence_and_no_key_is_nc():
    assert manifest_entry_subset({"license_id": "cc-by-nc"}) is DatasetSubset.NC


def test_explicit_key_is_used():
    assert manifest_entry_subset({"license_id": "cc-by-nc", "subset": "nc"}) is DatasetSubset.NC


def test_explicit_key_contradicting_the_licence_raises():
    with pytest.raises(ValueError):
        manifest_entry_subset({"license_id": "cc-by", "subset": "nc"})


@pytest.mark.parametrize("license_id", ["cc-by-nc-nd", "other-oa", None])
def test_entry_whose_licence_is_not_admitted_has_no_subset(license_id):
    assert manifest_entry_subset({"license_id": license_id}) is None


def test_figures_json_entry_without_licence_and_without_key_is_core():
    # figures.json carries no licence; the key is the only signal, absent = core
    assert manifest_entry_subset({"figure_id": "9048"}) is DatasetSubset.CORE


# --- results/*.json -----------------------------------------------------------------


def test_existing_result_is_core():
    assert result_subset({"dataset_version": "v0-eval-pilot-n91"}) is DatasetSubset.CORE


def test_nc_result():
    result = {"dataset_version": "nc-v0-eval-pilot-n12", "subset": "nc"}
    assert result_subset(result) is DatasetSubset.NC


def test_nc_flag_on_a_core_dataset_version_is_refused():
    with pytest.raises(ValueError, match="pool"):
        result_subset({"dataset_version": "v0-eval-pilot-n91", "subset": "nc"})


def test_nc_dataset_version_without_the_flag_is_refused():
    with pytest.raises(ValueError, match="pool"):
        result_subset({"dataset_version": "nc-v0-eval-pilot-n12"})
