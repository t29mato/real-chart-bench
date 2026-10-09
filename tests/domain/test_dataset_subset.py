"""Dataset subsets (design §7.88.1): ``core`` and ``nc`` live in separate
directories and on separate dataset_versions, so they are never pooled."""

import pytest

from real_chart_bench.domain.dataset_subset import (
    DatasetSubset,
    dataset_version_for_subset,
    distribution_dir_name,
    parse_subset,
    subset_of_dataset_version,
)


def test_values_are_the_manifest_strings():
    assert DatasetSubset.CORE.value == "core"
    assert DatasetSubset.NC.value == "nc"


def test_parse_subset_defaults_missing_to_core_for_backward_compatibility():
    assert parse_subset(None) is DatasetSubset.CORE
    assert parse_subset("core") is DatasetSubset.CORE
    assert parse_subset("nc") is DatasetSubset.NC


def test_parse_subset_rejects_unknown_values():
    with pytest.raises(ValueError):
        parse_subset("NC ")


def test_distribution_dirs_differ():
    assert distribution_dir_name(DatasetSubset.CORE) == "verified_pairs"
    assert distribution_dir_name(DatasetSubset.NC) == "verified_pairs_nc"


def test_core_dataset_version_is_unchanged():
    assert dataset_version_for_subset("v0-eval-pilot-n91", DatasetSubset.CORE) == (
        "v0-eval-pilot-n91"
    )


def test_nc_dataset_version_is_prefixed():
    assert dataset_version_for_subset("v0-eval-pilot-n12-noaxis", DatasetSubset.NC) == (
        "nc-v0-eval-pilot-n12-noaxis"
    )


def test_nc_prefix_is_not_applied_twice():
    with pytest.raises(ValueError):
        dataset_version_for_subset("nc-v0-eval-pilot-n12", DatasetSubset.NC)


@pytest.mark.parametrize(
    ("version", "subset"),
    [
        ("v0-eval-pilot-n91", DatasetSubset.CORE),
        ("synthetic-plotqa-dot-line-n100", DatasetSubset.CORE),
        (None, DatasetSubset.CORE),
        ("nc-v0-eval-pilot-n12", DatasetSubset.NC),
    ],
)
def test_subset_of_dataset_version(version, subset):
    assert subset_of_dataset_version(version) is subset
