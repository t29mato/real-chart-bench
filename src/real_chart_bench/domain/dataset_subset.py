"""Dataset subsets by figure licence (design §7.88 / §7.88.1).

``core`` holds the CC BY / CC BY-SA / CC0 figures and is what the main
leaderboard scores. ``nc`` holds the CC BY-NC / CC BY-NC-SA figures: it is
distributed from its own directory with its own ATTRIBUTION (non-commercial
use only), and scored on its own dataset_version so its numbers are never
pooled with ``core``'s.

Records written before the subset existed carry no ``subset`` key; they are
all ``core`` (every licence admitted before §7.88 was a core licence).
"""

from __future__ import annotations

from enum import Enum


class DatasetSubset(Enum):
    CORE = "core"
    NC = "nc"


_NC_VERSION_PREFIX = "nc-"

_DISTRIBUTION_DIRS = {
    DatasetSubset.CORE: "verified_pairs",
    DatasetSubset.NC: "verified_pairs_nc",
}


def parse_subset(raw: str | None) -> DatasetSubset:
    """A manifest/registry ``subset`` value; a missing key means ``core``."""
    if raw is None:
        return DatasetSubset.CORE
    return DatasetSubset(raw)


def distribution_dir_name(subset: DatasetSubset) -> str:
    """The directory under ``data/`` that holds this subset's committed
    figures and its ATTRIBUTION.md."""
    return _DISTRIBUTION_DIRS[subset]


def dataset_version_for_subset(base: str, subset: DatasetSubset) -> str:
    """``core`` keeps the version string it always had (existing results stay
    byte-identical); ``nc`` gets an ``nc-`` prefix, so the leaderboard's
    one-section-per-dataset_version rule keeps the two apart."""
    if base.startswith(_NC_VERSION_PREFIX):
        raise ValueError(f"dataset_version {base!r} already names a subset")
    if subset is DatasetSubset.CORE:
        return base
    return _NC_VERSION_PREFIX + base


def subset_of_dataset_version(version: str | None) -> DatasetSubset:
    if version is not None and version.startswith(_NC_VERSION_PREFIX):
        return DatasetSubset.NC
    return DatasetSubset.CORE


def strip_subset_prefix(version: str | None) -> str | None:
    """The version without its subset prefix (``core`` versions unchanged)."""
    if version is not None and version.startswith(_NC_VERSION_PREFIX):
        return version[len(_NC_VERSION_PREFIX) :]
    return version
