"""Reads the dataset subset (design §7.88.1) off the JSON shapes the scripts
pass around -- manifest entries (papers.json / figures.json) and results
files -- with the backward-compatibility rule in one place: no ``subset`` key
means ``core``, unless the entry's own licence says otherwise."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from real_chart_bench.domain.dataset_subset import (
    DatasetSubset,
    parse_subset,
    subset_of_dataset_version,
)
from real_chart_bench.domain.licensing import (
    LicenseStatus,
    licence_admitted_for,
    license_subset,
)


def declared_subset(entry: Mapping[str, Any]) -> DatasetSubset:
    """The one default rule for the subset a JSON record (registry entry,
    papers.json / figures.json row) declares: its ``subset`` key if present,
    else the subset its ``license_id`` admits it to, else ``core`` (every
    record written before the subset existed). Whether the licence actually
    admits the record is a separate question -- ``manifest_entry_subset`` /
    ``VerifiedPairing.is_licence_admitted``."""
    if "subset" in entry:
        return parse_subset(entry["subset"])
    decided = license_subset(entry.get("license_id"))
    if decided.status is LicenseStatus.REDISTRIBUTABLE and decided.subset is not None:
        return decided.subset
    return DatasetSubset.CORE


def subset_fields(subset: DatasetSubset) -> dict[str, str]:
    """The ``subset`` key to write on a new record: none for ``core`` (the
    default, so core records look exactly as they always have), explicit for
    ``nc`` (design §7.88.1)."""
    return {} if subset is DatasetSubset.CORE else {"subset": subset.value}


def manifest_entry_subset(entry: Mapping[str, Any]) -> DatasetSubset | None:
    """The subset of a papers.json / figures.json entry, or None when the
    entry's licence admits it to neither subset (ND, unknown, missing).

    An entry with no ``license_id`` key at all (figures.json) is read from its
    declared subset alone. An explicit ``subset`` that contradicts the licence
    raises -- the subset follows the licence, never a hand edit (§7.88)."""
    subset = declared_subset(entry)
    if "license_id" not in entry:
        return subset
    try:
        admitted = licence_admitted_for(entry["license_id"], subset)
    except ValueError as exc:
        raise ValueError(
            f"manifest entry {entry.get('paper_id') or entry.get('figure_id')}: {exc}"
        ) from exc
    return subset if admitted else None


def result_subset(result: Mapping[str, Any]) -> DatasetSubset:
    """The subset a results file was scored on. ``core`` results carry no
    ``subset`` key and an unprefixed dataset_version (all results to date);
    ``nc`` results carry both ``"subset": "nc"`` and an ``nc-`` dataset_version.
    Anything in between is refused, so an nc score can never be pooled into a
    core section of the leaderboard (or the other way round)."""
    flagged = parse_subset(result.get("subset"))
    by_version = subset_of_dataset_version(result.get("dataset_version"))
    if flagged is not by_version:
        raise ValueError(
            f"result {result.get('model_id')!r}: subset {flagged.value!r} but dataset_version "
            f"{result.get('dataset_version')!r} -- refusing to pool core and nc"
        )
    return flagged
