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
from real_chart_bench.domain.licensing import LicenseStatus, license_subset


def manifest_entry_subset(entry: Mapping[str, Any]) -> DatasetSubset | None:
    """The subset of a papers.json / figures.json entry, or None when the
    entry's licence admits it to neither subset (ND, unknown, missing).

    An entry with no ``license_id`` key at all (figures.json) is read from its
    ``subset`` key alone. An explicit ``subset`` that contradicts the licence
    raises -- the subset follows the licence, never a hand edit (§7.88)."""
    explicit = parse_subset(entry["subset"]) if "subset" in entry else None
    if "license_id" not in entry:
        return explicit or DatasetSubset.CORE
    decided = license_subset(entry["license_id"])
    if decided.status is not LicenseStatus.REDISTRIBUTABLE:
        return None
    if explicit is not None and explicit is not decided.subset:
        raise ValueError(
            f"manifest entry {entry.get('paper_id') or entry.get('figure_id')}: subset "
            f"{explicit.value!r} contradicts license_id {entry['license_id']!r}"
        )
    return decided.subset


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
