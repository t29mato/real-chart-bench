"""Guards the crop recipes written into data/verified_pairs/registry.json by
scripts/tools/recover_crop_boxes.py (scaling-verification 第0段).

The recipe is a claim about bytes: re-cut the recorded source with the
recorded box and you get the committed crop file back, byte for byte. That
claim is checked here, on the committed files, because the pixel
coordinates in tick_calibration.json are defined against exactly those
bytes -- a recipe that is merely close would silently shift main condition
2's calibration without failing anything else in the repo.

Deliberately NOT asserting a count: the number of recovered boxes is
expected to grow as the missing source images come back (110 of the 124
crop entries have no source image in the repository at all, see
docs/design/figure-fetch-distribution.md §3.5). What must never change is
that every recipe present is exact.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from real_chart_bench.adapter.crop_recovery import reproduce_crop, sha256_file
from real_chart_bench.adapter.verified_pairing_registry import parse_registry

REPO = Path(__file__).resolve().parents[2]
REGISTRY = REPO / "data" / "verified_pairs" / "registry.json"


def _entries_with_a_recipe():
    pairings = parse_registry(json.loads(REGISTRY.read_text()))
    return [p for p in pairings if p.crop is not None]


def _ids(pairings):
    return [f"{p.paper_id}-{p.figure_id}" for p in pairings]


ENTRIES = _entries_with_a_recipe()


def test_the_registry_loads_and_every_recipe_is_paired_with_a_hash():
    # parse_registry itself enforces both-or-neither; reaching here means the
    # committed file satisfies it.
    assert all(p.final_sha256 is not None for p in ENTRIES)


@pytest.mark.parametrize("pairing", ENTRIES, ids=_ids(ENTRIES))
def test_the_recorded_hash_is_the_hash_of_the_committed_crop(pairing):
    crop_path = REPO / pairing.image_path

    assert crop_path.is_file(), f"{pairing.image_path} is recorded but missing"
    assert sha256_file(crop_path) == pairing.final_sha256


@pytest.mark.parametrize("pairing", ENTRIES, ids=_ids(ENTRIES))
def test_recutting_the_source_reproduces_the_crop_byte_for_byte(pairing):
    source_path = REPO / pairing.crop.source_image_path
    crop_path = REPO / pairing.image_path

    assert source_path.is_file(), f"{pairing.crop.source_image_path} is recorded but missing"

    assert reproduce_crop(source_path, pairing.crop) == pairing.final_sha256
    assert sha256_file(crop_path) == pairing.final_sha256
