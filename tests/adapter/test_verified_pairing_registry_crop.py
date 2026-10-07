"""TDD for the crop recipe in registry.json (scaling-verification 第0段 /
figure-fetch-distribution §3.2): parsing, backward compatibility with the
entries that carry no recipe, and round-trip serialisation that leaves an
un-recovered entry untouched.
"""

import json

import pytest

from real_chart_bench.adapter.verified_pairing_registry import (
    parse_registry,
    serialize_entry,
)
from real_chart_bench.domain.verified_pairing import CropRecipe

SHA = "6c3cc3b3e44b5b2b3bbd07c49d2b9b98c2f4ea86a5c3df3e0f9c0e8e1f1a2b3c"


def _raw_verified(**overrides):
    raw = {
        "paper_id": "36305",
        "figure_id": "45818",
        "image_path": "data/verified_pairs/crops/36305/fig5a.png",
        "panel_label": "a",
        "x_range": [0.0008, 0.0011],
        "y_range": [0.0, 1.4],
        "status": "verified",
        "verified_at": "2026-10-07",
        "evidence": "ok",
    }
    raw.update(overrides)
    return raw


def _raw_crop(**overrides):
    crop = {
        "source_image_path": "data/verified_pairs/images/36305/p09_embedded_12.jpg",
        "box": [0, 0, 1605, 1425],
        "rotation_deg": 0,
    }
    crop.update(overrides)
    return crop


# --- absent: every entry before 第0段 ---------------------------------------


def test_an_entry_without_a_recipe_parses_with_no_crop():
    entry = parse_registry([_raw_verified()])[0]

    assert entry.crop is None
    assert entry.final_sha256 is None


def test_serialising_an_entry_without_a_recipe_adds_no_keys():
    raw = _raw_verified()

    out = serialize_entry(parse_registry([raw])[0], base=raw)

    assert "crop" not in out
    assert "final_sha256" not in out


# --- present -----------------------------------------------------------------


def test_parses_the_recovered_box_and_the_final_hash():
    raw = _raw_verified(crop=_raw_crop(), final_sha256=SHA)

    entry = parse_registry([raw])[0]

    assert entry.crop == CropRecipe(
        source_image_path="data/verified_pairs/images/36305/p09_embedded_12.jpg",
        box=(0, 0, 1605, 1425),
        rotation_deg=0,
    )
    assert entry.final_sha256 == SHA


def test_the_rotation_defaults_to_zero_when_the_recipe_omits_it():
    raw = _raw_crop()
    del raw["rotation_deg"]

    entry = parse_registry([_raw_verified(crop=raw, final_sha256=SHA)])[0]

    assert entry.crop.rotation_deg == 0


def test_round_trips_the_recipe_without_reshuffling_the_entry():
    raw = _raw_verified(
        crop=_raw_crop(rotation_deg=90), final_sha256=SHA, figure_reference="5(a)"
    )

    out = serialize_entry(parse_registry([raw])[0], base=raw)

    assert out["crop"] == {
        "source_image_path": "data/verified_pairs/images/36305/p09_embedded_12.jpg",
        "box": [0, 0, 1605, 1425],
        "rotation_deg": 90,
    }
    assert out["final_sha256"] == SHA
    assert out["figure_reference"] == "5(a)"
    assert json.loads(json.dumps(out)) == out


def test_the_serialised_box_is_a_list_of_plain_ints():
    raw = _raw_verified(crop=_raw_crop(), final_sha256=SHA)

    out = serialize_entry(parse_registry([raw])[0], base=raw)

    assert all(type(v) is int for v in out["crop"]["box"])


def test_dropping_a_recipe_removes_both_keys_from_the_serialised_entry():
    raw = _raw_verified(crop=_raw_crop(), final_sha256=SHA)
    entry = parse_registry([raw])[0]
    stripped = type(entry)(
        **{
            **{f: getattr(entry, f) for f in entry.__dataclass_fields__},
            "crop": None,
            "final_sha256": None,
        }
    )

    out = serialize_entry(stripped, base=raw)

    assert "crop" not in out
    assert "final_sha256" not in out


# --- validation reaches the loader ------------------------------------------


def test_a_registry_recipe_with_no_hash_fails_to_load():
    with pytest.raises(ValueError, match="final_sha256"):
        parse_registry([_raw_verified(crop=_raw_crop())])


def test_a_registry_hash_with_no_recipe_fails_to_load():
    with pytest.raises(ValueError, match="crop"):
        parse_registry([_raw_verified(final_sha256=SHA)])


def test_a_registry_recipe_with_a_bad_rotation_fails_to_load():
    with pytest.raises(ValueError, match="rotation_deg"):
        parse_registry([_raw_verified(crop=_raw_crop(rotation_deg=45), final_sha256=SHA)])
