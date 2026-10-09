"""registry.json ``subset`` key (design §7.88.1): backward compatible -- an
entry without the key is read from its licence (every existing entry is
cc-by, i.e. core) -- and a core entry is written back without the key, so the
committed registry stays byte-identical."""

import json

import pytest

from real_chart_bench.adapter.verified_pairing_registry import (
    parse_registry,
    serialize_registry,
)
from real_chart_bench.domain.dataset_subset import DatasetSubset


def _raw(**extra):
    entry = {
        "paper_id": "1",
        "figure_id": "1",
        "image_path": "data/verified_pairs/crops/1/a.png",
        "panel_label": None,
        "x_range": [0.0, 1.0],
        "y_range": [0.0, 1.0],
        "x_scale": "linear",
        "y_scale": "linear",
        "status": "verified",
        "verified_at": "2026-10-10",
        "evidence": "x",
        "license_id": "cc-by",
    }
    entry.update(extra)
    return entry


def test_entry_without_subset_key_is_core():
    assert parse_registry([_raw()])[0].subset is DatasetSubset.CORE


def test_entry_without_licence_or_subset_is_core():
    raw = _raw()
    del raw["license_id"]
    assert parse_registry([raw])[0].subset is DatasetSubset.CORE


def test_nc_licence_without_subset_key_is_read_as_nc():
    raw = _raw(license_id="cc-by-nc", image_path="data/verified_pairs_nc/crops/1/a.png")
    assert parse_registry([raw])[0].subset is DatasetSubset.NC


def test_explicit_nc_subset_is_parsed():
    raw = _raw(
        license_id="cc-by-nc-sa",
        subset="nc",
        image_path="data/verified_pairs_nc/crops/1/a.png",
    )
    assert parse_registry([raw])[0].subset is DatasetSubset.NC


def test_explicit_subset_that_contradicts_the_licence_fails_loudly():
    with pytest.raises(ValueError):
        parse_registry([_raw(subset="nc", image_path="data/verified_pairs_nc/crops/1/a.png")])


def test_core_entry_round_trips_without_a_subset_key():
    raw = [_raw()]
    out = serialize_registry(parse_registry(raw), raw)
    assert "subset" not in out[0]
    assert json.dumps(out) == json.dumps(raw)


def test_nc_entry_is_written_with_its_subset():
    raw = _raw(license_id="cc-by-nc", image_path="data/verified_pairs_nc/crops/1/a.png")
    out = serialize_registry(parse_registry([raw]))
    assert out[0]["subset"] == "nc"
