"""TDD for design §7.57: the adapter's handling of x_tick_range /
y_tick_range / tick_range_source -- parsing, backward compatibility with
entries that lack them, and round-trip serialisation preserving field order
and unknown keys.
"""

import json

from real_chart_bench.adapter.verified_pairing_registry import (
    parse_registry,
    serialize_entry,
)
from real_chart_bench.domain.verified_pairing import TickRangeProvenance


def _raw_verified(**overrides):
    raw = {
        "paper_id": "16111",
        "figure_id": "15452",
        "image_path": "p04_embedded_4.jpg",
        "panel_label": None,
        "x_range": [350.0, 500.0],
        "y_range": [0.0, 0.4],
        "status": "verified",
        "verified_at": "2026-09-04",
        "evidence": "ok",
    }
    raw.update(overrides)
    return raw


# --- parsing: new fields absent (the unpromoted majority) -------------------


def test_x_tick_range_defaults_to_none_when_absent():
    entry = parse_registry([_raw_verified()])[0]
    assert entry.x_tick_range is None


def test_y_tick_range_defaults_to_none_when_absent():
    entry = parse_registry([_raw_verified()])[0]
    assert entry.y_tick_range is None


def test_tick_range_source_defaults_to_none_when_absent():
    entry = parse_registry([_raw_verified()])[0]
    assert entry.tick_range_source is None


# --- parsing: new fields present ---------------------------------------------


def test_parses_both_tick_ranges_and_source():
    raw = _raw_verified(
        x_tick_range=[350.0, 500.0],
        y_tick_range=[0.0, 0.4],
        tick_range_source="owner_reviewed",
    )

    entry = parse_registry([raw])[0]

    assert entry.x_tick_range == (350.0, 500.0)
    assert entry.y_tick_range == (0.0, 0.4)
    assert entry.tick_range_source is TickRangeProvenance.OWNER_REVIEWED
    # frame range is untouched and independent
    assert entry.x_range == (350.0, 500.0)


def test_parses_a_tick_range_that_numerically_differs_from_the_frame_range():
    raw = _raw_verified(
        x_range=[330.0, 510.0],
        x_tick_range=[350.0, 500.0],
        tick_range_source="owner_reviewed",
    )

    entry = parse_registry([raw])[0]

    assert entry.x_range == (330.0, 510.0)
    assert entry.x_tick_range == (350.0, 500.0)


# --- round-trip serialisation ------------------------------------------------


def test_round_trip_preserves_unknown_keys_and_field_order():
    raw = _raw_verified(
        x_tick_range=[350.0, 500.0],
        y_tick_range=[0.0, 0.4],
        tick_range_source="owner_reviewed",
    )
    # figure_reference is a real registry.json key the domain model doesn't
    # parse at all -- an "unknown key" from VerifiedPairing's point of view.
    raw = {
        "paper_id": raw["paper_id"],
        "figure_id": raw["figure_id"],
        "figure_reference": "4",
        **{k: v for k, v in raw.items() if k not in ("paper_id", "figure_id")},
    }

    entry = parse_registry([raw])[0]
    out = serialize_entry(entry, base=raw)

    assert out["figure_reference"] == "4"
    assert list(out.keys())[:3] == ["paper_id", "figure_id", "figure_reference"]
    assert out["x_tick_range"] == [350.0, 500.0]
    assert out["y_tick_range"] == [0.0, 0.4]
    assert out["tick_range_source"] == "owner_reviewed"


def test_round_trip_through_json_preserves_new_fields():
    raw = _raw_verified(
        x_tick_range=[350.0, 500.0],
        y_tick_range=[0.0, 0.4],
        tick_range_source="owner_reviewed",
    )
    entry = parse_registry([raw])[0]
    out = serialize_entry(entry, base=raw)
    round_tripped = json.loads(json.dumps(out))

    entry2 = parse_registry([round_tripped])[0]

    assert entry2.x_tick_range == (350.0, 500.0)
    assert entry2.y_tick_range == (0.0, 0.4)
    assert entry2.tick_range_source is TickRangeProvenance.OWNER_REVIEWED


def test_serialize_entry_without_base_omits_absent_tick_range_fields():
    entry = parse_registry([_raw_verified()])[0]
    out = serialize_entry(entry)
    assert "x_tick_range" not in out
    assert "y_tick_range" not in out
    assert "tick_range_source" not in out


def test_round_trip_of_an_unpromoted_entry_stays_free_of_tick_range_keys():
    # the unpromoted majority: no tick range keys in `base`, none should be
    # introduced by a round trip.
    raw = _raw_verified()
    entry = parse_registry([raw])[0]
    out = serialize_entry(entry, base=raw)
    assert "x_tick_range" not in out
    assert "y_tick_range" not in out
    assert "tick_range_source" not in out
