"""TDD for design §7.59: the adapter's handling of figure_kind /
figure_tags -- parsing, backward compatibility with entries that lack
them, an unknown enum value, and round-trip serialisation preserving field
order and unknown keys.
"""

import json

import pytest

from real_chart_bench.adapter.verified_pairing_registry import (
    parse_registry,
    serialize_entry,
)
from real_chart_bench.domain.verified_pairing import FigureKind


def _raw_verified(**overrides):
    raw = {
        "paper_id": "10939",
        "figure_id": "1527",
        "image_path": "fig3a.png",
        "panel_label": None,
        "x_range": [0.0, 1.0],
        "y_range": [0.0, 1.0],
        "status": "verified",
        "verified_at": "2026-09-07",
        "evidence": "ok",
    }
    raw.update(overrides)
    return raw


# --- parsing: new fields absent (pre-§7.59 entries) -------------------------


def test_figure_kind_defaults_to_none_when_absent():
    entry = parse_registry([_raw_verified()])[0]
    assert entry.figure_kind is None


def test_figure_tags_defaults_to_empty_tuple_when_absent():
    entry = parse_registry([_raw_verified()])[0]
    assert entry.figure_tags == ()


# --- parsing: new fields present ---------------------------------------------


def test_parses_figure_kind_markers():
    raw = _raw_verified(figure_kind="markers")
    entry = parse_registry([raw])[0]
    assert entry.figure_kind is FigureKind.MARKERS


def test_parses_figure_kind_line_only():
    raw = _raw_verified(figure_kind="line_only")
    entry = parse_registry([raw])[0]
    assert entry.figure_kind is FigureKind.LINE_ONLY


def test_parses_figure_tags():
    raw = _raw_verified(figure_tags=["inset", "dense_overlap"])
    entry = parse_registry([raw])[0]
    assert entry.figure_tags == ("inset", "dense_overlap")


def test_unknown_figure_kind_value_raises_on_parse():
    raw = _raw_verified(figure_kind="scatter")
    with pytest.raises(ValueError):
        parse_registry([raw])


def test_parses_figure_tags_without_figure_kind():
    raw = _raw_verified(figure_tags=["error_bars"])
    entry = parse_registry([raw])[0]
    assert entry.figure_kind is None
    assert entry.figure_tags == ("error_bars",)


# --- round-trip serialisation ------------------------------------------------


def test_round_trip_preserves_unknown_keys_and_field_order():
    raw = _raw_verified(figure_kind="markers", figure_tags=["inset"])
    # figure_reference is a real registry.json key the domain model doesn't
    # parse at all -- an "unknown key" from VerifiedPairing's point of view.
    raw = {
        "paper_id": raw["paper_id"],
        "figure_id": raw["figure_id"],
        "figure_reference": "3(a)",
        **{k: v for k, v in raw.items() if k not in ("paper_id", "figure_id")},
    }

    entry = parse_registry([raw])[0]
    out = serialize_entry(entry, base=raw)

    assert out["figure_reference"] == "3(a)"
    assert list(out.keys())[:3] == ["paper_id", "figure_id", "figure_reference"]
    assert out["figure_kind"] == "markers"
    assert out["figure_tags"] == ["inset"]


def test_round_trip_through_json_preserves_new_fields():
    raw = _raw_verified(figure_kind="line_only", figure_tags=["scan_artifact"])
    entry = parse_registry([raw])[0]
    out = serialize_entry(entry, base=raw)
    round_tripped = json.loads(json.dumps(out))

    entry2 = parse_registry([round_tripped])[0]

    assert entry2.figure_kind is FigureKind.LINE_ONLY
    assert entry2.figure_tags == ("scan_artifact",)


def test_serialize_entry_without_base_omits_absent_new_fields():
    entry = parse_registry([_raw_verified()])[0]
    out = serialize_entry(entry)
    assert "figure_kind" not in out
    assert "figure_tags" not in out


def test_round_trip_of_an_unmigrated_entry_stays_free_of_new_keys():
    # entries not yet touched by the §7.59 migration: no figure_kind/
    # figure_tags keys in `base`, none should be introduced by a round trip.
    raw = _raw_verified()
    entry = parse_registry([raw])[0]
    out = serialize_entry(entry, base=raw)
    assert "figure_kind" not in out
    assert "figure_tags" not in out


def test_round_trip_of_entry_with_figure_tags_but_no_figure_kind():
    raw = _raw_verified(figure_tags=["reference_curve"])
    entry = parse_registry([raw])[0]
    out = serialize_entry(entry, base=raw)

    assert "figure_kind" not in out
    assert out["figure_tags"] == ["reference_curve"]

    entry2 = parse_registry([out])[0]
    assert entry2.figure_kind is None
    assert entry2.figure_tags == ("reference_curve",)
