"""TDD for design §7.84: the registry's second y axis (y2_range / y2_scale) --
parsing, backward compatibility with the 157 entries that have neither, and
round-trip serialisation that leaves an entry without a second axis untouched.
"""

import json

import pytest

from real_chart_bench.adapter.verified_pairing_registry import (
    parse_registry,
    serialize_entry,
)
from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.verified_pairing import VerifiedPairing


def _raw_verified(**overrides):
    raw = {
        "paper_id": "10939",
        "figure_id": "1528",
        "image_path": "data/verified_pairs/crops/10939/fig3b.png",
        "panel_label": "b",
        "x_range": [300.0, 900.0],
        "y_range": [-110.0, -30.0],
        "status": "verified",
        "verified_at": "2026-10-07",
        "evidence": "ok",
    }
    raw.update(overrides)
    return raw


# --- absent: every entry in the registry today ------------------------------


def test_a_figure_with_one_y_axis_parses_with_no_second_range():
    entry = parse_registry([_raw_verified()])[0]

    assert entry.y2_range is None
    assert entry.y2_scale is None


def test_serialising_an_entry_without_a_second_axis_adds_no_keys():
    raw = _raw_verified()

    out = serialize_entry(parse_registry([raw])[0], base=raw)

    assert "y2_range" not in out
    assert "y2_scale" not in out


# --- present ------------------------------------------------------------------


def test_parses_the_second_y_range_and_scale():
    raw = _raw_verified(y2_range=[0.0, 300.0], y2_scale="log")

    entry = parse_registry([raw])[0]

    assert entry.y2_range == (0.0, 300.0)
    assert entry.y2_scale is ScaleType.LOG


def test_the_second_y_scale_defaults_to_linear_when_only_the_range_is_given():
    entry = parse_registry([_raw_verified(y2_range=[0.0, 300.0])])[0]

    assert entry.y2_scale is ScaleType.LINEAR


def test_round_trips_the_second_axis_without_reshuffling_the_entry():
    raw = _raw_verified(y2_range=[0.0, 300.0], y2_scale="log", figure_reference="Fig. 3(b)")

    out = serialize_entry(parse_registry([raw])[0], base=raw)

    assert out["y2_range"] == [0.0, 300.0]
    assert out["y2_scale"] == "log"
    assert out["figure_reference"] == "Fig. 3(b)"
    assert json.loads(json.dumps(out)) == out


# --- validation ------------------------------------------------------------------


def test_a_second_y_scale_without_a_second_y_range_is_rejected():
    with pytest.raises(ValueError, match="y2_range"):
        VerifiedPairing(
            paper_id="1",
            figure_id="2",
            image_path="i.png",
            panel_label=None,
            x_range=(0.0, 1.0),
            y_range=(0.0, 1.0),
            status=parse_registry([_raw_verified()])[0].status,
            verified_at="2026-10-07",
            evidence="ok",
            y2_scale=ScaleType.LOG,
        )


def test_a_second_y_range_without_a_first_one_is_rejected():
    raw = _raw_verified(y_range=None, y2_range=[0.0, 300.0])

    with pytest.raises(ValueError, match="y_range"):
        parse_registry([raw])
