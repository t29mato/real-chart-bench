"""TDD for design §7.4x (戦略メモ「柱G」): the adapter's handling of
rejection_category / gt_suspect_status / rejection_evidence -- parsing,
backward compatibility with entries that lack them, and round-trip
serialisation preserving field order and unknown keys.
"""

import json

import pytest

from real_chart_bench.adapter.verified_pairing_registry import (
    parse_registry,
    serialize_entry,
)
from real_chart_bench.domain.verified_pairing import (
    GtSuspectStatus,
    RejectionCategory,
)


def _raw_rejected(**overrides):
    raw = {
        "paper_id": "17049",
        "figure_id": "13287",
        "image_path": None,
        "panel_label": None,
        "x_range": None,
        "y_range": None,
        "status": "rejected",
        "verified_at": "2026-08-30",
        "evidence": "no chart found among the extracted images",
    }
    raw.update(overrides)
    return raw


# --- parsing: new fields absent (backward compatibility) -------------------


def test_rejection_category_defaults_to_none_when_absent():
    entry = parse_registry([_raw_rejected()])[0]
    assert entry.rejection_category is None


def test_gt_suspect_status_defaults_to_none_when_absent():
    entry = parse_registry([_raw_rejected()])[0]
    assert entry.gt_suspect_status is None


def test_rejection_evidence_defaults_to_none_when_absent():
    entry = parse_registry([_raw_rejected()])[0]
    assert entry.rejection_evidence is None


# --- parsing: new fields present --------------------------------------------


def test_parses_rejection_category():
    entry = parse_registry([_raw_rejected(rejection_category="image")])[0]
    assert entry.rejection_category is RejectionCategory.IMAGE


def test_parses_gt_suspect_status_alongside_gt_suspect_category():
    entry = parse_registry(
        [
            _raw_rejected(
                rejection_category="gt_suspect",
                gt_suspect_status="llm_flagged",
            )
        ]
    )[0]
    assert entry.rejection_category is RejectionCategory.GT_SUSPECT
    assert entry.gt_suspect_status is GtSuspectStatus.LLM_FLAGGED


def test_parses_rejection_evidence_structured_fields():
    entry = parse_registry(
        [
            _raw_rejected(
                rejection_category="gt_suspect",
                gt_suspect_status="llm_flagged",
                rejection_evidence={
                    "axis_range_mismatch": False,
                    "point_count_mismatch": None,
                    "y_value_offset_magnitude": 100.0,
                    "missing_series": True,
                },
            )
        ]
    )[0]
    assert entry.rejection_evidence.y_value_offset_magnitude == 100.0
    assert entry.rejection_evidence.missing_series is True
    assert entry.rejection_evidence.axis_range_mismatch is False


def test_unknown_rejection_category_string_raises():
    with pytest.raises(ValueError):
        parse_registry([_raw_rejected(rejection_category="bogus")])


def test_unknown_gt_suspect_status_string_raises():
    with pytest.raises(ValueError):
        parse_registry(
            [_raw_rejected(rejection_category="gt_suspect", gt_suspect_status="bogus")]
        )


def test_verified_entry_may_carry_a_gt_suspect_flag():
    raw = _raw_rejected(
        status="verified",
        x_range=[0.0, 1.0],
        y_range=[0.0, 1.0],
        rejection_category="gt_suspect",
        gt_suspect_status="human_confirmed",
    )
    entry = parse_registry([raw])[0]
    assert entry.status.value == "verified"
    assert entry.rejection_category is RejectionCategory.GT_SUSPECT
    assert entry.is_confirmed_gt_error is True


# --- round-trip serialisation ------------------------------------------------


def test_round_trip_preserves_unknown_keys_and_field_order():
    raw = _raw_rejected(
        rejection_category="image",
    )
    # figure_reference is a real registry.json key the domain model doesn't
    # parse at all -- an "unknown key" from VerifiedPairing's point of view.
    raw = {"paper_id": raw["paper_id"], "figure_id": raw["figure_id"], "figure_reference": "7", **{
        k: v for k, v in raw.items() if k not in ("paper_id", "figure_id")
    }}

    entry = parse_registry([raw])[0]
    out = serialize_entry(entry, base=raw)

    assert out["figure_reference"] == "7"
    assert list(out.keys())[:3] == ["paper_id", "figure_id", "figure_reference"]
    assert out["rejection_category"] == "image"


def test_round_trip_through_json_preserves_new_fields():
    raw = _raw_rejected(
        rejection_category="gt_suspect",
        gt_suspect_status="llm_flagged",
        rejection_evidence={
            "axis_range_mismatch": None,
            "point_count_mismatch": None,
            "y_value_offset_magnitude": 100.0,
            "missing_series": None,
        },
    )
    entry = parse_registry([raw])[0]
    out = serialize_entry(entry, base=raw)
    round_tripped = json.loads(json.dumps(out))

    entry2 = parse_registry([round_tripped])[0]
    assert entry2.rejection_category is RejectionCategory.GT_SUSPECT
    assert entry2.gt_suspect_status is GtSuspectStatus.LLM_FLAGGED
    assert entry2.rejection_evidence.y_value_offset_magnitude == 100.0


def test_serialize_entry_without_base_builds_fresh_dict():
    raw = _raw_rejected(rejection_category="pairing")
    entry = parse_registry([raw])[0]
    out = serialize_entry(entry)
    assert out["rejection_category"] == "pairing"
    assert out["paper_id"] == "17049"


def test_serialize_omits_rejection_fields_when_absent_and_no_base():
    raw = _raw_rejected()
    entry = parse_registry([raw])[0]
    out = serialize_entry(entry)
    assert "rejection_category" not in out
    assert "gt_suspect_status" not in out
    assert "rejection_evidence" not in out
