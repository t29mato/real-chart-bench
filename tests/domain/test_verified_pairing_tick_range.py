"""TDD for design §7.57: x_tick_range / y_tick_range / TickRangeProvenance
on VerifiedPairing, and the promote_tick_range helper.

See src/real_chart_bench/domain/verified_pairing.py's field comments and
promote_tick_range docstring for the invariants these tests pin down:
x_range/y_range record the drawn axis FRAME extent; x_tick_range/
y_tick_range record the printed tick-label extent, and are only ever
attached via promote_tick_range, which refuses any source that is not
"owner_reviewed" in axis_pixel_candidates.json (mirrors design §7.48's
llm_flagged/human_confirmed discipline).
"""

import pytest

from real_chart_bench.domain.verified_pairing import (
    TickRangeProvenance,
    VerificationStatus,
    VerifiedPairing,
    promote_tick_range,
)


def _base_kwargs(**overrides):
    kwargs = dict(
        paper_id="16111",
        figure_id="15452",
        image_path="p04_embedded_4.jpg",
        panel_label=None,
        x_range=(350.0, 500.0),
        y_range=(0.0, 0.4),
        status=VerificationStatus.VERIFIED,
        verified_at="2026-09-04",
        evidence="ok",
    )
    kwargs.update(overrides)
    return kwargs


# --- TickRangeProvenance -----------------------------------------------------


def test_tick_range_provenance_has_owner_reviewed_value():
    assert {p.value for p in TickRangeProvenance} == {"owner_reviewed"}


def test_unknown_tick_range_provenance_value_raises():
    with pytest.raises(ValueError):
        TickRangeProvenance("not-a-real-provenance")


# --- construction: fields default to None / are independent ----------------


def test_tick_range_fields_default_to_none():
    pairing = VerifiedPairing(**_base_kwargs())
    assert pairing.x_tick_range is None
    assert pairing.y_tick_range is None
    assert pairing.tick_range_source is None


def test_entry_may_carry_only_the_frame_range_unpromoted_majority():
    # The unreviewed 73/111 majority: frame range present, no tick range at
    # all. This must remain a perfectly ordinary, legal VerifiedPairing.
    pairing = VerifiedPairing(**_base_kwargs())
    assert pairing.x_range == (350.0, 500.0)
    assert pairing.x_tick_range is None
    assert pairing.tick_range_source is None


def test_entry_may_carry_both_frame_and_tick_range():
    pairing = VerifiedPairing(
        **_base_kwargs(
            x_tick_range=(350.0, 500.0),
            y_tick_range=(0.0, 0.4),
            tick_range_source=TickRangeProvenance.OWNER_REVIEWED,
        )
    )
    assert pairing.x_range == (350.0, 500.0)
    assert pairing.x_tick_range == (350.0, 500.0)
    assert pairing.tick_range_source is TickRangeProvenance.OWNER_REVIEWED


def test_tick_range_may_differ_numerically_from_frame_range():
    # e.g. a chart drawn with margin past the outermost printed tick.
    pairing = VerifiedPairing(
        **_base_kwargs(
            x_range=(330.0, 510.0),
            x_tick_range=(350.0, 500.0),
            tick_range_source=TickRangeProvenance.OWNER_REVIEWED,
        )
    )
    assert pairing.x_range != pairing.x_tick_range


# --- illegality: tick range without a frame range for that axis ------------


def test_x_tick_range_without_x_range_is_illegal():
    with pytest.raises(ValueError, match="x_tick_range requires x_range"):
        VerifiedPairing(
            **_base_kwargs(
                x_range=None,
                x_tick_range=(350.0, 500.0),
                tick_range_source=TickRangeProvenance.OWNER_REVIEWED,
            )
        )


def test_y_tick_range_without_y_range_is_illegal():
    with pytest.raises(ValueError, match="y_tick_range requires y_range"):
        VerifiedPairing(
            **_base_kwargs(
                y_range=None,
                y_tick_range=(0.0, 0.4),
                tick_range_source=TickRangeProvenance.OWNER_REVIEWED,
            )
        )


# --- illegality: tick_range_source <-> presence of a tick range ------------


def test_tick_range_source_required_when_a_tick_range_is_set():
    with pytest.raises(ValueError, match="tick_range_source is required"):
        VerifiedPairing(**_base_kwargs(x_tick_range=(350.0, 500.0)))


def test_tick_range_source_forbidden_when_no_tick_range_is_set():
    with pytest.raises(ValueError, match="tick_range_source is only allowed"):
        VerifiedPairing(**_base_kwargs(tick_range_source=TickRangeProvenance.OWNER_REVIEWED))


# --- promote_tick_range ------------------------------------------------------


def test_promote_tick_range_attaches_ranges_and_source():
    pairing = VerifiedPairing(**_base_kwargs())

    promoted = promote_tick_range(
        pairing,
        x_tick_range=(350.0, 500.0),
        y_tick_range=(0.0, 0.4),
        candidate_status="owner_reviewed",
    )

    assert promoted.x_tick_range == (350.0, 500.0)
    assert promoted.y_tick_range == (0.0, 0.4)
    assert promoted.tick_range_source is TickRangeProvenance.OWNER_REVIEWED
    # the original is untouched (frozen dataclass, pure function)
    assert pairing.x_tick_range is None


def test_promote_tick_range_refuses_llm_candidate_source():
    pairing = VerifiedPairing(**_base_kwargs())

    with pytest.raises(ValueError, match="owner_reviewed"):
        promote_tick_range(
            pairing,
            x_tick_range=(350.0, 500.0),
            y_tick_range=(0.0, 0.4),
            candidate_status="llm_candidate",
        )


def test_promote_tick_range_refuses_excluded_source():
    pairing = VerifiedPairing(**_base_kwargs())

    with pytest.raises(ValueError, match="owner_reviewed"):
        promote_tick_range(
            pairing,
            x_tick_range=(350.0, 500.0),
            y_tick_range=(0.0, 0.4),
            candidate_status="excluded",
        )
