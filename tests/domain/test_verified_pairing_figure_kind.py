"""TDD for design §7.59: FigureKind / figure_kind / figure_tags on
VerifiedPairing.

The connector-vs-fit-curve distinction (line/scatter/mixed) is abandoned;
see src/real_chart_bench/domain/verified_pairing.py's FigureKind docstring
and docs/design/benchmark-architecture.md §7.59 for why. This file pins
down the resulting binary taxonomy: markers vs line_only, both optional
(older entries may have neither), independent of each other.
"""

import pytest

from real_chart_bench.domain.verified_pairing import (
    FigureKind,
    VerificationStatus,
    VerifiedPairing,
)


def _base_kwargs(**overrides):
    kwargs = dict(
        paper_id="10939",
        figure_id="1527",
        image_path="fig3a.png",
        panel_label=None,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        status=VerificationStatus.VERIFIED,
        verified_at="2026-09-07",
        evidence="ok",
    )
    kwargs.update(overrides)
    return kwargs


# --- FigureKind enum ---------------------------------------------------------


def test_figure_kind_has_exactly_the_two_values():
    assert {k.value for k in FigureKind} == {"markers", "line_only"}


def test_unknown_figure_kind_value_raises():
    with pytest.raises(ValueError):
        FigureKind("scatter")


# --- construction: figure_kind is optional and defaults to None ------------


def test_figure_kind_defaults_to_none():
    pairing = VerifiedPairing(**_base_kwargs())
    assert pairing.figure_kind is None


def test_entry_may_carry_markers_figure_kind():
    pairing = VerifiedPairing(**_base_kwargs(figure_kind=FigureKind.MARKERS))
    assert pairing.figure_kind is FigureKind.MARKERS


def test_entry_may_carry_line_only_figure_kind():
    pairing = VerifiedPairing(**_base_kwargs(figure_kind=FigureKind.LINE_ONLY))
    assert pairing.figure_kind is FigureKind.LINE_ONLY


# --- construction: figure_tags is optional and defaults to empty -----------


def test_figure_tags_defaults_to_empty_tuple():
    pairing = VerifiedPairing(**_base_kwargs())
    assert pairing.figure_tags == ()


def test_entry_may_carry_figure_tags():
    pairing = VerifiedPairing(
        **_base_kwargs(figure_tags=("inset", "dense_overlap"))
    )
    assert pairing.figure_tags == ("inset", "dense_overlap")


def test_entry_may_carry_figure_tags_without_figure_kind():
    # figure_kind and figure_tags are independent -- one may be set without
    # the other (e.g. a migration pass that records tags before a kind is
    # assigned, or vice versa).
    pairing = VerifiedPairing(**_base_kwargs(figure_tags=("error_bars",)))
    assert pairing.figure_kind is None
    assert pairing.figure_tags == ("error_bars",)


def test_entry_may_carry_figure_kind_without_figure_tags():
    pairing = VerifiedPairing(**_base_kwargs(figure_kind=FigureKind.MARKERS))
    assert pairing.figure_kind is FigureKind.MARKERS
    assert pairing.figure_tags == ()
