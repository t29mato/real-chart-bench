"""normalize_figure_reference: reduces the notation drift observed in real
Starrydata figure_name values (design §2.3 "図番号表記ゆれ") to a canonical
(number, panel) pair so pairing can compare them for equality.

Fixture values below are the *actual* distinct figure_name strings pulled
from a Phase 2 pilot run against ThermoelectricMaterials_curves.csv.gz
(design §7.9) — not synthesized, so this pins down real-world behavior.
"""

import pytest

from real_chart_bench.domain.figure_reference import (
    FigureReference,
    UnparseableFigureReferenceError,
    normalize_figure_reference,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2(a)", FigureReference(number="2", panel="a")),
        ("2a", FigureReference(number="2", panel="a")),
        ("Figure 6(a)", FigureReference(number="6", panel="a")),
        ("Fig 9(a)", FigureReference(number="9", panel="a")),
        ("6(a)", FigureReference(number="6", panel="a")),
        ("6a", FigureReference(number="6", panel="a")),
        ("7_b", FigureReference(number="7", panel="b")),
        ("6", FigureReference(number="6", panel=None)),
        ("7", FigureReference(number="7", panel=None)),
        ("8", FigureReference(number="8", panel=None)),
        ("Figure 5(b)", FigureReference(number="5", panel="b")),
        ("3(f)", FigureReference(number="3", panel="f")),
        ("9d", FigureReference(number="9", panel="d")),
    ],
)
def test_real_world_notations_normalize_to_the_same_shape(raw, expected):
    assert normalize_figure_reference(raw) == expected


@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("2(a)", "2a"),
        ("6(a)", "Figure 6(a)"),
        ("6(a)", "Fig 6(a)"),
        ("3(b)", "3B"),
    ],
)
def test_equivalent_notations_compare_equal_after_normalization(a, b):
    assert normalize_figure_reference(a) == normalize_figure_reference(b)


def test_panel_is_case_insensitive():
    assert normalize_figure_reference("6C") == normalize_figure_reference("6(c)")


def test_leading_trailing_whitespace_is_ignored():
    assert normalize_figure_reference("  6(a)  ") == FigureReference(number="6", panel="a")


@pytest.mark.parametrize("raw", ["", "   ", "not a figure", "Table 3", "S1"])
def test_unparseable_or_out_of_scope_references_raise(raw):
    with pytest.raises(UnparseableFigureReferenceError):
        normalize_figure_reference(raw)
