"""Sibling Starrydata figures: the same printed plot digitized twice (sigma and
log(sigma), or one curve set under two figure records). Design
docs/design/pairing-automation.md 12.9."""

from __future__ import annotations

import math

from real_chart_bench.domain.pairing_siblings import (
    SiblingFigure,
    are_siblings,
    normalise_reference,
    sibling_groups,
)

XS = (0.00085, 0.0009, 0.00096, 0.001)
SIGMA = (673.6, 587.6, 525.3, 400.0)


def _fig(fid, ref="8a sigma", prop_y="Electrical conductivity", unit_y="S*m^(-1)",
         curves=None, prop_x="Inverse temperature", unit_x="K^(-1)", paper="1"):
    return SiblingFigure(paper, fid, ref, prop_x, unit_x, prop_y, unit_y,
                         curves if curves is not None else ((XS, SIGMA),))


def _log(ys):
    return tuple(100 * math.log10(v) - 200 for v in ys)  # Starrydata's stored log(sigma)


def test_reference_drops_figure_word_and_trailing_quantity():
    assert normalise_reference("8a sigma") == "8a"
    assert normalise_reference("Fig. 8(a)") == "8a"
    assert normalise_reference("FIGURE4(ZT)") == "4zt"
    assert normalise_reference("") == ""


def test_sigma_and_its_logarithm_are_siblings():
    a = _fig("1", "8a sigma", curves=((XS, SIGMA),))
    b = _fig("2", "8a", prop_y="log(Electrical conductivity)", curves=((XS, _log(SIGMA)),))
    assert are_siblings(a, b) and are_siblings(b, a)


def test_logarithm_relation_must_be_monotone():
    wobbly = (_log(SIGMA)[0], _log(SIGMA)[2], _log(SIGMA)[1], _log(SIGMA)[3])
    a = _fig("1", "8a sigma")
    b = _fig("2", "8a", prop_y="log(Electrical conductivity)", curves=((XS, wobbly),))
    assert not are_siblings(a, b)


def test_two_figure_records_with_identical_points_are_siblings():
    a = _fig("1", "3")
    b = _fig("2", "3")
    assert are_siblings(a, b)


def test_identical_points_under_a_different_quantity_are_not_siblings():
    a = _fig("1", "3")
    b = _fig("2", "3", prop_y="Seebeck coefficient", unit_y="V*K^(-1)")
    assert not are_siblings(a, b)


def test_monotone_pair_of_different_quantities_is_not_a_sibling():
    # sigma(T) and kappa(T) both rising with T are co-monotone but not one plot twice
    a = _fig("1", "2", prop_y="Electrical conductivity")
    b = _fig("2", "2", prop_y="Thermal conductivity", unit_y="W*m^(-1)*K^(-1)",
             curves=((XS, (1.0, 2.0, 3.0, 4.0)),))
    assert not are_siblings(a, b)


def test_different_figure_reference_is_not_a_sibling():
    a = _fig("1", "8a sigma")
    b = _fig("2", "8b", prop_y="log(Electrical conductivity)", curves=((XS, _log(SIGMA)),))
    assert not are_siblings(a, b)


def test_different_x_points_are_not_a_sibling():
    other_x = tuple(x + 0.0001 for x in XS)
    a = _fig("1", "8a sigma")
    b = _fig("2", "8a", prop_y="log(Electrical conductivity)", curves=((other_x, _log(SIGMA)),))
    assert not are_siblings(a, b)


def test_curve_order_does_not_matter_and_each_curve_is_used_once():
    xs2 = (0.0009, 0.00095, 0.001)
    s2 = (100.0, 50.0, 10.0)
    a = _fig("1", "8a sigma", curves=((XS, SIGMA), (xs2, s2)))
    b = _fig("2", "8a", prop_y="log(Electrical conductivity)",
             curves=((xs2, _log(s2)), (XS, _log(SIGMA))))
    assert are_siblings(a, b)


def test_at_least_half_of_the_smaller_figures_curves_must_match():
    xs2 = (0.0009, 0.00095, 0.001)
    a = _fig("1", "8a sigma", curves=((XS, SIGMA), (xs2, (100.0, 50.0, 10.0))))
    b = _fig("2", "8a", prop_y="log(Electrical conductivity)",
             curves=((XS, _log(SIGMA)), (xs2, (3.0, 1.0, 2.0))))
    assert are_siblings(a, b)  # 1 of 2 matches
    c = _fig("3", "8a", prop_y="log(Electrical conductivity)",
             curves=((XS, (3.0, 1.0, 2.0, 0.0)), (xs2, (3.0, 1.0, 2.0))))
    assert not are_siblings(a, c)  # 0 of 2


def test_figures_without_curves_or_across_papers_are_never_siblings():
    a = _fig("1", "8a sigma")
    assert not are_siblings(a, _fig("2", "8a sigma", curves=()))
    assert not are_siblings(a, _fig("2", "8a sigma", paper="2"))


def test_two_point_log_relation_is_too_weak_to_call_a_sibling():
    a = _fig("1", "8a sigma", curves=(((1.0, 2.0), (10.0, 100.0)),))
    b = _fig("2", "8a", prop_y="log(Electrical conductivity)", curves=(((1.0, 2.0), (1.0, 2.0)),))
    assert not are_siblings(a, b)


def test_groups_are_transitive_and_listed_by_numeric_id():
    a = _fig("10", "5a", curves=((XS, SIGMA),))
    b = _fig("9", "5a", prop_y="log(Electrical conductivity)", curves=((XS, _log(SIGMA)),))
    c = _fig("11", "5a", curves=((XS, SIGMA),))
    lone = _fig("12", "7")
    assert sibling_groups([a, b, c, lone]) == [("9", "10", "11")]


def test_no_siblings_gives_no_groups():
    assert sibling_groups([_fig("1", "1"), _fig("2", "2")]) == []
    assert sibling_groups([]) == []


def test_tied_x_values_must_have_tied_logs_and_a_flat_step_is_not_monotone():
    xs = (1.0, 2.0, 3.0, 4.0)
    a = _fig("1", "8a sigma", curves=((xs, (10.0, 10.0, 100.0, 1000.0)),))
    ok = _fig("2", "8a", prop_y="log(Electrical conductivity)",
              curves=((xs, (1.0, 1.0, 2.0, 3.0)),))
    assert are_siblings(a, ok)
    split_tie = _fig("3", "8a", prop_y="log(Electrical conductivity)",
                     curves=((xs, (1.0, 1.5, 2.0, 3.0)),))
    assert not are_siblings(a, split_tie)
    flat = _fig("4", "8a", prop_y="log(Electrical conductivity)",
                curves=((xs, (1.0, 2.0, 2.0, 3.0)),))
    a2 = _fig("5", "8a sigma", curves=((xs, (10.0, 20.0, 100.0, 1000.0)),))
    assert not are_siblings(a2, flat)


def test_a_different_x_quantity_or_y_unit_is_not_a_sibling():
    a = _fig("1", "3")
    assert not are_siblings(a, _fig("2", "3", prop_x="Temperature", unit_x="K"))
    assert not are_siblings(a, _fig("2", "3", unit_y="W*m^(-1)"))
