"""Curve is the domain value object for a single (x, y) series.

Boundary cases covered here map to docs/design/benchmark-architecture.md §3.3:
- GT curve with a single point must be valid (interpolation-related edge case
  is handled in the metric, not here — Curve itself must simply accept it).
- Empty series label must be accepted, not rejected.
"""

import pytest

from real_chart_bench.domain.curve import Curve, ScaleType, curves_for_curve_scoring


def test_curve_holds_points_in_construction_order():
    curve = Curve(x_values=(1.0, 2.0, 3.0), y_values=(10.0, 20.0, 30.0))

    assert curve.x_values == (1.0, 2.0, 3.0)
    assert curve.y_values == (10.0, 20.0, 30.0)


def test_curve_defaults_to_empty_label_and_linear_scale():
    curve = Curve(x_values=(1.0,), y_values=(1.0,))

    assert curve.series_label == ""
    assert curve.x_scale is ScaleType.LINEAR


def test_curve_accepts_explicit_empty_label():
    # §3.3: 系列ラベルが空文字列/欠損
    curve = Curve(x_values=(1.0, 2.0), y_values=(1.0, 2.0), series_label="")

    assert curve.series_label == ""


def test_curve_with_single_point_is_valid():
    # §3.3: GT曲線が1点のみ
    curve = Curve(x_values=(5.0,), y_values=(42.0,))

    assert len(curve.x_values) == 1
    assert curve.x_min == curve.x_max == 5.0
    assert curve.y_min == curve.y_max == 42.0


def test_curve_rejects_empty_points():
    with pytest.raises(ValueError, match="at least one point"):
        Curve(x_values=(), y_values=())


def test_curve_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        Curve(x_values=(1.0, 2.0), y_values=(1.0,))


def test_curve_is_immutable():
    curve = Curve(x_values=(1.0,), y_values=(1.0,))

    with pytest.raises(AttributeError):
        curve.series_label = "mutated"  # type: ignore[misc]


def test_curve_sorts_points_by_x_for_deterministic_interpolation():
    curve = Curve(x_values=(3.0, 1.0, 2.0), y_values=(30.0, 10.0, 20.0))

    assert curve.x_values == (1.0, 2.0, 3.0)
    assert curve.y_values == (10.0, 20.0, 30.0)


def test_curve_x_min_max_and_y_min_max():
    curve = Curve(x_values=(3.0, 1.0, 2.0), y_values=(5.0, -1.0, 9.0))

    assert (curve.x_min, curve.x_max) == (1.0, 3.0)
    assert (curve.y_min, curve.y_max) == (-1.0, 9.0)


def test_curve_len_returns_point_count():
    curve = Curve(x_values=(1.0, 2.0, 3.0), y_values=(1.0, 2.0, 3.0))

    assert len(curve) == 3


# --- curves_for_curve_scoring: non-finite points (owner decision 2026-10-05) ---------


_INF = float("inf")


def test_all_finite_curves_pass_through_as_the_same_objects():
    a = Curve(x_values=(1.0, 2.0), y_values=(1.0, 2.0))
    b = Curve(x_values=(5.0,), y_values=(1.0,))

    out = curves_for_curve_scoring([a, b])

    assert out[0] is a and out[1] is b


def test_non_finite_points_are_removed_keeping_label_and_scale():
    c = Curve(
        x_values=(1.0, 10.0, _INF),
        y_values=(1.0, 2.0, -_INF),
        series_label="s",
        x_scale=ScaleType.LOG,
    )

    (out,) = curves_for_curve_scoring([c])

    assert out.x_values == (1.0, 10.0)
    assert out.y_values == (1.0, 2.0)
    assert out.series_label == "s"
    assert out.x_scale is ScaleType.LOG


def test_a_series_with_fewer_than_two_finite_points_left_is_dropped():
    one = Curve(x_values=(1.0, _INF), y_values=(1.0, 2.0))
    none = Curve(x_values=(float("nan"),), y_values=(1.0,))
    keep = Curve(x_values=(1.0, 2.0), y_values=(1.0, 2.0))

    assert curves_for_curve_scoring([one, none, keep]) == [keep]


def test_non_positive_values_are_finite_and_kept():
    c = Curve(x_values=(-1.0, 0.0, 1.0), y_values=(0.0, -2.0, 1.0), x_scale=ScaleType.LOG)

    assert curves_for_curve_scoring([c]) == [c]
