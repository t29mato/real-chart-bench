"""Ground truth in each figure's printed space (design 7.82): answers given
under the old reporting rules (10^tick on log10-printed axes, kelvin on a
degC-printed axis) are mapped back to what the axis prints before scoring."""

import math

from real_chart_bench.adapter.printed_space import (
    PRINTED_SPACE_MIGRATION,
    answer_to_printed,
    to_printed,
)


def test_log10_printed_axis_takes_the_log():
    assert to_printed(1e-6, "log10") == -6.0
    assert math.isclose(to_printed(10**-2.5, "log10"), -2.5)


def test_log10_of_a_non_positive_value_is_undefined():
    assert to_printed(0.0, "log10") is None
    assert to_printed(-1e-3, "log10") is None


def test_kelvin_back_to_celsius():
    assert math.isclose(to_printed(723.15, "k_to_degc"), 450.0)


def test_answer_is_converted_on_the_named_axis_only():
    answer = [{"label": "a", "x": [0.9, 1.1], "y": [1e-3, 1e-1]}]

    out = answer_to_printed(answer, {"y": "log10"})

    assert out == [{"label": "a", "x": [0.9, 1.1], "y": [-3.0, -1.0]}]


def test_an_undefined_point_stays_as_a_point_that_can_never_match():
    # the old scoring counted a non-positive value on a log axis as a predicted
    # point that matches nothing (NaN in point_metrics); dropping it instead
    # would raise precision, so it is kept as NaN
    out = answer_to_printed([{"label": "a", "x": [1.0, 2.0], "y": [0.0, 10.0]}], {"y": "log10"})

    assert out[0]["x"] == [1.0, 2.0]
    assert math.isnan(out[0]["y"][0]) and out[0]["y"][1] == 1.0


def test_non_numeric_entries_and_non_dict_series_pass_through_for_the_parser():
    answer = [{"label": "a", "x": [450.0, None], "y": [1, 2]}, "junk"]

    out = answer_to_printed(answer, {"x": "k_to_degc"})

    assert out[0]["y"] == [1, 2] and out[0]["x"][1] is None
    assert math.isclose(out[0]["x"][0], 176.85)
    assert out[1] == "junk"


def test_migration_covers_the_thirteen_log_axes_and_the_one_celsius_axis():
    ops = [op for axes in PRINTED_SPACE_MIGRATION.values() for op in axes.values()]
    assert ops.count("log10") == 13 and ops.count("k_to_degc") == 1
    assert PRINTED_SPACE_MIGRATION["20121"] == {"x": "k_to_degc"}


# --- answers whose space is not known per run (design 7.82 addendum) ---------


def test_an_answer_in_the_old_10_pow_space_is_recognised_and_converted():
    from real_chart_bench.adapter.printed_space import answer_to_printed_if_old

    # printed y range is log10 values -6..-1; the answer gave 10^tick
    answer = [{"x": [1.0, 1.5], "y": [1e-5, 1e-2]}]

    out, was_old = answer_to_printed_if_old(answer, {"y": "log10"}, {"y": (-6.0, -1.0)})

    assert was_old is True and out[0]["y"] == [-5.0, -2.0]


def test_an_answer_already_printed_is_left_alone():
    from real_chart_bench.adapter.printed_space import answer_to_printed_if_old

    answer = [{"x": [1.0, 1.5], "y": [-5.0, -2.0]}]

    out, was_old = answer_to_printed_if_old(answer, {"y": "log10"}, {"y": (-6.0, -1.0)})

    assert was_old is False and out == answer


def test_a_kelvin_answer_on_a_celsius_axis_is_recognised():
    from real_chart_bench.adapter.printed_space import answer_to_printed_if_old

    answer = [{"x": [723.15, 1123.15], "y": [1, 2]}]

    out, was_old = answer_to_printed_if_old(answer, {"x": "k_to_degc"}, {"x": (450.0, 850.0)})

    assert was_old is True and abs(out[0]["x"][0] - 450.0) < 1e-9
