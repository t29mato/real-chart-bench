"""Value -> pixel along a label's axis ticks, and the self-check every
training label passes: its value points, projected through its own ticks,
land on its pixel points (docs/design/local-model.md, データ: 合成)."""

import pytest

from real_chart_bench.domain.training_data import axis_value_to_px, max_axis_residual_px

LIN = {"scale": "linear", "ticks": [{"px": 100.0, "value": 0}, {"px": 700.0, "value": 10}]}
# y grows upward: the larger value has the smaller pixel row
LOG_Y = {"scale": "log", "ticks": [{"px": 500.0, "value": 1e-3}, {"px": 50.0, "value": 1}]}


def test_linear_axis_interpolates_and_extrapolates():
    assert axis_value_to_px(5, LIN) == pytest.approx(400.0)
    assert axis_value_to_px(-1, LIN) == pytest.approx(40.0)


def test_log_axis_is_linear_in_decades():
    assert axis_value_to_px(1e-3, LOG_Y) == pytest.approx(500.0)
    assert axis_value_to_px(1e-2, LOG_Y) == pytest.approx(350.0)
    assert axis_value_to_px(10, LOG_Y) == pytest.approx(-100.0)


def test_many_ticks_are_fitted_by_least_squares():
    # a tick box one pixel off does not move the axis much
    axis = {
        "scale": "linear",
        "ticks": [
            {"px": 100.0, "value": 0},
            {"px": 201.0, "value": 1},
            {"px": 300.0, "value": 2},
            {"px": 400.0, "value": 3},
        ],
    }
    assert axis_value_to_px(1.5, axis) == pytest.approx(250.0, abs=0.5)


def test_an_axis_without_two_distinct_tick_values_is_refused():
    axis = {"scale": "linear", "ticks": [{"px": 1.0, "value": 2}, {"px": 9.0, "value": 2}]}
    with pytest.raises(ValueError, match="distinct"):
        axis_value_to_px(1, axis)


def test_log_axis_refuses_non_positive_values():
    with pytest.raises(ValueError, match="log"):
        axis_value_to_px(0, LOG_Y)


def _label(points_px, points_value, axes=None):
    return {
        "axes": {"x": LIN, "y": LOG_Y} if axes is None else axes,
        "series": [{"points_px": points_px, "points_value": points_value}],
    }


def test_a_consistent_label_has_zero_residual():
    label = _label([[400.0, 350.0], [100.0, 500.0]], [[5, 1e-2], [0, 1e-3]])
    assert max_axis_residual_px(label) == pytest.approx(0.0, abs=1e-9)


def test_residual_is_the_worst_pixel_distance_on_either_axis():
    label = _label([[403.0, 350.0], [100.0, 507.0]], [[5, 1e-2], [0, 1e-3]])
    assert max_axis_residual_px(label) == pytest.approx(7.0)


def test_residual_is_unknown_without_axes_or_values():
    assert max_axis_residual_px(_label([[1.0, 1.0]], [[1, 1]]) | {"axes": None}) is None
    assert max_axis_residual_px(_label([[1.0, 1.0]], None)) is None


def test_series_without_values_are_skipped_but_others_count():
    label = _label([[400.0, 352.0]], [[5, 1e-2]])
    label["series"].append({"points_px": [[1.0, 1.0]], "points_value": None})
    assert max_axis_residual_px(label) == pytest.approx(2.0)
