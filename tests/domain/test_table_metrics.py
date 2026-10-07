"""Chart-as-table metrics (design §7.85): RMS (DePlot, Liu et al. 2023) as a
secondary reported metric, its header-neutralized variant, and NMS (the
axis-range normalization PaperUnPlot uses instead of |g|).

Boundary cases are the list in §7.85 "TDD の境界ケース", written before the
implementation: empty prediction, empty ground truth, log axes, |g| = 0,
duplicate x values, missing series labels, permutation invariance.
"""

import math

import pytest

from real_chart_bench.domain.curve import Curve, ScaleType
from real_chart_bench.domain.point_metrics import AxisFrame
from real_chart_bench.domain.table_metrics import (
    NUMBER_THETA,
    VALUE_NORM_AXIS_RANGE,
    VALUE_NORM_RELATIVE,
    TableEntry,
    anls,
    entries_from_curves,
    evaluate_chart_table,
    evaluate_table,
    format_cell,
)

# y spans 100 units, so an absolute error of 1 is 1% of the axis range
FRAME = AxisFrame(x_range=(0.0, 10.0), y_range=(0.0, 100.0))
LOG_FRAME = AxisFrame(x_range=(1.0, 10.0), y_range=(1.0, 1000.0), y_scale=ScaleType.LOG)


def _curve(points, series_label=""):
    return Curve(
        x_values=tuple(float(p[0]) for p in points),
        y_values=tuple(float(p[1]) for p in points),
        series_label=series_label,
    )


def _entry(x, label, value):
    return TableEntry(row_header=format_cell(x), column_header=label, value=value)


def _rms(predicted, ground_truth, **kwargs):
    return evaluate_table(predicted, ground_truth, **kwargs)


def _nms(predicted, ground_truth, frame=FRAME, **kwargs):
    return evaluate_table(
        predicted,
        ground_truth,
        header_term=False,
        value_norm=VALUE_NORM_AXIS_RANGE,
        frame=frame,
        **kwargs,
    )


# --- the table view of a chart -------------------------------------------------


def test_each_point_becomes_one_row_header_column_header_value_triple():
    curves = [_curve([(300, 1.5), (400, 2.5)], "HP680")]

    entries = entries_from_curves(curves)

    assert [(e.row_header, e.column_header, e.value) for e in entries] == [
        ("300", "hp680", 1.5),
        ("400", "hp680", 2.5),
    ]
    # DePlot's key is "<row header> <column header>", one string
    assert entries[0].key == "300 hp680"


def test_whole_numbers_print_without_a_decimal_point_and_headers_are_lowercased():
    assert format_cell(300.0) == "300"
    assert format_cell(320.02) == "320.02"
    assert format_cell(-0.5) == "-0.5"
    assert entries_from_curves([_curve([(1, 1)], "SPS760")])[0].column_header == "sps760"


def test_a_non_finite_x_still_becomes_an_entry_with_a_printable_header():
    entries = entries_from_curves([_curve([(math.inf, 1.0)], "a")])

    assert entries[0].row_header == "inf"


# --- a perfect answer ----------------------------------------------------------


def test_a_perfect_answer_scores_one_in_all_three_variants():
    gt = [_curve([(300, 1.5), (400, 2.5)], "HP680"), _curve([(300, 3.5)], "SPS760")]

    metrics = evaluate_chart_table(gt, gt, FRAME)

    assert (metrics.rms.precision, metrics.rms.recall, metrics.rms.f1) == (1.0, 1.0, 1.0)
    assert metrics.rms_value_only.f1 == 1.0
    assert metrics.nms.f1 == 1.0
    assert (metrics.rms.n_predicted, metrics.rms.n_ground_truth) == (3, 3)


# --- empty sides ---------------------------------------------------------------


def test_an_empty_prediction_has_no_recall_and_an_f1_of_zero():
    gt = entries_from_curves([_curve([(300, 1.5)], "a")])

    result = _rms([], gt)

    # DePlot: nothing was claimed, so precision is vacuously 1
    assert (result.precision, result.recall, result.f1) == (1.0, 0.0, 0.0)
    assert (result.n_predicted, result.n_ground_truth) == (0, 1)


def test_an_empty_ground_truth_has_no_precision_and_an_f1_of_zero():
    pred = entries_from_curves([_curve([(300, 1.5)], "a")])

    result = _rms(pred, [])

    assert (result.precision, result.recall, result.f1) == (0.0, 1.0, 0.0)


def test_two_empty_tables_match_perfectly():
    result = _rms([], [])

    assert (result.precision, result.recall, result.f1) == (1.0, 1.0, 1.0)


def test_empty_sides_behave_the_same_in_every_variant():
    gt = entries_from_curves([_curve([(300, 1.5)], "a")])

    assert _rms([], gt, header_term=False).f1 == 0.0
    assert _nms([], gt).f1 == 0.0
    assert _nms([], []).f1 == 1.0


def test_nothing_matching_at_all_scores_zero_on_every_axis():
    gt = [_entry(300, "a", 1.0)]
    pred = [_entry(999, "zzzz", 500.0)]

    result = _rms(pred, gt)

    assert (result.precision, result.recall, result.f1) == (0.0, 0.0, 0.0)
    assert result.score == 0.0


# --- the value term: RMS's relative error --------------------------------------


def test_the_value_score_is_one_minus_the_relative_error_below_the_threshold():
    gt = [_entry(300, "a", 10.0)]
    pred = [_entry(300, "a", 10.5)]  # 5% relative error

    result = _rms(pred, gt)

    assert result.f1 == pytest.approx(0.95)


def test_a_relative_error_at_or_above_the_threshold_scores_zero():
    gt = [_entry(300, "a", 10.0)]

    assert _rms([_entry(300, "a", 11.0)], gt).f1 == 0.0  # exactly theta
    assert _rms([_entry(300, "a", 12.0)], gt).f1 == 0.0
    assert NUMBER_THETA == 0.1


def test_a_non_finite_predicted_value_never_scores():
    gt = [_entry(300, "a", 10.0)]

    assert _rms([_entry(300, "a", math.nan)], gt).f1 == 0.0
    assert _rms([_entry(300, "a", math.inf)], gt).f1 == 0.0
    assert _nms([_entry(300, "a", math.inf)], gt).f1 == 0.0


# --- the value term: a ground-truth value of exactly 0 -------------------------


def test_a_ground_truth_value_of_zero_is_hit_only_by_exactly_zero():
    gt = [_entry(300, "a", 0.0)]

    assert _rms([_entry(300, "a", 0.0)], gt).f1 == 1.0
    assert _rms([_entry(300, "a", 0.001)], gt).f1 == 0.0


def test_the_axis_range_variant_grades_a_ground_truth_value_of_zero():
    gt = [_entry(300, "a", 0.0)]

    # 1 unit off a 100-unit axis is 1%, which RMS cannot express at all
    assert _nms([_entry(300, "a", 1.0)], gt).f1 == pytest.approx(0.99)


# --- the value term: NMS's axis-range error ------------------------------------


def test_the_axis_range_variant_divides_by_the_axis_range_not_by_the_true_value():
    gt = [_entry(300, "a", 1.0)]
    pred = [_entry(300, "a", 1.2)]  # 20% of the value, 0.2% of the axis

    assert _rms(pred, gt).f1 == 0.0
    assert _nms(pred, gt).f1 == pytest.approx(0.998)


def test_the_axis_range_variant_keeps_the_same_threshold_so_only_the_denominator_changes():
    gt = [_entry(300, "a", 50.0)]
    # 10 units is 10% of the axis range -- exactly theta, so it scores zero
    assert _nms([_entry(300, "a", 60.0)], gt).f1 == 0.0
    assert _nms([_entry(300, "a", 59.0)], gt).f1 == pytest.approx(0.91)


def test_the_axis_range_variant_compares_in_log10_space_on_a_log_axis():
    gt = [_entry(1, "a", 10.0)]

    # one decade off a three-decade axis
    assert _nms([_entry(1, "a", 100.0)], gt, frame=LOG_FRAME).f1 == 0.0
    # log10(11) - log10(10) = 0.0414, over 3 decades
    expected = 1.0 - abs(math.log10(11.0) - math.log10(10.0)) / 3.0
    assert _nms([_entry(1, "a", 11.0)], gt, frame=LOG_FRAME).f1 == pytest.approx(expected)


def test_a_non_positive_value_cannot_be_placed_on_a_log_axis_and_scores_zero():
    gt = [_entry(1, "a", 10.0)]

    assert _nms([_entry(1, "a", 0.0)], gt, frame=LOG_FRAME).f1 == 0.0
    assert _nms([_entry(1, "a", -10.0)], gt, frame=LOG_FRAME).f1 == 0.0


def test_the_axis_range_variant_needs_a_frame():
    with pytest.raises(ValueError, match="frame"):
        evaluate_table([], [], value_norm=VALUE_NORM_AXIS_RANGE)


def test_an_unknown_value_norm_is_rejected():
    with pytest.raises(ValueError, match="value_norm"):
        evaluate_table([], [], value_norm="per-series-span")


# --- the header term -----------------------------------------------------------


def test_a_close_legend_string_costs_part_of_the_score_in_faithful_rms():
    gt = [_entry(300, "hp680", 10.0)]
    pred = [_entry(300, "hp681", 10.0)]  # one character off out of 9

    result = _rms(pred, gt)

    assert result.f1 == pytest.approx(anls("300 hp680", "300 hp681"))
    assert result.f1 == pytest.approx(1.0 - 1.0 / 9.0)


def test_a_legend_string_beyond_the_text_threshold_wipes_the_entry_out():
    gt = [_entry(300, "hp680", 10.0)]
    pred = [_entry(300, "sample a from ref 12", 10.0)]

    # the database's name and the printed legend need not agree (§7.85): the
    # faithful metric scores this a total miss even though the value is exact
    assert _rms(pred, gt).f1 == 0.0
    # which is exactly what the value-only variant exists to remove
    assert _rms(pred, gt, header_term=False).f1 == 1.0
    assert _nms(pred, gt).f1 == 1.0


def test_the_value_only_variant_still_multiplies_in_the_value_score():
    gt = [_entry(300, "hp680", 10.0)]
    pred = [_entry(300, "something else entirely", 10.5)]

    assert _rms(pred, gt, header_term=False).f1 == pytest.approx(0.95)


def test_anls_is_one_minus_the_normalized_edit_distance_thresholded():
    assert anls("abc", "abc") == 1.0
    assert anls("abcd", "abcx") == pytest.approx(0.75)
    assert anls("ab", "xy") == 0.0  # normalized distance 1.0 >= theta
    assert anls("", "") == 1.0


# --- missing series labels -----------------------------------------------------


def test_a_missing_series_label_falls_back_to_an_empty_column_header():
    gt = entries_from_curves([_curve([(300, 10.0)], series_label="")])
    pred = entries_from_curves([_curve([(300, 10.0)], series_label="")])

    assert gt[0].key == "300 "
    # both sides unlabeled: the header reduces to the x value and still matches
    assert _rms(pred, gt).f1 == 1.0


def test_an_unlabeled_prediction_against_a_labeled_ground_truth_still_scores_on_value():
    gt = entries_from_curves([_curve([(300, 10.0)], series_label="HP680")])
    pred = entries_from_curves([_curve([(300, 10.0)], series_label="")])

    assert _rms(pred, gt).f1 == 0.0  # "300 hp680" vs "300 " is beyond theta
    assert _rms(pred, gt, header_term=False).f1 == 1.0
    assert _nms(pred, gt).f1 == 1.0


def test_a_none_series_label_is_treated_as_missing_and_does_not_raise():
    curve = Curve(x_values=(300.0,), y_values=(10.0,), series_label=None)

    assert entries_from_curves([curve])[0].column_header == ""


# --- duplicate x values --------------------------------------------------------


def test_duplicate_x_values_in_one_series_stay_two_entries():
    gt = entries_from_curves([_curve([(300, 10.0), (300, 20.0)], "a")])

    assert len(gt) == 2
    assert gt[0].key == gt[1].key


def test_duplicate_headers_are_paired_by_value_so_the_order_cannot_change_the_score():
    gt = [_entry(300, "a", 10.0), _entry(300, "a", 20.0)]
    forward = [_entry(300, "a", 10.0), _entry(300, "a", 20.0)]
    reversed_ = [_entry(300, "a", 20.0), _entry(300, "a", 10.0)]

    assert _rms(forward, gt).f1 == 1.0
    assert _rms(reversed_, gt).f1 == 1.0
    assert _nms(reversed_, gt).f1 == 1.0


# --- permutation invariance ----------------------------------------------------


def _shuffled_figure():
    a = _curve([(300, 1.0), (400, 2.0), (500, 3.0)], "HP680")
    b = _curve([(300, 4.0), (400, 5.0)], "SPS760")
    pred_a = _curve([(300, 1.02), (400, 2.01), (500, 3.1)], "HP680")
    pred_b = _curve([(300, 4.05), (400, 5.4)], "SPS760")
    return [a, b], [pred_a, pred_b]


def test_the_score_does_not_depend_on_the_order_of_the_series():
    gt, pred = _shuffled_figure()

    straight = evaluate_chart_table(pred, gt, FRAME)
    swapped = evaluate_chart_table(pred[::-1], gt[::-1], FRAME)

    assert swapped.rms.f1 == pytest.approx(straight.rms.f1)
    assert swapped.rms_value_only.f1 == pytest.approx(straight.rms_value_only.f1)
    assert swapped.nms.f1 == pytest.approx(straight.nms.f1)


def test_the_score_does_not_depend_on_the_order_of_the_entries():
    gt, pred = _shuffled_figure()
    gt_entries, pred_entries = entries_from_curves(gt), entries_from_curves(pred)

    straight = _rms(pred_entries, gt_entries)
    jumbled = _rms(pred_entries[::-1], gt_entries[::-1])

    assert jumbled.f1 == pytest.approx(straight.f1)
    assert jumbled.score == pytest.approx(straight.score)


def test_shuffling_only_one_side_leaves_the_score_alone():
    gt, pred = _shuffled_figure()
    gt_entries, pred_entries = entries_from_curves(gt), entries_from_curves(pred)

    straight = _rms(pred_entries, gt_entries)
    one_side = _rms(pred_entries[::-1], gt_entries)

    assert one_side.f1 == pytest.approx(straight.f1)


# --- precision and recall are counted over entries, not curves -----------------


def test_a_missed_series_lowers_recall_while_precision_stays_perfect():
    gt = [_curve([(300, 1.0), (400, 2.0)], "a"), _curve([(300, 3.0)], "b")]
    pred = [_curve([(300, 1.0), (400, 2.0)], "a")]

    result = evaluate_chart_table(pred, gt, FRAME).rms

    assert result.n_predicted == 2
    assert result.n_ground_truth == 3
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(2.0 / 3.0)
    assert result.f1 == pytest.approx(2 * 1.0 * (2 / 3) / (1.0 + 2 / 3))


def test_an_invented_series_lowers_precision_while_recall_stays_perfect():
    gt = [_curve([(300, 1.0)], "a")]
    pred = [_curve([(300, 1.0)], "a"), _curve([(900, 9.0)], "ghost")]

    result = evaluate_chart_table(pred, gt, FRAME).rms

    assert result.recall == pytest.approx(1.0)
    assert result.precision == pytest.approx(0.5)


# --- the three reported variants -----------------------------------------------


def test_the_three_variants_differ_only_in_the_header_term_and_the_denominator():
    gt = [_curve([(300, 1.0), (400, 2.0)], "HP680")]
    pred = [_curve([(300, 1.2), (400, 2.0)], "printed legend text")]

    metrics = evaluate_chart_table(pred, gt, FRAME)

    # faithful RMS: the legend text is beyond theta, so both entries die
    assert metrics.rms.f1 == 0.0
    # value-only: the 20%-of-value error is still beyond RMS's number theta
    assert metrics.rms_value_only.f1 == pytest.approx(0.5)
    # NMS: 0.2 units off a 100-unit axis is 0.2%
    assert metrics.nms.f1 == pytest.approx((0.998 + 1.0) / 2.0)


def test_the_default_value_norm_is_the_relative_one_rms_publishes():
    gt = [_entry(300, "a", 1.0)]
    pred = [_entry(300, "a", 1.2)]

    assert _rms(pred, gt).f1 == _rms(pred, gt, value_norm=VALUE_NORM_RELATIVE).f1
