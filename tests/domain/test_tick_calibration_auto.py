"""方式C (docs/design/local-model.md): the pure helpers of the automatic axis
calibration -- splitting merged labels, numbering 10^n decades, and how far an automatic
axis fit is from the person's two-tick calibration (measured on the
benchmark, never tuned on it)."""

import math

import pytest

from real_chart_bench.domain.tick_calibration import (
    AxisFit,
    axis_agreement,
)


class TestAxisAgreement:
    PERSON = [{"px": 100.0, "value": 0.0}, {"px": 500.0, "value": 400.0}]

    def test_identical_line_is_zero_error(self):
        fit = AxisFit("linear", 1.0, 100.0, ((100, 0), (500, 400)), 0.0)
        assert axis_agreement(fit, self.PERSON, "linear") == pytest.approx(0.0, abs=1e-12)

    def test_error_is_fraction_of_the_person_tick_span(self):
        # auto line is shifted by 4 value units at every pixel: 4/400 = 1%
        fit = AxisFit("linear", 1.0, 96.0, ((96, 0), (496, 400)), 0.0)
        assert axis_agreement(fit, self.PERSON, "linear") == pytest.approx(0.01)

    def test_worst_of_the_two_ticks(self):
        # slope off: exact at px 100, 10% of the span off at px 500
        fit = AxisFit("linear", 440 / 400, 100.0, ((100, 0), (540, 400)), 0.0)
        assert axis_agreement(fit, self.PERSON, "linear") == pytest.approx(
            (400 - 400 / 1.1) / 400
        )

    def test_log_axis_compares_in_decades(self):
        person = [{"px": 0.0, "value": 1e-3}, {"px": 300.0, "value": 1.0}]
        fit = AxisFit("log", 100.0, 300.0, ((0, 1e-3), (300, 1.0)), 0.0)
        assert axis_agreement(fit, person, "log") == pytest.approx(0.0, abs=1e-12)
        # one decade off at both ticks out of a 3-decade span
        off = AxisFit("log", 100.0, 400.0, ((0, 1e-4), (300, 0.1)), 0.0)
        assert axis_agreement(off, person, "log") == pytest.approx(1 / 3)

    def test_scale_mismatch_is_infinite(self):
        fit = AxisFit("linear", 1.0, 100.0, ((100, 0), (500, 400)), 0.0)
        person = [{"px": 100.0, "value": 1.0}, {"px": 500.0, "value": 100.0}]
        assert axis_agreement(fit, person, "log") == math.inf

    def test_missing_fit_is_infinite(self):
        assert axis_agreement(None, self.PERSON, "linear") == math.inf


class TestSplitMergedLabels:
    def test_equal_width_run(self):
        from real_chart_bench.domain.tick_calibration import split_merged_labels

        assert split_merged_labels("300320340360") == ["300", "320", "340", "360"]

    def test_growing_width_run(self):
        from real_chart_bench.domain.tick_calibration import split_merged_labels

        assert split_merged_labels("708090100110") == ["70", "80", "90", "100", "110"]

    def test_decimals(self):
        from real_chart_bench.domain.tick_calibration import split_merged_labels

        assert split_merged_labels("0.00.51.01.5") == ["0.0", "0.5", "1.0", "1.5"]

    def test_negative_run(self):
        from real_chart_bench.domain.tick_calibration import split_merged_labels

        assert split_merged_labels("-20-1001020") == ["-20", "-10", "0", "10", "20"]

    def test_single_number_is_not_split(self):
        from real_chart_bench.domain.tick_calibration import split_merged_labels

        assert split_merged_labels("300") is None
        assert split_merged_labels("1000") is None

    def test_no_progression_is_not_split(self):
        from real_chart_bench.domain.tick_calibration import split_merged_labels

        assert split_merged_labels("317295") is None


class TestDecadeReadings:
    """Log-axis labels "10^n" whose exponents OCR mostly loses: equally
    spaced labels that start with "10" are consecutive decades, and the
    exponents that were read vote for the offset."""

    def _vals(self, rd):
        return [(px, [r.value for r in rs]) for px, rs in rd]

    def test_consecutive_decades_from_one_readable_exponent(self):
        from real_chart_bench.domain.tick_calibration import decade_readings

        # a y axis (direction -1: value grows upward = px shrinks)
        labels = [(400.0, "10"), (300.0, "10"), (200.0, "10^2"), (100.0, "10")]
        rd = decade_readings(labels, direction=-1)
        assert self._vals(rd) == [(100.0, [1e3]), (200.0, [1e2]), (300.0, [1e1]), (400.0, [1e0])]

    def test_majority_vote_beats_a_misread_exponent(self):
        from real_chart_bench.domain.tick_calibration import decade_readings

        labels = [(100.0, "101"), (250.0, "102"), (400.0, "109"), (550.0, "104")]
        rd = decade_readings(labels, direction=+1)
        assert [rs[0].value for _, rs in rd] == [1e1, 1e2, 1e3, 1e4]

    def test_negative_exponents(self):
        from real_chart_bench.domain.tick_calibration import decade_readings

        labels = [(500.0, "10-4"), (400.0, "10"), (300.0, "10"), (200.0, "10-1")]
        rd = decade_readings(labels, direction=-1)
        assert [rs[0].value for _, rs in sorted(rd, reverse=True)] == pytest.approx(
            [1e-4, 1e-3, 1e-2, 1e-1]
        )

    def test_a_missing_label_keeps_the_count(self):
        from real_chart_bench.domain.tick_calibration import decade_readings

        labels = [(100.0, "10^0"), (200.0, "10"), (400.0, "10")]
        rd = decade_readings(labels, direction=+1)
        assert [rs[0].value for _, rs in rd] == [1.0, 10.0, 1000.0]

    def test_no_vote_no_readings(self):
        from real_chart_bench.domain.tick_calibration import decade_readings

        assert decade_readings([(100.0, "10"), (200.0, "10"), (300.0, "10")], direction=1) == []

    def test_uneven_spacing_is_not_decades(self):
        from real_chart_bench.domain.tick_calibration import decade_readings

        labels = [(100.0, "10^1"), (130.0, "10"), (300.0, "10")]
        assert decade_readings(labels, direction=1) == []

    def test_too_few_labels(self):
        from real_chart_bench.domain.tick_calibration import decade_readings

        assert decade_readings([(100.0, "10^1"), (200.0, "10^2")], direction=1) == []


def test_decade_readings_recovers_lost_minus_signs():
    from real_chart_bench.domain.tick_calibration import decade_readings

    # y axis, 10^-3 at the bottom: the minus signs were not read
    labels = [(300.0, "103"), (200.0, "102"), (100.0, "101")]
    rd = decade_readings(labels, direction=-1)
    assert [rs[0].value for _, rs in sorted(rd, reverse=True)] == pytest.approx(
        [1e-3, 1e-2, 1e-1]
    )
