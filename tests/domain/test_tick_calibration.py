"""Axis calibration from locally OCR'd tick labels, value->pixel projection of
Starrydata ground truth, and the self-consistency gate that decides whether a
real figure becomes a training label (docs/design/local-model.md, "データ:
実図(Starrydata)"). No model reads a value here: OCR text in, numbers out."""

import math

import numpy as np
import pytest

from real_chart_bench.domain.tick_calibration import (
    AxisFit,
    Transform,
    candidate_transforms,
    fit_axis,
    ink_contrast,
    inside_fraction,
    parse_tick_label,
)


def _values(readings):
    return sorted((r.kind, r.value) for r in readings)


class TestParseTickLabel:
    @pytest.mark.parametrize(
        "text,value",
        [("300", 300), ("-0.5", -0.5), ("1.25", 1.25), (".5", 0.5), ("0", 0),
         ("−40", -40), ("–2", -2), ("1e-3", 1e-3), ("2E4", 2e4), ("50.", 50)],
    )
    def test_plain_numbers(self, text, value):
        (r,) = parse_tick_label(text)
        assert r.kind == "plain" and r.value == pytest.approx(value)

    @pytest.mark.parametrize("text", ["", "a", "Fig", "1.2.3", "--3", "T(K)", "12a", "1,5"])
    def test_not_a_number(self, text):
        assert parse_tick_label(text) == []

    def test_explicit_power_of_ten(self):
        assert _values(parse_tick_label("10^-3")) == [("pow10", 1e-3)]
        assert _values(parse_tick_label("10⁻³")) == [("pow10", 1e-3)]
        assert _values(parse_tick_label("10²")) == [("pow10", 100)]

    def test_superscript_lost_by_ocr_is_ambiguous(self):
        # "10" with a raised exponent comes out of OCR as "103": either the
        # plain number or 10^3. The axis fit decides, never this parser.
        assert _values(parse_tick_label("103")) == [("plain", 103), ("pow10", 1e3)]
        assert _values(parse_tick_label("10-3")) == [("pow10", 1e-3)]
        assert _values(parse_tick_label("10")) == [("plain", 10)]

    def test_exponent_is_bounded(self):
        assert _values(parse_tick_label("10123")) == [("plain", 10123)]

    def test_whitespace_and_trailing_garbage_dots(self):
        assert _values(parse_tick_label(" 200 ")) == [("plain", 200)]


def _r(px, text):
    return (px, parse_tick_label(text))


class TestFitAxis:
    def test_linear_x_axis(self):
        fit = fit_axis([_r(100, "300"), _r(200, "400"), _r(300, "500"), _r(400, "600")],
                       direction=+1)
        assert fit.scale == "linear"
        assert fit.value_to_px(450) == pytest.approx(250)
        assert fit.px_to_value(150) == pytest.approx(350)
        assert len(fit.ticks) == 4 and fit.residual_px == pytest.approx(0, abs=1e-9)

    def test_y_axis_values_grow_upward(self):
        fit = fit_axis([_r(500, "0"), _r(400, "1"), _r(300, "2")], direction=-1)
        assert fit.value_to_px(1.5) == pytest.approx(350)

    def test_wrong_direction_is_rejected(self):
        # values that grow downward on a y axis: an OCR misread, not a chart
        assert fit_axis([_r(300, "0"), _r(400, "1"), _r(500, "2")], direction=-1) is None

    def test_an_ocr_misread_tick_is_dropped_as_outlier(self):
        fit = fit_axis(
            [_r(100, "300"), _r(200, "400"), _r(300, "800"), _r(400, "600"), _r(500, "700")],
            direction=+1,
        )
        assert [t[1] for t in fit.ticks] == [300, 400, 600, 700]

    def test_needs_three_consistent_ticks(self):
        assert fit_axis([_r(100, "300"), _r(200, "400")], direction=+1) is None
        assert fit_axis([_r(100, "1"), _r(200, "5"), _r(300, "2")], direction=+1) is None

    def test_log_axis_with_plain_decade_labels(self):
        fit = fit_axis([_r(500, "0.01"), _r(400, "0.1"), _r(300, "1"), _r(200, "10")],
                       direction=-1)
        assert fit.scale == "log"
        assert fit.value_to_px(math.sqrt(10)) == pytest.approx(250)

    def test_lost_superscripts_read_as_powers_of_ten(self):
        fit = fit_axis([_r(500, "102"), _r(400, "103"), _r(300, "104")], direction=-1)
        assert fit.scale == "log"
        assert sorted(t[1] for t in fit.ticks) == [1e2, 1e3, 1e4]

    def test_log10_printed_labels_are_a_linear_axis(self):
        fit = fit_axis([_r(500, "-3"), _r(400, "-2"), _r(300, "-1"), _r(200, "0")],
                       direction=-1)
        assert fit.scale == "linear"

    def test_duplicate_labels_do_not_make_a_fit(self):
        assert fit_axis([_r(100, "5"), _r(200, "5"), _r(300, "5")], direction=+1) is None

    def test_tolerance_is_a_fraction_of_the_tick_span(self):
        # 3 px off on a 300 px span is within tolerance; 30 px is not
        ok = fit_axis([_r(100, "0"), _r(203, "1"), _r(300, "2"), _r(400, "3")], direction=+1)
        assert ok is not None and len(ok.ticks) == 4
        bad = fit_axis([_r(100, "0"), _r(230, "1"), _r(300, "2"), _r(400, "3")], direction=+1)
        assert bad is not None and len(bad.ticks) == 3


class TestAxisFitProjection:
    def test_log_projection_rejects_non_positive(self):
        fit = AxisFit(scale="log", slope=-100.0, intercept=300.0,
                      ticks=((300.0, 1.0), (200.0, 10.0)), residual_px=0.0)
        assert fit.value_to_px(0) is None
        assert fit.value_to_px(100) == pytest.approx(100)


class TestTransforms:
    def test_power_of_ten_unit_changes(self):
        names = {t.name for t in candidate_transforms("Seebeck coefficient", "V*K^(-1)")}
        assert "x1e6" in names and "x1e-3" in names and "x1e0" in names

    def test_kelvin_to_celsius_only_for_kelvin(self):
        assert any(t.name == "K->degC" for t in candidate_transforms("Temperature", "K"))
        assert not any(t.name == "K->degC" for t in candidate_transforms("ZT", "-"))

    def test_reciprocal_temperature_for_arrhenius_axes(self):
        t = next(t for t in candidate_transforms("Temperature", "K") if t.name == "1e3/v")
        assert t.apply(500.0) == pytest.approx(2.0)

    def test_log_printed_axis(self):
        t = next(t for t in candidate_transforms("Electrical conductivity", "S/m")
                 if t.name == "log10(v)+-2")
        assert t.apply(1e4) == pytest.approx(2.0)
        assert t.apply(-1.0) is None

    def test_already_logged_quantity_shifts_by_whole_decades(self):
        ts = {t.name: t for t in candidate_transforms("log(Electrical conductivity)", "S*m^(-1)")}
        assert ts["v+-2"].apply(3.0) == pytest.approx(1.0)
        assert not any(n.startswith("log10") for n in ts)

    def test_transform_is_a_plain_value_object(self):
        t = Transform("scale", 3)
        assert t.name == "x1e3" and t.apply(2.0) == pytest.approx(2000.0)


class TestGate:
    def test_inside_fraction(self):
        pts = [(10, 10), (50, 50), (200, 50)]
        assert inside_fraction(pts, (0, 0, 100, 100), margin_px=0) == pytest.approx(2 / 3)
        assert inside_fraction(pts, (0, 0, 100, 100), margin_px=100) == 1.0
        assert inside_fraction([], (0, 0, 100, 100), margin_px=0) == 0.0

    def test_ink_contrast_high_when_points_sit_on_markers(self):
        ink = np.zeros((200, 200), dtype=bool)
        pts = [(40 + 30 * i, 150 - 20 * i) for i in range(5)]
        for x, y in pts:
            ink[y - 3 : y + 4, x - 3 : x + 4] = True
        hit, null = ink_contrast(pts, ink, radius=2)
        assert hit == 1.0 and null < 0.2

    def test_ink_contrast_low_when_points_miss(self):
        ink = np.zeros((200, 200), dtype=bool)
        pts = [(40 + 30 * i, 150 - 20 * i) for i in range(5)]
        for x, y in pts:
            ink[y + 12 : y + 18, x + 12 : x + 18] = True  # markers drawn elsewhere
        hit, _ = ink_contrast(pts, ink, radius=2)
        assert hit == 0.0

    def test_ink_contrast_sees_through_uniform_ink(self):
        # a plot area full of ink (hatching, an image) explains nothing
        ink = np.ones((200, 200), dtype=bool)
        hit, null = ink_contrast([(50, 50), (100, 100)], ink, radius=2)
        assert hit == 1.0 and null == 1.0

    def test_points_off_image_count_as_misses(self):
        ink = np.ones((50, 50), dtype=bool)
        hit, _ = ink_contrast([(-10, 5), (500, 5)], ink, radius=2)
        assert hit == 0.0


class TestReadingsForAxis:
    FRAME = (100.0, 50.0, 500.0, 350.0)

    def test_x_labels_below_the_axis_snap_to_ticks(self):
        from real_chart_bench.domain.tick_calibration import readings_for_axis

        words = [("300", 92, 358, 110, 368), ("400", 193, 358, 211, 368),
                 ("T(K)", 280, 380, 320, 392), ("999", 300, 100, 318, 110)]
        r = readings_for_axis(words, self.FRAME, "x", ticks=[100.0, 200.0, 300.0])
        assert [(px, [x.value for x in rs]) for px, rs in r] == [(100.0, [300]), (200.0, [400])]

    def test_unsnapped_label_keeps_its_centre(self):
        from real_chart_bench.domain.tick_calibration import readings_for_axis

        r = readings_for_axis([("5", 246, 358, 252, 368)], self.FRAME, "x", ticks=[])
        assert r[0][0] == pytest.approx(249)

    def test_y_labels_left_of_the_axis_use_their_vertical_centre(self):
        from real_chart_bench.domain.tick_calibration import readings_for_axis

        words = [("0", 85, 344, 93, 356), ("10", 80, 245, 93, 255), ("12", 10, 145, 20, 155)]
        r = readings_for_axis(words, self.FRAME, "y", ticks=[])
        # "12" sits far left in the axis-title column, not the label column
        assert [(px, rs[0].value) for px, rs in r] == [(250.0, 10), (350.0, 0)]

    def test_split_base_and_exponent_are_joined(self):
        from real_chart_bench.domain.tick_calibration import readings_for_axis

        words = [("10", 70, 246, 84, 256), ("-3", 85, 241, 93, 249)]
        r = readings_for_axis(words, self.FRAME, "y", ticks=[])
        assert [x.value for x in r[0][1]] == [1e-3]


class TestBestProjection:
    """Stored SI values -> printed values -> pixels, choosing the unit
    transform per axis by frame containment and coverage, then by ink."""

    def _fits(self):
        # x: 300..800 K printed as K at px 100..600; y: 0..300 uV/K at px 400..100
        x = AxisFit("linear", 1.0, -200.0, ((100.0, 300.0), (600.0, 800.0)), 0.0)
        y = AxisFit("linear", -1.0, 400.0, ((400.0, 0.0), (100.0, 300.0)), 0.0)
        return x, y

    def test_finds_microvolts_and_lands_on_markers(self):
        from real_chart_bench.domain.tick_calibration import best_projection

        x, y = self._fits()
        xs = [350.0, 450.0, 550.0, 650.0, 750.0]
        ys = [50e-6, 100e-6, 150e-6, 200e-6, 250e-6]
        ink = np.zeros((500, 700), dtype=bool)
        for xv, yv in zip(xs, ys, strict=True):
            px, py = int(xv - 200), int(400 - yv * 1e6)
            ink[py - 3 : py + 4, px - 3 : px + 4] = True
        best = best_projection(
            [(xs, ys)], x, y, (100.0, 100.0, 600.0, 400.0), ink,
            candidate_transforms("Temperature", "K"),
            candidate_transforms("Seebeck coefficient", "V*K^(-1)"),
        )
        assert best.x_transform.name == "x1e0" and best.y_transform.name == "x1e6"
        assert best.hit == 1.0 and best.null < 0.2
        assert best.points_px[0][0] == pytest.approx((150.0, 350.0))
        assert best.points_value[0][0] == pytest.approx((350.0, 50.0))

    def test_none_when_no_transform_fits_the_frame(self):
        from real_chart_bench.domain.tick_calibration import best_projection

        x, y = self._fits()
        best = best_projection(
            [([5000.0, 9000.0], [1.0, 2.0])], x, y, (100.0, 100.0, 600.0, 400.0),
            np.zeros((500, 700), dtype=bool),
            [Transform("scale", 0)], [Transform("scale", 0)],
        )
        assert best is None

    def test_tiny_coverage_is_rejected(self):
        from real_chart_bench.domain.tick_calibration import best_projection

        x, y = self._fits()
        # all GT x within 2 K: a frame this wide is not where it was read from
        best = best_projection(
            [([500.0, 501.0], [100e-6, 120e-6])], x, y, (100.0, 100.0, 600.0, 400.0),
            np.zeros((500, 700), dtype=bool),
            [Transform("scale", 0)], [Transform("scale", 6)],
        )
        assert best is None


class TestInkRadius:
    def test_scales_with_the_frame_with_a_floor(self):
        from real_chart_bench.domain.tick_calibration import ink_radius, null_shift

        assert ink_radius((0, 0, 100, 100)) == 2
        assert ink_radius((0, 0, 1000, 600)) == 6
        assert null_shift((0, 0, 1000, 600), 6) == 28
        assert null_shift((0, 0, 100, 100), 2) == 12


class TestFrameLinesAreNotMarkers:
    def test_points_on_the_axis_line_do_not_hit(self):
        from real_chart_bench.domain.tick_calibration import without_frame_lines

        ink = np.zeros((200, 300), dtype=bool)
        ink[150, 50:251] = True  # x axis
        ink[20:151, 50] = True  # y axis
        ink[80:85, 120:125] = True  # a marker
        clean = without_frame_lines(ink, (50, 20, 250, 150), band=4)
        hit, _ = ink_contrast([(100, 150), (200, 151), (50, 90)], clean, radius=2)
        assert hit == 0.0
        assert clean[80:85, 120:125].all()

    def test_best_projection_ignores_points_squashed_onto_the_axis(self):
        from real_chart_bench.domain.tick_calibration import best_projection

        x = AxisFit("linear", 1.0, -200.0, ((100.0, 300.0), (600.0, 800.0)), 0.0)
        y = AxisFit("linear", -1.0, 400.0, ((400.0, 0.0), (100.0, 300.0)), 0.0)
        ink = np.zeros((500, 700), dtype=bool)
        ink[399:402, 100:601] = True
        best = best_projection([([350.0, 450.0, 550.0], [0.0, 0.0, 0.0])], x, y,
                               (100.0, 100.0, 600.0, 400.0), ink,
                               [Transform("scale", 0)], [Transform("scale", 0)])
        assert best is None or best.hit == 0.0


class TestScoredContrast:
    def test_points_on_frame_lines_do_not_testify(self):
        from real_chart_bench.domain.tick_calibration import scored_contrast

        ink = np.zeros((300, 400), dtype=bool)
        ink[95:106, 195:206] = True
        frame = (50, 20, 350, 250)
        # one marker hit, one point on the x axis (not scored)
        assert scored_contrast([(200, 100), (300, 250)], ink, frame, 2)[0] == 1.0

    def test_mostly_on_the_axis_proves_nothing(self):
        from real_chart_bench.domain.tick_calibration import scored_contrast

        ink = np.ones((300, 400), dtype=bool)
        frame = (50, 20, 350, 250)
        assert scored_contrast([(100, 250), (200, 250), (300, 100)], ink, frame, 2) == (0, 0)


class TestNiceNumberGrid:
    def test_a_misread_digit_off_the_label_grid_is_dropped(self):
        # "800" read as "809": close enough in pixels to pass as an inlier
        fit = fit_axis([_r(100, "200"), _r(200, "400"), _r(300, "600"), _r(400, "809")],
                       direction=+1)
        assert [v for _, v in fit.ticks] == [200, 400, 600]

    def test_missing_labels_keep_the_grid(self):
        fit = fit_axis([_r(100, "0"), _r(200, "0.5"), _r(400, "1.5"), _r(500, "2")],
                       direction=+1)
        assert len(fit.ticks) == 4

    def test_negative_and_fractional_steps(self):
        fit = fit_axis([_r(500, "-0.3"), _r(400, "-0.2"), _r(300, "-0.1"), _r(200, "0")],
                       direction=-1)
        assert len(fit.ticks) == 4

    def test_a_misread_cannot_invent_a_fine_grid(self):
        # 305 makes step 5 fit everything; a grid with most positions empty
        # is not how axes are labelled
        fit = fit_axis([_r(100, "100"), _r(200, "200"), _r(300, "300"), _r(301, "305")],
                       direction=+1)
        assert [v for _, v in fit.ticks] == [100, 200, 300]

    def test_tiny_values(self):
        fit = fit_axis([_r(100, "1e-13"), _r(200, "2e-13"), _r(300, "3e-13")], direction=+1)
        assert len(fit.ticks) == 3
