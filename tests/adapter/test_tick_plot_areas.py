"""The tick-calibrated LineFormer row (design §7.66) maps LineFormer's pixels
through the owner-reviewed tick-mark positions instead of the full image
frame. Those positions are only trusted when they still describe the figure
as scored: same image, tick labels equal to the registry's ranges, not
marked stale."""

import json

from real_chart_bench.adapter.tick_plot_areas import load_tick_plot_areas
from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.verified_pairing import VerifiedPairing


def _pairing(**overrides):
    fields = dict(
        paper_id="1",
        figure_id="2",
        image_path="data/verified_pairs/crops/1/a.png",
        x_range=(300.0, 900.0),
        y_range=(0.0, 40.0),
        x_scale=ScaleType.LINEAR,
        y_scale=ScaleType.LINEAR,
    )
    fields.update(overrides)
    return fields


def _entry(**overrides):
    entry = {
        "paper_id": "1",
        "figure_id": "2",
        "image_path": "data/verified_pairs/crops/1/a.png",
        "x_min_label": 300,
        "x_max_label": 900,
        "y_min_label": 0,
        "y_max_label": 40,
        "pixel_bbox_mean": {"x_min_px": 10, "x_max_px": 410, "y_min_px": 300, "y_max_px": 20},
    }
    entry.update(overrides)
    return entry


def _load(tmp_path, entry, pairing_fields):
    path = tmp_path / "axis.json"
    path.write_text(json.dumps([{"_meta": {}}, entry]))
    pairing = _make(pairing_fields)
    return load_tick_plot_areas(path, [pairing])


def _make(fields):
    # VerifiedPairing has many required fields; build through the registry
    # parser's own defaults would couple this test to the file format, so
    # use a minimal stand-in exposing what the loader reads.
    class _P:
        pass

    p = _P()
    for k, v in fields.items():
        setattr(p, k, v)
    return p


def test_valid_entry_gives_the_tick_bbox_as_pixel_bbox(tmp_path):
    areas = _load(tmp_path, _entry(), _pairing())

    # (x0, y0, x1, y1) with y0 the top (y_max tick) and y1 the bottom
    assert areas == {("1", "2"): (10.0, 20.0, 410.0, 300.0)}


def test_stale_entry_is_not_used(tmp_path):
    entry = _entry(pixel_coords_stale_since="2026-09-30")
    assert _load(tmp_path, entry, _pairing()) == {}


def test_labels_that_differ_from_the_scored_ranges_are_not_used(tmp_path):
    assert _load(tmp_path, _entry(), _pairing(x_range=(280.0, 900.0))) == {}


def test_a_different_image_is_not_used(tmp_path):
    entry = _entry(image_path="data/verified_pairs/crops/1/b.png")
    assert _load(tmp_path, entry, _pairing()) == {}


def test_missing_labels_are_not_used(tmp_path):
    assert _load(tmp_path, _entry(y_max_label=None), _pairing()) == {}


def test_verified_pairing_exposes_the_fields_the_loader_reads():
    names = VerifiedPairing.__dataclass_fields__
    for field in ("paper_id", "figure_id", "image_path", "x_range", "y_range"):
        assert field in names


# --- tick_calibration.json (design 7.77): any two ticks, not the range ends ---


def test_plot_area_from_two_ticks_extrapolates_to_the_range_ends():
    from real_chart_bench.adapter.tick_plot_areas import plot_area_from_ticks

    cal = {
        "x": [{"px": 100.0, "value": 400.0}, {"px": 300.0, "value": 800.0}],
        "y": [{"px": 500.0, "value": 0.0}, {"px": 100.0, "value": 40.0}],
    }

    box = plot_area_from_ticks(cal, (300.0, 900.0), (0.0, 50.0), "linear", "linear")

    # x: 0.5 px per unit -> 300 at 50, 900 at 350; y: -10 px per unit -> 50 at 0
    assert box == (50.0, 0.0, 350.0, 500.0)


def test_plot_area_from_ticks_on_a_log_axis_interpolates_in_log10():
    from real_chart_bench.adapter.tick_plot_areas import plot_area_from_ticks

    cal = {
        "x": [{"px": 0.0, "value": 0.0}, {"px": 100.0, "value": 10.0}],
        "y": [{"px": 400.0, "value": 1e-4}, {"px": 100.0, "value": 1e-1}],
    }

    box = plot_area_from_ticks(cal, (0.0, 10.0), (1e-5, 1.0), "linear", "log")

    # 100 px per decade: 1e-5 one decade below 1e-4 -> 500; 1 one above 1e-1 -> 0
    assert box[1] == 0.0 and box[3] == 500.0


# --- two-stage pixcal (design 7.79): the model gives pixels, the scorer converts ---


def test_pixel_to_value_follows_the_two_ticks_on_a_linear_axis():
    from real_chart_bench.adapter.tick_plot_areas import pixel_to_value

    ticks = [{"px": 100.0, "value": 400.0}, {"px": 300.0, "value": 800.0}]

    assert pixel_to_value(ticks, 200.0, "linear") == 600.0
    assert pixel_to_value(ticks, 50.0, "linear") == 300.0  # outside the ticks too


def test_pixel_to_value_on_a_log_axis_is_linear_in_log10():
    from real_chart_bench.adapter.tick_plot_areas import pixel_to_value

    ticks = [{"px": 400.0, "value": 1e-4}, {"px": 100.0, "value": 1e-1}]

    assert abs(pixel_to_value(ticks, 300.0, "log") - 1e-3) < 1e-15
    assert abs(pixel_to_value(ticks, 0.0, "log") - 1.0) < 1e-12


_CAL = {
    "x_scale": "linear",
    "y_scale": "linear",
    "x": [{"px": 0.0, "value": 0.0}, {"px": 100.0, "value": 10.0}],
    "y": [{"px": 200.0, "value": 0.0}, {"px": 0.0, "value": 20.0}],
}


def test_pixel_answer_is_converted_series_by_series():
    from real_chart_bench.adapter.tick_plot_areas import values_from_pixel_answer

    answer = [{"label": "a", "x": [0, 50], "y": [200, 100]}]

    out = values_from_pixel_answer(answer, _CAL, image_size=(100, 200), coords="pixel")

    assert out == [{"label": "a", "x": [0.0, 5.0], "y": [0.0, 10.0]}]


def test_normalized_0_1000_answer_is_scaled_to_the_image_first():
    from real_chart_bench.adapter.tick_plot_areas import values_from_pixel_answer

    # 500/1000 of a 100-px-wide image is pixel 50; 250/1000 of 200 px is 50
    answer = [{"label": "a", "x": [500], "y": [250]}]

    out = values_from_pixel_answer(answer, _CAL, image_size=(100, 200), coords="norm1000")

    assert out == [{"label": "a", "x": [5.0], "y": [15.0]}]


def test_non_numeric_entries_are_left_for_the_curve_parser_to_drop():
    from real_chart_bench.adapter.tick_plot_areas import values_from_pixel_answer

    answer = [{"label": "a", "x": [0, None], "y": [200, 100]}, "junk"]

    out = values_from_pixel_answer(answer, _CAL, image_size=(100, 200), coords="pixel")

    assert out == [{"label": "a", "x": [0.0, None], "y": [0.0, 10.0]}]
