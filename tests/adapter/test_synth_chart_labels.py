"""Pure helpers of the materials-style chart generator
(scripts/train/gen_synth_materials.py): matplotlib marker codes, display ->
image pixels, printed tick text -> value, and points hidden under boxes."""

import pytest

from real_chart_bench.adapter.synth_chart_labels import (
    display_to_image,
    inside_any,
    marker_class,
    parse_tick_text,
)


@pytest.mark.parametrize(
    "code,name",
    [("o", "circle"), ("s", "square"), ("^", "triangle"), ("v", "triangle"),
     ("<", "triangle"), (">", "triangle"), ("D", "diamond"), ("d", "diamond"),
     ("x", "cross"), ("+", "cross"), ("*", "other"), ("p", "other"), ("h", "other")],
)
def test_matplotlib_markers_map_to_label_classes(code, name):
    assert marker_class(code) == name


def test_an_unknown_marker_code_is_refused():
    with pytest.raises(ValueError):
        marker_class("$a$")


def test_display_origin_is_bottom_left_image_origin_is_top_left():
    assert display_to_image(10.0, 0.0, height=600) == (10.0, 600.0)
    assert display_to_image(10.0, 600.0, height=600) == (10.0, 0.0)
    assert display_to_image(2.5, 100.25, height=480) == (2.5, 379.75)


def test_a_supersampled_drawing_maps_onto_the_shrunk_image():
    # drawn 4x larger (2400 px tall), saved shrunk to 600 px
    assert display_to_image(40.0, 2400.0, height=2400, scale=4) == (10.0, 0.0)
    assert display_to_image(10.0, 2.0, height=2400, scale=4) == (2.5, 599.5)


@pytest.mark.parametrize(
    "text,value",
    [("300", 300.0), ("0.5", 0.5), ("−1.5", -1.5), ("-2", -2.0),
     ("$\\mathdefault{10^{-3}}$", 1e-3), ("$\\mathdefault{10^{2}}$", 100.0),
     ("$\\mathdefault{10^{0}}$", 1.0), ("$\\mathdefault{2\\times10^{-1}}$", 0.2),
     ("$\\mathdefault{0.5}$", 0.5), ("1,000", 1000.0)],
)
def test_printed_tick_text_is_read_as_its_value(text, value):
    assert parse_tick_text(text) == pytest.approx(value)


@pytest.mark.parametrize("text", ["", "abc", "$\\mathdefault{}$", "1e"])
def test_tick_text_that_is_not_a_number_is_unknown(text):
    assert parse_tick_text(text) is None


def test_points_under_a_legend_or_inset_box_are_found():
    boxes = [(10.0, 10.0, 50.0, 40.0)]
    assert inside_any((20.0, 20.0), boxes)
    assert not inside_any((60.0, 20.0), boxes)
    assert not inside_any((20.0, 20.0), [])
