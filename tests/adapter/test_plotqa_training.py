"""PlotQA dot_line annotation -> one labels.jsonl line (docs/design/local-model.md)."""

import pytest

from real_chart_bench.adapter.plotqa_training import plotqa_label
from real_chart_bench.domain.training_data import max_axis_residual_px, validate_label


def _box(x, y, w=8, h=8):
    return {"x": x, "y": y, "w": w, "h": h}


def _ann(**model_overrides):
    # PlotQA repeats every tick/label list twice; x ticks are indices 0..n-1
    # whose printed labels are the years.
    x_axis = {
        "major_ticks": {"values": [0, 1, 2] * 2, "bboxes": [_box(96, 578), _box(196, 578),
                                                            _box(296, 578)] * 2},
        "major_labels": {"values": ["2001", "2002", "2003"] * 2, "bboxes": []},
        "label": {"text": "Year"},
    }
    y_axis = {
        "major_ticks": {"values": [0, 50, 100] * 2,
                        "bboxes": [_box(45, 496), _box(45, 296), _box(45, 96)] * 2},
        "major_labels": {"values": ["0", "50", "100"] * 2, "bboxes": []},
        "label": {"text": "Share (%)"},
    }
    model = {
        "name": "Mexico", "label": "Mexico", "color": "#5F9EA0",
        "x": [0, 1, 2], "y": [0.0, 25.0, 100.0],
        "bboxes": [_box(94, 494, 12, 12), _box(194, 394, 12, 12), _box(294, 94, 12, 12)],
    }
    model.update(model_overrides)
    return {
        "image_index": 7,
        "type": "dot_line",
        "general_figure_info": {
            "x_axis": x_axis, "y_axis": y_axis,
            "plot_info": {"bbox": {"x": 51, "y": 34, "w": 400, "h": 500}},
            "figure_info": {"bbox": {"bbox": {"x": 0, "y": 0, "w": 500, "h": 650}}},
        },
        "models": [model],
    }


def test_marker_centres_years_and_ticks_become_a_valid_label():
    label = plotqa_label(_ann(), image="images/7.png", width=500, height=650)

    assert validate_label(label) == []
    assert label["source"] == "plotqa" and label["license"] == "CC-BY-4.0"
    assert label["paper_id"] is None
    s = label["series"][0]
    assert s["points_px"] == [[100.0, 500.0], [200.0, 400.0], [300.0, 100.0]]
    assert s["points_value"] == [[2001.0, 0.0], [2002.0, 25.0], [2003.0, 100.0]]
    assert s["label"] == "Mexico" and s["color"] == "#5F9EA0"
    assert s["marker"] == "circle" and s["filled"] is True


def test_duplicated_tick_lists_are_halved_and_centred():
    label = plotqa_label(_ann(), image="images/7.png", width=500, height=650)
    assert label["axes"]["x"] == {
        "scale": "linear",
        "ticks": [{"px": 100.0, "value": 2001.0}, {"px": 200.0, "value": 2002.0},
                  {"px": 300.0, "value": 2003.0}],
    }
    assert [t["px"] for t in label["axes"]["y"]["ticks"]] == [500.0, 300.0, 100.0]
    assert label["plot_bbox"] == [51, 34, 451, 534]


def test_the_label_is_consistent_with_its_own_ticks():
    label = plotqa_label(_ann(), image="images/7.png", width=500, height=650)
    assert max_axis_residual_px(label) == pytest.approx(0.0)


def test_unparseable_printed_tick_labels_are_dropped_not_guessed():
    ann = _ann()
    ann["general_figure_info"]["y_axis"]["major_labels"]["values"] = ["0", "50", "1e2x"] * 2
    label = plotqa_label(ann, image="images/7.png", width=500, height=650)
    assert [t["value"] for t in label["axes"]["y"]["ticks"]] == [0.0, 50.0]


def test_a_series_whose_boxes_do_not_match_its_values_is_refused():
    with pytest.raises(ValueError, match="bboxes"):
        plotqa_label(_ann(y=[1.0, 2.0]), image="images/7.png", width=500, height=650)


def test_only_dot_line_plots_are_converted():
    with pytest.raises(ValueError, match="dot_line"):
        plotqa_label(_ann() | {"type": "line"}, image="i.png", width=500, height=650)


def test_unevenly_stepped_years_make_the_x_axis_categorical():
    # 1990, 2000, 2003 are printed at equal steps: not a linear axis, so the
    # axes are unknown; the y ticks are still kept on their own.
    ann = _ann()
    ann["general_figure_info"]["x_axis"]["major_labels"]["values"] = ["1990", "2000", "2003"] * 2
    label = plotqa_label(ann, image="images/7.png", width=500, height=650)

    assert validate_label(label) == []
    assert label["axes"] is None and label["x_categorical"] is True
    assert [t["value"] for t in label["y_ticks"]] == [0.0, 50.0, 100.0]
    assert label["series"][0]["points_value"][1] == [2000.0, 25.0]


def test_evenly_stepped_years_keep_both_axes():
    label = plotqa_label(_ann(), image="images/7.png", width=500, height=650)
    assert label["x_categorical"] is False and label["y_ticks"] is None
