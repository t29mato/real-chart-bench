"""CHART-Infographics 2024 annotation -> one labels.jsonl line (docs/design/local-model.md)."""

from real_chart_bench.adapter.chartinfo_training import (
    LICENSE,
    chartinfo_label,
    excluded_pmcids,
    pmcid_of,
)
from real_chart_bench.domain.training_data import assert_no_benchmark_leak, validate_label


def _ann(chart_type="scatter", elements=None, x_labels=("0", "50", "100"),
         y_labels=("0", "10", "20"), legend=None, data=None):
    blocks, x_axis, y_axis = [], [], []
    for i, (t, px) in enumerate(zip(x_labels, (100, 200, 300), strict=False)):
        blocks.append({"id": i, "text": t})
        x_axis.append({"id": i, "tick_pt": {"x": px, "y": 400}})
    for j, (t, py) in enumerate(zip(y_labels, (400, 250, 100), strict=False)):
        blocks.append({"id": 10 + j, "text": t})
        y_axis.append({"id": 10 + j, "tick_pt": {"x": 100, "y": py}})
    if elements is None:
        elements = [[{"x": 150.0, "y": 325.0}, {"x": 250.0, "y": 175.0}]]
    key = "scatter points" if chart_type == "scatter" else "lines"
    return {
        "task1": {"output": {"chart_type": chart_type}},
        "task2": {"output": {"text_blocks": blocks}},
        "task4": {"output": {
            "_plot_bb": {"x0": 100, "y0": 100, "width": 200, "height": 300},
            "axes": {"x-axis": x_axis, "y-axis": y_axis},
        }},
        "task5": {"output": {"legend_pairs": legend or []}},
        "task6": {"output": {
            "data series": data if data is not None else [
                {"name": "a", "data": [{"x": 25.0, "y": 5.0}, {"x": 75.0, "y": 15.0}]}],
            "visual elements": {key: elements},
        }},
    }


def _label(ann, kind="scatter", stem="PMC123___fig.1"):
    return chartinfo_label(ann, image=f"images/{stem}.jpg", width=500, height=500,
                           stem=stem, kind=kind)


def test_pmcid_is_the_part_before_the_triple_underscore():
    assert pmcid_of("PMC1173100___1471-2156-6-24-1") == "PMC1173100"
    assert pmcid_of("CHARTINFO_2024_Train/images/scatter/PMC5457113___m-g002.jpg") == "PMC5457113"


def test_scatter_label_is_valid_and_carries_paper_and_licence():
    lab = _label(_ann())
    assert validate_label(lab) == []
    assert lab["source"] == "chartinfo"
    assert lab["license"] == LICENSE == "CC-BY-NC-SA-4.0"
    assert lab["paper_id"] == "PMC123"
    assert lab["plot_bbox"] == [100, 100, 300, 400]
    assert lab["series"][0]["points_px"] == [[150.0, 325.0], [250.0, 175.0]]


def test_scatter_values_are_kept_when_counts_match():
    s = _label(_ann())["series"][0]
    assert s["points_value"] == [[25.0, 5.0], [75.0, 15.0]]
    assert s["label"] == "a"


def test_values_are_dropped_not_guessed_when_counts_differ():
    lab = _label(_ann(data=[{"name": "a", "data": [{"x": 1.0, "y": 2.0}]}]))
    assert lab["series"][0].get("points_value") is None
    assert validate_label(lab) == []


def test_ticks_parse_printed_numbers_and_axes_need_two_each():
    lab = _label(_ann())
    assert [t["value"] for t in lab["axes"]["x"]["ticks"]] == [0, 50, 100]
    assert lab["axes"]["y"]["scale"] == "linear"
    no_axes = _label(_ann(x_labels=("a", "b", "50")))
    assert no_axes["axes"] is None  # unknown, not guessed; points are still usable
    assert validate_label(no_axes) == []


def test_log_axis_is_recognised_from_a_geometric_ladder():
    lab = _label(_ann(y_labels=("1", "10", "100")))
    assert lab["axes"]["y"]["scale"] == "log"


def test_line_chart_marks_vertices_as_unverified_markers():
    lab = _label(_ann(chart_type="line"), kind="line")
    assert lab["extra"]["vertices_only"] is True
    assert lab["series"][0]["points_px"]


def test_legend_boxes_are_recorded_as_negatives():
    lab = _label(_ann(legend=[{"bb": {"x0": 10, "y0": 20, "width": 5, "height": 6}, "id": 3}]))
    assert lab["extra"]["legend_bboxes"] == [[10, 20, 15, 26]]


def test_points_outside_the_image_are_dropped_and_empty_series_removed():
    lab = _label(_ann(elements=[[{"x": 900.0, "y": 10.0}], [{"x": 150.0, "y": 300.0}]]))
    assert [s["points_px"] for s in lab["series"]] == [[[150.0, 300.0]]]


def test_no_points_gives_no_label():
    assert _label(_ann(elements=[])) is None


def test_excluded_pmcids_cover_every_measurement_figure_paper():
    keys = [{"fig_01.png": {"image": "CHARTINFO_2024_Train/images/scatter/PMC9___a.jpg",
                            "source": "CHARTINFO_2024_Train/annotations_JSON/scatter/PMC9___a.json"}},
            {"fig_01.png": {"source": "x/PMC7___b.json"}}]
    pm = excluded_pmcids(keys)
    assert pm == {"PMC9", "PMC7"}
    lab = _label(_ann(), stem="PMC9___other-figure")
    try:
        assert_no_benchmark_leak([lab], pm)
    except ValueError:
        pass
    else:
        raise AssertionError("a figure from a measured paper must be rejected")
