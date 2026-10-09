"""Detector / extractor answer -> starry-digitizer auto-extraction v1 document
(/var/tmp/starry-digitizer-auto-extraction-schema-v1.md, 2026-10-09)."""

import pytest

from real_chart_bench.adapter.starry_digitizer_export import auto_extraction_doc


def _answer():
    return [{"label": "series_1", "x": [10.0, 20.0], "y": [30.0, 40.0]},
            {"label": "series_2", "x": [50.0], "y": [60.0]}]


def test_pixel_answer_becomes_points_px_with_the_v1_header():
    doc = auto_extraction_doc(_answer(), image="fig.png", width=800, height=600,
                              source={"extractor": "markernet-v2"})
    assert doc["format"] == "starry-digitizer/auto-extraction"
    assert doc["version"] == 1
    assert doc["image"] == {"url": "fig.png", "width": 800, "height": 600}
    assert doc["axes"] is None
    assert doc["series"][0] == {"label": "series_1", "points_px": [[10.0, 30.0], [20.0, 40.0]],
                                "points_value": None, "scores": None}
    assert doc["source"] == {"extractor": "markernet-v2"}


def test_pixel_coordinates_are_not_shifted():
    # our labels and detections are continuous with the origin at the top-left
    # corner of the top-left pixel -- the same convention as starry-digitizer
    doc = auto_extraction_doc([{"label": "a", "x": [0.0], "y": [0.0]}], image=None)
    assert doc["series"][0]["points_px"] == [[0.0, 0.0]]


def test_value_answers_need_axes_or_are_written_as_values():
    axes = {"x": {"scale": "linear", "ticks": [{"px": 0, "value": 0}, {"px": 10, "value": 1}]},
            "y": {"scale": "log", "ticks": [{"px": 10, "value": 1}, {"px": 0, "value": 10}]}}
    doc = auto_extraction_doc(_answer(), image=None, axes=axes, space="value")
    assert doc["axes"] == axes
    assert doc["series"][0]["points_px"] is None
    assert doc["series"][0]["points_value"] == [[10.0, 30.0], [20.0, 40.0]]
    with pytest.raises(ValueError):
        auto_extraction_doc(_answer(), image=None, space="value")


def test_scores_are_attached_per_series_and_checked_for_length():
    doc = auto_extraction_doc(_answer(), image=None, scores=[[0.9, 0.4], [0.7]])
    assert doc["series"][1]["scores"] == [0.7]
    with pytest.raises(ValueError):
        auto_extraction_doc(_answer(), image=None, scores=[[0.9], [0.7]])


def test_empty_or_unlabelled_series():
    doc = auto_extraction_doc([{"x": [1.0], "y": [2.0]}, {"label": "e", "x": [], "y": []}],
                              image=None)
    assert [s["label"] for s in doc["series"]] == ["series 1"]
