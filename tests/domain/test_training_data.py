"""Training data for the local extractor (docs/design/local-model.md):
benchmark papers never enter it, and every label line has one checked shape."""

import pytest

from real_chart_bench.domain.training_data import (
    BenchmarkLeakError,
    assert_no_benchmark_leak,
    validate_label,
)


def _label(**overrides):
    label = {
        "image": "img/a.png",
        "width": 800,
        "height": 600,
        "source": "plotqa",
        "license": "CC-BY-4.0",
        "paper_id": None,
        "axes": {
            "x": {
                "scale": "linear",
                "ticks": [{"px": 100.0, "value": 0}, {"px": 700.0, "value": 10}],
            },
            "y": {
                "scale": "log",
                "ticks": [{"px": 500.0, "value": 1e-3}, {"px": 50.0, "value": 1}],
            },
        },
        "plot_bbox": [100, 50, 700, 500],
        "series": [
            {
                "label": "a",
                "marker": "circle",
                "filled": True,
                "color": "#1f77b4",
                "points_px": [[150.0, 400.0]],
                "points_value": [[1.0, 0.01]],
            }
        ],
    }
    label.update(overrides)
    return label


def test_a_complete_label_has_no_errors():
    assert validate_label(_label()) == []


def test_unknown_fields_may_be_null_instead_of_guessed():
    s = _label()["series"][0] | {
        "marker": None,
        "filled": None,
        "color": None,
        "points_value": None,
    }
    assert validate_label(_label(axes=None, plot_bbox=None, series=[s])) == []


def test_points_outside_the_image_are_reported():
    s = _label()["series"][0] | {"points_px": [[900.0, 10.0]]}
    assert any("outside the image" in e for e in validate_label(_label(series=[s])))


def test_value_and_pixel_point_counts_must_match():
    s = _label()["series"][0] | {"points_value": [[1.0, 0.01], [2.0, 0.02]]}
    assert any("points_value" in e for e in validate_label(_label(series=[s])))


def test_an_axis_needs_two_ticks_and_a_known_scale():
    axes = _label()["axes"] | {"x": {"scale": "lin", "ticks": [{"px": 1.0, "value": 0}]}}
    errors = validate_label(_label(axes=axes))
    assert any("scale" in e for e in errors) and any("ticks" in e for e in errors)


def test_a_real_figure_needs_its_paper_id():
    assert any("paper_id" in e for e in validate_label(_label(source="starrydata", paper_id=None)))


def test_benchmark_papers_are_refused():
    labels = [
        _label(source="starrydata", paper_id="4173"),
        _label(source="starrydata", paper_id="9"),
    ]

    with pytest.raises(BenchmarkLeakError, match="4173"):
        assert_no_benchmark_leak(labels, benchmark_paper_ids={"4173", "28331"})


def test_synthetic_labels_and_other_papers_pass():
    labels = [_label(), _label(source="starrydata", paper_id="9")]
    assert_no_benchmark_leak(labels, benchmark_paper_ids={"4173"})
