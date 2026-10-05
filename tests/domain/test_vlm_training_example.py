"""Approach B (docs/design/local-model.md): one labels.jsonl line becomes the
task entry and the target answer of the v3 single-shot prompt, so the
fine-tuned model learns exactly the shape it is scored on."""

import json
import math
import pathlib

import pytest

from real_chart_bench.domain.vlm_training_example import (
    PRINTED_REPORT_RULE,
    format_answer,
    noaxis_task,
    pixcal_task,
    round_sig,
    split_of,
    target_answer,
)


def _label(**overrides):
    label = {
        "image": "img/a.png",
        "width": 800,
        "height": 600,
        "source": "synth-materials",
        "license": "CC0-1.0",
        "paper_id": None,
        "axes": {
            "x": {
                "scale": "linear",
                "ticks": [
                    {"px": 300.0, "value": 5},
                    {"px": 100.0, "value": 0},
                    {"px": 700.0, "value": 20},
                ],
            },
            "y": {
                "scale": "log",
                "ticks": [{"px": 500.0, "value": 1e-3}, {"px": 50.0, "value": 1}],
            },
        },
        "plot_bbox": [100, 50, 700, 500],
        "series": [
            {
                "label": "x=0.1",
                "marker": "circle",
                "points_px": [[400.0, 400.0], [150.0, 300.0]],
                "points_value": [[7.5, 0.0123456], [1.25, 0.05]],
            },
            {
                "label": None,
                "marker": "square",
                "points_px": [[200.0, 200.0]],
                "points_value": [[2.0, 0.1]],
            },
        ],
    }
    label.update(overrides)
    return label


def test_noaxis_task_uses_the_printed_space_rule_for_both_axes():
    assert noaxis_task("fig_007.png") == {
        "id": "fig_007.png",
        "x_report": PRINTED_REPORT_RULE,
        "y_report": PRINTED_REPORT_RULE,
    }


def test_printed_rule_is_the_benchmark_default_wording():
    # the same sentence the benchmark's noaxis tasks carry for a plain axis
    repo = pathlib.Path(__file__).resolve().parents[2]
    tasks = json.loads((repo / "data/llm_run_v3/noaxis/tasks.json").read_text())
    assert tasks[0]["x_report"] == PRINTED_REPORT_RULE


def test_pixcal_task_takes_the_two_outermost_ticks_per_axis():
    task = pixcal_task("fig_007.png", _label())
    assert task == {
        "id": "fig_007.png",
        "image_size": [800, 600],
        "x_scale": "linear",
        "y_scale": "log",
        "x_ticks": [{"pixel_x": 100.0, "value": 0}, {"pixel_x": 700.0, "value": 20}],
        "y_ticks": [{"pixel_y": 500.0, "value": 1e-3}, {"pixel_y": 50.0, "value": 1}],
    }


def test_pixcal_task_rounds_pixels_and_values():
    lab = _label()
    lab["axes"]["x"]["ticks"] = [
        {"px": 94.791666, "value": 300.0},
        {"px": 417.70833, "value": 800.0},
    ]
    task = pixcal_task("f.png", lab)
    assert task["x_ticks"] == [{"pixel_x": 94.8, "value": 300}, {"pixel_x": 417.7, "value": 800}]
    assert type(task["x_ticks"][0]["value"]) is int


def test_pixcal_task_is_none_without_axes():
    assert pixcal_task("f.png", _label(axes=None)) is None


def test_pixcal_task_is_none_with_one_tick():
    lab = _label()
    lab["axes"]["y"]["ticks"] = lab["axes"]["y"]["ticks"][:1]
    assert pixcal_task("f.png", lab) is None


def test_target_answer_sorts_points_by_x_and_names_unlabelled_series():
    ans = target_answer("fig_007.png", _label())
    assert ans == {
        "fig_007.png": [
            {"label": "x=0.1", "x": [1.25, 7.5], "y": [0.05, 0.01235]},
            {"label": "series 2", "x": [2], "y": [0.1]},
        ]
    }


def test_target_answer_is_none_without_values():
    lab = _label()
    lab["series"][0]["points_value"] = None
    assert target_answer("f.png", lab) is None


def test_target_answer_is_none_with_non_finite_values():
    lab = _label()
    lab["series"][0]["points_value"] = [[1.0, math.inf], [2.0, 1.0]]
    assert target_answer("f.png", lab) is None


def test_target_answer_drops_empty_series_and_is_none_if_nothing_left():
    lab = _label()
    lab["series"][1]["points_value"] = []
    lab["series"][1]["points_px"] = []
    assert len(target_answer("f.png", lab)["f.png"]) == 1
    lab["series"] = [lab["series"][1]]
    assert target_answer("f.png", lab) is None


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (0.0123456, 0.01235),
        (123456.0, 123500),
        (2.0, 2),
        (-0.000123449, -0.0001234),
        (0.0, 0),
        (1e-12, 1e-12),
    ],
)
def test_round_sig_keeps_four_significant_digits_and_integers_as_int(value, expected):
    out = round_sig(value)
    assert out == expected
    assert type(out) is type(expected)


def test_format_answer_is_one_line_json_that_parses_back():
    ans = target_answer("fig_007.png", _label())
    text = format_answer(ans)
    assert "\n" not in text
    assert json.loads(text) == ans


def test_split_is_deterministic_and_roughly_the_requested_fraction():
    keys = [f"plotqa/img/{i}.png" for i in range(4000)]
    val = [k for k in keys if split_of(k, val_fraction=0.05) == "val"]
    assert 120 < len(val) < 280
    assert val == [k for k in keys if split_of(k, val_fraction=0.05) == "val"]
    assert split_of("x", val_fraction=0.0) == "train"
