"""scripts/eval/chartinfo_rescale_gain.py: how much point F1 a power-of-ten
rescaling of one axis of the answer would recover (an upper bound on the loss
from %, x10^n and similar unit conventions, paper 4.8.2)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "chartinfo_rescale_gain.py"


def load():
    spec = importlib.util.spec_from_file_location("chartinfo_rescale_gain", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


GT = [{"data": [{"x": 1.0, "y": 0.10}, {"x": 2.0, "y": 0.25}, {"x": 3.0, "y": 0.40}]}]


def series(points):
    return [{"data": [{"x": x, "y": y} for x, y in points]}]


def test_percent_answer_is_recovered_by_dividing_y_by_100():
    pred = series([(1.0, 10.0), (2.0, 25.0), (3.0, 40.0)])
    best = load().best_rescale(pred, GT)
    assert best["f1"] == pytest.approx(1.0)
    assert best["axis"] == "y"
    assert best["k"] == -2
    assert best["f1_as_answered"] == pytest.approx(0.0)


def test_correct_answer_needs_no_rescaling():
    pred = series([(1.0, 0.10), (2.0, 0.25), (3.0, 0.40)])
    best = load().best_rescale(pred, GT)
    assert best["f1"] == pytest.approx(1.0)
    assert best["k"] == 0


def test_rescaling_x_is_also_tried():
    pred = series([(1000.0, 0.10), (2000.0, 0.25), (3000.0, 0.40)])
    best = load().best_rescale(pred, GT)
    assert best["f1"] == pytest.approx(1.0)
    assert (best["axis"], best["k"]) == ("x", -3)


def test_wrong_reading_is_not_rescued():
    pred = series([(1.0, 0.40), (2.0, 0.10), (3.0, 0.25)])
    best = load().best_rescale(pred, GT)
    assert best["f1"] < 1.0
