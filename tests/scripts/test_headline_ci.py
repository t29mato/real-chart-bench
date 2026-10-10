"""scripts/eval/headline_ci.py: figure- and paper-level bootstrap of macro point F1."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "headline_ci.py"


def load():
    spec = importlib.util.spec_from_file_location("headline_ci", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def row(fig, f1, dense=False, point=True):
    r = {"figure_id": fig, "marker_density": {"dense": dense}}
    r["point"] = {"by_tau": {"0.02": {"point_f1": f1}}} if point else None
    return r


def test_load_point_f1_skips_dense_and_pointless():
    m = load()
    data = {
        "per_figure": [row("1-1", 0.5), row("1-2", 1.0, dense=True), row("2-1", 0, point=False)]
    }
    assert m.load_point_f1(data) == {"1-1": 0.5}


def test_paper_of():
    assert load().paper_of("14649-43660") == "14649"


def test_constant_values_give_degenerate_interval():
    m = load()
    assert m.bootstrap_mean([0.7] * 5, None, n_boot=200) == (pytest.approx(0.7), 0.7, 0.7)
    assert m.bootstrap_mean([0.7] * 4, ["a", "a", "b", "c"], n_boot=200)[1:] == (0.7, 0.7)


def test_mean_is_plain_mean_and_interval_brackets_it():
    m = load()
    vals = [0.0, 1.0, 0.5, 0.9, 0.2, 1.0]
    for groups in (None, ["a", "a", "b", "b", "c", "c"]):
        mean, lo, hi = m.bootstrap_mean(vals, groups, n_boot=2000, seed=1)
        assert mean == pytest.approx(sum(vals) / 6)
        assert lo <= mean <= hi


def test_fixed_seed_is_reproducible():
    m = load()
    vals = [0.1, 0.9, 0.4, 0.8]
    g = ["a", "b", "b", "c"]
    assert m.bootstrap_mean(vals, g, n_boot=500, seed=3) == m.bootstrap_mean(
        vals, g, n_boot=500, seed=3
    )


def test_paper_bootstrap_is_wider_when_papers_are_homogeneous():
    m = load()
    # two papers, figures within a paper identical: only 2 distinct resample outcomes per paper
    vals = [0.0] * 10 + [1.0] * 10
    groups = ["a"] * 10 + ["b"] * 10
    _, flo, fhi = m.bootstrap_mean(vals, None, n_boot=2000, seed=0)
    _, plo, phi = m.bootstrap_mean(vals, groups, n_boot=2000, seed=0)
    assert (phi - plo) > (fhi - flo)


def test_empty_and_mismatched_inputs_raise():
    m = load()
    with pytest.raises(ValueError):
        m.bootstrap_mean([])
    with pytest.raises(ValueError):
        m.bootstrap_mean([1.0, 2.0], ["a"])


def test_paired_difference_uses_common_figures_only():
    m = load()
    keys, d = m.paired_difference({"1-1": 1.0, "1-2": 0.5, "2-1": 0.0}, {"1-1": 0.5, "2-1": 0.25})
    assert keys == ["1-1", "2-1"]
    assert d == [0.5, -0.25]
