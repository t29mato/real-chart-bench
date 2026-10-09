"""scripts/eval/chartinfo_paired_ci.py: per-figure score parsing and the paired
bootstrap used for the model-difference intervals in paper 4.8."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "eval" / "chartinfo_paired_ci.py"


def load():
    spec = importlib.util.spec_from_file_location("chartinfo_paired_ci", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SCORE = """図              正解   予測      総合     名前     データ    点F1     再現     適合  出典
fig_01.png     45   45   0.742  1.000   0.657  1.000  1.000  1.000  CMMM2015-574238.004
fig_02.png     50   51   0.779  0.944   0.723  0.990  1.000  0.980  materials-10-01307-g007

軸レンジの出どころ: {'axis': 27, 'gt-spread': 2, 'partial': 1}

全30図      総合 0.6897  名前 0.8697  データ 0.6297  |  点F1 0.848  再現 0.846  適合 0.850
"""


def test_parse_reads_per_figure_rows_only():
    rows = load().parse_score_text(SCORE)
    assert set(rows) == {"fig_01.png", "fig_02.png"}
    assert rows["fig_02.png"]["combined"] == pytest.approx(0.779)
    assert rows["fig_02.png"]["data"] == pytest.approx(0.723)
    assert rows["fig_02.png"]["point_f1"] == pytest.approx(0.990)
    assert rows["fig_01.png"]["precision"] == pytest.approx(1.0)


def test_identical_scores_give_a_zero_interval():
    mean, lo, hi = load().paired_bootstrap([0.3, 0.9, 0.5], [0.3, 0.9, 0.5], n_boot=200, seed=0)
    assert (mean, lo, hi) == (0.0, 0.0, 0.0)


def test_constant_difference_is_recovered_exactly():
    a = [0.5, 0.7, 0.9, 0.1]
    b = [x - 0.1 for x in a]
    mean, lo, hi = load().paired_bootstrap(a, b, n_boot=200, seed=0)
    assert mean == pytest.approx(0.1)
    assert lo == pytest.approx(0.1)
    assert hi == pytest.approx(0.1)


def test_interval_contains_the_mean_and_is_deterministic():
    mod = load()
    a = [1.0, 0.0, 1.0, 0.5, 0.2, 0.9]
    b = [0.0, 1.0, 0.8, 0.5, 0.4, 0.1]
    first = mod.paired_bootstrap(a, b, n_boot=500, seed=3)
    assert first == mod.paired_bootstrap(a, b, n_boot=500, seed=3)
    mean, lo, hi = first
    assert lo <= mean <= hi


def test_unpaired_input_is_rejected():
    with pytest.raises(ValueError):
        load().paired_bootstrap([0.1, 0.2], [0.1], n_boot=10, seed=0)
