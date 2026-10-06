"""Curve F1 for dense-marker figures (design 7.83): no one-to-one point
matching -- a ground-truth point counts as covered when some predicted point
of the paired series lies within tau, and a predicted point counts as on the
curve when some ground-truth point lies within tau. Series are paired one to
one to maximise curve F1. Works for curves that are not functions of x
(Nyquist arcs), unlike the curve-distance metric."""

import numpy as np

from real_chart_bench.domain.point_metrics import curve_f1


def arc(n, r=0.4, cx=0.5, cy=0.0):
    t = np.linspace(0, np.pi, n)
    return np.c_[cx + r * np.cos(t), cy + r * np.sin(t)]


def test_resampled_dense_curve_scores_full_marks():
    gt = [arc(200)]
    pred = [arc(57)]  # same arc, different sampling: one-to-one matching would fail

    out = curve_f1(pred, gt, tau=0.02)

    assert out["f1"] == 1.0


def test_half_the_curve_gives_half_recall_and_full_precision():
    gt = [arc(200)]
    pred = [arc(200)[:100]]

    out = curve_f1(pred, gt, tau=0.02)

    assert abs(out["recall"] - 0.5) < 0.02 and out["precision"] == 1.0


def test_points_off_the_curve_lower_precision():
    gt = [arc(200)]
    pred = [np.vstack([arc(100), [[0.5, 0.9]] * 100])]

    out = curve_f1(pred, gt, tau=0.02)

    assert out["recall"] == 1.0 and abs(out["precision"] - 0.5) < 1e-9


def test_series_are_paired_one_to_one_and_unpaired_series_count_as_misses():
    a, b = arc(100), arc(100, r=0.2)
    out = curve_f1([b, a], [a, b], tau=0.02)
    assert out["f1"] == 1.0

    out = curve_f1([a], [a, b], tau=0.02)
    assert abs(out["recall"] - 0.5) < 1e-9 and out["precision"] == 1.0


def test_no_prediction_scores_zero():
    assert curve_f1([], [arc(10)], tau=0.02)["f1"] == 0.0
