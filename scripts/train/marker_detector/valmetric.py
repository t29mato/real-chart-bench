"""Validation metric on the training-data split: the benchmark's own point F1
(domain.point_metrics, series matched by Hungarian, tau 2%) computed in
pixel space, with the plot box (or the whole image) as the axis frame.
Nothing here reads the benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from real_chart_bench.domain.curve import Curve  # noqa: E402
from real_chart_bench.domain.point_metrics import AxisFrame, evaluate_points  # noqa: E402


def _frame(label: dict) -> AxisFrame:
    b = label.get("plot_bbox")
    if b and b[2] > b[0] and b[3] > b[1]:
        return AxisFrame(x_range=(b[0], b[2]), y_range=(b[1], b[3]))
    return AxisFrame(x_range=(0, label["width"]), y_range=(0, label["height"]))


def _curves(series) -> list[Curve]:
    out = []
    for xs, ys in series:
        if len(xs) >= 2:
            out.append(Curve(x_values=tuple(map(float, xs)), y_values=tuple(map(float, ys))))
    return out


def series_point_f1(
    answer: list[dict], truth_series: list[list], label: dict, tau: float = 0.02
) -> float:
    pred = _curves([(c["x"], c["y"]) for c in answer])
    gt = _curves([([p[0] for p in s], [p[1] for p in s]) for s in truth_series])
    return evaluate_points(pred, gt, _frame(label), tau).point_f1
