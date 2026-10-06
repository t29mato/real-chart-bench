"""Point F1 (tau 2%, the benchmark's primary metric) on a run's held-out
training examples: worker_ft.py val mode writes val.jsonl with each answer
and its label; this scores them. For tuning approach B without the benchmark.

The axis frame is the extent of the label's ticks and points (the benchmark
uses the registry's axis range; labels have no such field), log10 on a log
axis. Answers are parsed the way score_llm_predictions.parse_curves does.

Usage: python score_val.py <val.jsonl> [...]
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys
from collections import defaultdict

REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.domain.curve import Curve, ScaleType  # noqa: E402
from real_chart_bench.domain.point_metrics import AxisFrame, evaluate_points  # noqa: E402

TAU = 0.02


def _curves(raw, x_scale) -> list[Curve]:
    out = []
    for c in raw or []:
        if not isinstance(c, dict):
            continue
        pairs = [(float(x), float(y)) for x, y in zip(c.get("x") or [], c.get("y") or [],
                                                      strict=False)
                 if isinstance(x, int | float) and isinstance(y, int | float)]
        if len(pairs) < 2:
            continue
        out.append(Curve(x_values=tuple(p[0] for p in pairs),
                         y_values=tuple(p[1] for p in pairs), x_scale=x_scale))
    return out


def _gt_curves(answer, x_scale) -> list[Curve]:
    return [Curve(x_values=tuple(s["x"]), y_values=tuple(s["y"]), x_scale=x_scale)
            for s in answer if s["x"]]


def _range(vals, scale):
    vals = [v for v in vals if math.isfinite(v) and (scale is ScaleType.LINEAR or v > 0)]
    lo, hi = min(vals), max(vals)
    if lo == hi:
        lo, hi = (lo - 1, hi + 1) if scale is ScaleType.LINEAR else (lo / 10, hi * 10)
    return lo, hi


def score(path: pathlib.Path) -> dict:
    by_cond = defaultdict(list)
    for line in path.read_text().splitlines():
        rec = json.loads(line)
        ans = rec["answer"]
        # labels do not carry the scale here; infer log when the values span decades
        xs = [v for s in ans for v in s["x"]]
        ys = [v for s in ans for v in s["y"]]

        def scale_of(v):
            pos = [a for a in v if a > 0]
            return (ScaleType.LOG if len(pos) == len(v) and max(pos) / min(pos) > 1e3
                    else ScaleType.LINEAR)

        sx, sy = scale_of(xs), scale_of(ys)
        frame = AxisFrame(x_range=_range(xs, sx), y_range=_range(ys, sy), x_scale=sx, y_scale=sy)
        ev = evaluate_points(_curves(rec.get("parsed"), sx), _gt_curves(ans, sx), frame, TAU)
        by_cond[rec.get("condition_of_example", "?")].append(ev.point_f1)
        by_cond["all"].append(ev.point_f1)
    return {k: {"n": len(v), "point_f1": round(statistics.mean(v), 4)} for k, v in by_cond.items()}


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(p, json.dumps(score(pathlib.Path(p))))
