"""方式C: the automatic axis calibration (adapter/auto_axis_calibration.py)
on the synthetic training data's validation split -- where its options are
developed and chosen (the benchmark is only measured,
scripts/eval/auto_calibration_check.py).

The truth is each label's own ticks (matplotlib's tick positions and printed
values); agreement is domain.tick_calibration.axis_agreement at the first
and last tick of each axis.

  .venv/bin/python scripts/train/calib_val.py --source synth-materials --n 300 [--log-only]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.auto_axis_calibration import (  # noqa: E402
    calibrate_image,
    load_rgb,
)
from real_chart_bench.domain.marker_detection import is_validation, split_key  # noqa: E402
from real_chart_bench.domain.tick_calibration import axis_agreement  # noqa: E402

TRAIN = Path.home() / ".cache/real-chart-bench/train-data"
OPTS: dict = {}


def one(item):
    root, lab = item
    rgb = load_rgb(root / lab["image"])
    c = calibrate_image(rgb, **OPTS) if rgb is not None else None
    rec = {"image": lab["image"], "y_scale": lab["axes"]["y"]["scale"],
           "x_scale": lab["axes"]["x"]["scale"]}
    for ax in ("x", "y"):
        t = sorted(lab["axes"][ax]["ticks"], key=lambda t: t["px"])
        person = [t[0], t[-1]]
        fit = None if c is None else (c.x_fit if ax == "x" else c.y_fit)
        e = axis_agreement(fit, person, lab["axes"][ax]["scale"])
        rec[f"{ax}_err"] = None if math.isinf(e) else e
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synth-materials")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--log-only", action="store_true")
    ap.add_argument("--no-superscripts", action="store_true")
    ap.add_argument("--no-right-axes", action="store_true")
    ap.add_argument("--no-split-merged", action="store_true")
    args = ap.parse_args()
    OPTS.update(superscripts=not args.no_superscripts, right_axes=not args.no_right_axes,
                split_merged=not args.no_split_merged)
    root = TRAIN / args.source
    labs = [json.loads(line) for line in (root / "labels.jsonl").read_text().splitlines()
            if line.strip()]
    labs = [lab for lab in labs if lab.get("axes") and is_validation(split_key(lab), 0.05)]
    if args.log_only:
        labs = [lab for lab in labs if "log" in (lab["axes"]["x"]["scale"],
                                                  lab["axes"]["y"]["scale"])]
    random.Random(0).shuffle(labs)
    labs = labs[: args.n]
    with Pool(16) as pool:
        recs = pool.map(one, [(root, lab) for lab in labs])

    def ok(e):
        return e is not None and e <= 0.01

    out = {"n": len(recs), "options": OPTS,
           "x_ok": sum(ok(r["x_err"]) for r in recs), "y_ok": sum(ok(r["y_err"]) for r in recs),
           "both_ok": sum(ok(r["x_err"]) and ok(r["y_err"]) for r in recs)}
    for ax in ("x", "y"):
        logs = [r for r in recs if r[f"{ax}_scale"] == "log"]
        out[f"{ax}_log"] = f"{sum(ok(r[f'{ax}_err']) for r in logs)}/{len(logs)}"
    print(json.dumps(out))


if __name__ == "__main__":
    main()
