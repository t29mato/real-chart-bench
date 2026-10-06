"""方式C: how often the automatic axis calibration (frame rules + Tesseract
tick OCR, adapter/auto_axis_calibration.py) agrees with the person's
calibration (data/verified_pairs/tick_calibration.json) on the scored
figures. Measurement only -- nothing here is tuned on the benchmark.

An axis agrees when, at both of the person's ticks, the automatic line gives
the person's value within `--tol` of the person's tick span (in decades on a
log axis); a figure succeeds when both axes agree.

  .venv/bin/python scripts/eval/auto_calibration_check.py [--no-superscripts] --out <json>
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.auto_axis_calibration import (  # noqa: E402
    calibrate_image,
    load_rgb,
)
from real_chart_bench.adapter.tick_plot_areas import load_tick_calibration  # noqa: E402
from real_chart_bench.domain.tick_calibration import axis_agreement  # noqa: E402

OPTS: dict = {}


def one(item):
    fig, k, cal = item
    rgb = load_rgb(REPO / k["image_path"])
    rec = {"fig": fig, "paper_figure": f"{k['paper_id']}-{k['figure_id']}",
           "x_scale": cal["x_scale"], "y_scale": cal["y_scale"]}
    c = calibrate_image(rgb, **OPTS) if rgb is not None else None
    if c is None:
        rec.update(frame=None, x_err=None, y_err=None, x_fit=None, y_fit=None)
        return rec
    rec["frame"] = [round(v, 1) for v in c.frame]
    for ax, fit in (("x", c.x_fit), ("y", c.y_fit)):
        e = axis_agreement(fit, cal[ax], cal[f"{ax}_scale"])
        rec[f"{ax}_err"] = None if math.isinf(e) else round(e, 5)
        rec[f"{ax}_fit"] = None if fit is None else {
            "scale": fit.scale, "family": fit.family, "n_ticks": len(fit.ticks),
            "ticks": [[round(p, 1), v] for p, v in fit.ticks]}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-superscripts", action="store_true")
    ap.add_argument("--no-right-axes", action="store_true")
    ap.add_argument("--no-split-merged", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    OPTS.update(superscripts=not args.no_superscripts, right_axes=not args.no_right_axes,
                split_merged=not args.no_split_merged)
    key = json.loads((REPO / "data/llm_run_pixcal/_key.json").read_text())
    cals = load_tick_calibration(REPO / "data/verified_pairs/tick_calibration.json")
    items = [(f, k, cals[(k["paper_id"], k["figure_id"])]) for f, k in sorted(key.items())]
    with Pool(16) as pool:
        recs = pool.map(one, items)

    def ok(e, tol):
        return e is not None and e <= tol

    summary = {}
    for tol in (0.005, 0.01, 0.02):
        summary[str(tol)] = {
            "x": sum(ok(r["x_err"], tol) for r in recs),
            "y": sum(ok(r["y_err"], tol) for r in recs),
            "both": sum(ok(r["x_err"], tol) and ok(r["y_err"], tol) for r in recs),
        }
    logy = [r for r in recs if r["y_scale"] == "log"]
    summary["n"] = len(recs)
    summary["frame_found"] = sum(r["frame"] is not None for r in recs)
    summary["log_y"] = {"n": len(logy), "y_ok@0.01": sum(ok(r["y_err"], 0.01) for r in logy)}
    summary["options"] = dict(OPTS)
    args.out.write_text(json.dumps({"summary": summary, "figures": recs}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
