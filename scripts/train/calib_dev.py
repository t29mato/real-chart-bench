"""Development sets for the automatic axis calibration (方式C / 主条件1,
docs/design/local-model.md「自動校正の改善」). Nothing here reads the
benchmark: options are developed and chosen on these sets, the benchmark is
only measured afterwards (scripts/eval/auto_calibration_check.py, run_hybrid.py).

Sets (truth = each label's own first and last tick per axis):
  val_synth   synth-materials validation split (5%, by image)
  stress      synth-stress-val (the detector's stress validation set)
  chartinfo   CHART-Info 2024 line charts with numeric ticks (PubMed Central;
              measured papers excluded when the data were built)
  inverted    val_synth with the colours inverted (white-on-black figures)
  lowres      val_synth shrunk to a 600 px long side and saved as JPEG q 60
              (small, blurry embedded images)
  hires       val_synth enlarged to a 2500 px long side (very large images)

A figure is correct when both axes agree with the label within 1% of its
tick span (domain.tick_calibration.axis_agreement), wrong when the
calibration claims success but an axis disagrees, unreadable otherwise.

  .venv/bin/python scripts/train/calib_dev.py [--n 300] [--v3] [--normalize --max-side N]

Choice rule (fixed before measuring): the most correct - 2 x wrong summed
over all sets (a wrong calibration silently costs points; unreadable is
visible and can be handed to a person).
"""

from __future__ import annotations

import argparse
import io
import json
import math
import random
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.auto_axis_calibration import (  # noqa: E402
    calibrate_image,
    load_rgb,
)
from real_chart_bench.domain.marker_detection import is_validation, split_key  # noqa: E402
from real_chart_bench.domain.tick_calibration import axis_agreement  # noqa: E402

TRAIN = Path.home() / ".cache/real-chart-bench/train-data"
OUT = Path.home() / ".cache/real-chart-bench/calib-dev"
OPTS: dict = {}
VARIANTS = {"inverted": "val_synth", "lowres": "val_synth", "hires": "val_synth"}


def _labels(source: str, val_only: bool) -> list[dict]:
    root = TRAIN / source
    labs = [json.loads(line) for line in (root / "labels.jsonl").read_text().splitlines()
            if line.strip()]
    labs = [dict(lab, _root=str(root)) for lab in labs if lab.get("axes")]
    if val_only:
        labs = [lab for lab in labs if is_validation(split_key(lab), 0.05)]
    random.Random(0).shuffle(labs)
    return labs


def transform(rgb: np.ndarray, variant: str | None) -> tuple[np.ndarray, float]:
    """The image of a variant set and its scale relative to the label."""
    if variant == "inverted":
        return 255 - rgb, 1.0
    h, w = rgb.shape[:2]
    if variant in ("lowres", "hires"):
        s = (600 if variant == "lowres" else 2500) / max(w, h)
        im = Image.fromarray(rgb).resize((round(w * s), round(h * s)), Image.Resampling.BILINEAR)
        if variant == "lowres":
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=60)
            im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
        return np.asarray(im), s
    return rgb, 1.0


def one(item):
    lab, variant = item
    rgb = load_rgb(Path(lab["_root"]) / lab["image"])
    if rgb is None:
        return "unreadable"
    rgb, s = transform(rgb, variant)
    c = calibrate_image(rgb, **OPTS)
    if c is None or not c.ok:
        return "unreadable"
    for ax in ("x", "y"):
        t = sorted(lab["axes"][ax]["ticks"], key=lambda t: t["px"])
        truth = [{"px": t[0]["px"] * s, "value": t[0]["value"]},
                 {"px": t[-1]["px"] * s, "value": t[-1]["value"]}]
        e = axis_agreement(c.x_fit if ax == "x" else c.y_fit, truth, lab["axes"][ax]["scale"])
        if math.isinf(e) or e > 0.01:
            return "wrong"
    return "correct"


def _init(opts: dict) -> None:  # spawned workers (macOS) do not inherit OPTS
    OPTS.update(opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--v3", action="store_true")
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--max-side", type=int, default=None)
    ap.add_argument("--min-side", type=int, default=None)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--sets", nargs="*", default=None)
    args = ap.parse_args()
    OPTS.update(v3=args.v3, normalize=args.normalize, max_side=args.max_side,
                min_side=args.min_side)
    base = {"val_synth": _labels("synth-materials", True)[: args.n],
            "stress": _labels("synth-stress-val", False)[: args.n],
            "chartinfo": _labels("chartinfo-line", False)[: args.n]}
    jobs = {name: [(lab, None) for lab in labs] for name, labs in base.items()}
    jobs |= {v: [(lab, v) for lab in base[src]] for v, src in VARIANTS.items()}
    if args.sets:
        jobs = {k: v for k, v in jobs.items() if k in args.sets}
    out = {"options": OPTS}
    with Pool(12, initializer=_init, initargs=(dict(OPTS),)) as pool:
        for name, items in jobs.items():
            res = pool.map(one, items, chunksize=4)
            out[name] = {k: res.count(k) for k in ("correct", "wrong", "unreadable")}
            print(name, out[name], flush=True)
    if args.tag:
        OUT.mkdir(parents=True, exist_ok=True)
        (OUT / f"{args.tag}.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
