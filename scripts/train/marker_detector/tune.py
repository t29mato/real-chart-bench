"""Pick the inference settings (peak threshold, grouping threshold, input
size) on the validation split of the training data -- never on the
benchmark -- and write them to <ckpt dir>/tuned.json for run_benchmark.py."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_labels, split  # noqa: E402
from model import MarkerNet  # noqa: E402
from train import TRAIN_ROOT, evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--root", default=str(TRAIN_ROOT))
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--val-n", type=int, default=400)
    ap.add_argument("--require-marker", action="store_true")
    ap.add_argument("--real-val-fraction", type=float, default=None)
    ap.add_argument(
        "--on", choices=("all", "real", "synthetic"), default="all", help="validation subset"
    )
    args = ap.parse_args()

    labels = load_labels([Path(args.root) / d for d in args.data])
    _, val = split(labels, args.val_fraction, args.real_val_fraction)
    random.Random(0).shuffle(val)
    if args.on != "all":
        val = [lab for lab in val if (lab.get("paper_id") is not None) == (args.on == "real")]
    print(f"検証 {len(val)} 図({args.on})", flush=True)
    model = MarkerNet().cuda()
    model.load_state_dict(torch.load(args.ckpt, map_location="cuda", weights_only=False)["model"])
    grid = {}
    for ls in (768, 1024, 1280):
        m = evaluate(
            model,
            val,
            args.val_n,
            args.require_marker,
            peak_thresholds=(0.15, 0.2, 0.3, 0.4, 0.5),
            group_thresholds=(0.3, 0.5, 0.7, 1.0),
            long_side=ls,
        )
        grid[ls] = m
        print(ls, json.dumps(m), flush=True)
    best = max(((v, ls, k) for ls, m in grid.items() for k, v in m.items() if k.startswith("f1@")))
    v, ls, k = best
    t, g = k[3:].split("/")
    tuned = {
        "long_side": ls,
        "threshold": float(t),
        "group_threshold": float(g),
        "val_series_point_f1": v,
        "val_n": min(args.val_n, len(val)),
        "data": args.data,
        "tuned_on": args.on,
        "grid": grid,
    }
    out = Path(args.ckpt).parent / "tuned.json"
    out.write_text(json.dumps(tuned, indent=1))
    print(f"選択: {k} long_side {ls} -> {v}", flush=True)


if __name__ == "__main__":
    main()
