"""Keep the chartinfo line figures whose vertices carry markers (方式A v2,
docs/design/local-model.md「データ」): a line's vertex is not always a marker,
so run a checkpoint and keep a figure when >= 70% of its vertices have a
detection (score >= 0.3) within r = max(3, 1% of the long side) px -- the
rule of the v3 chartinfo dev data. Labels go to train-data/chartinfo-line-markers
(images are relative symlinks into chartinfo-line/images), with
extra.vertices_only false and extra.marker_check = the fraction.

  ~/.cache/real-chart-bench/mps-venv/bin/python \\
      scripts/train/marker_detector/select_marker_lines.py --ckpt <best.pt>
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import load_labels, open_rgb  # noqa: E402
from infer import DEVICE, detect  # noqa: E402
from model import MarkerNet  # noqa: E402
from train import TRAIN_ROOT  # noqa: E402


def marker_fraction(dets, vertices: np.ndarray, r: float) -> float:
    if not len(vertices):
        return 0.0
    if not dets:
        return 0.0
    centres = np.array([[d.x, d.y] for d in dets])
    dist = np.linalg.norm(vertices[:, None, :] - centres[None, :, :], axis=2).min(axis=1)
    return float((dist <= r).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", default="chartinfo-line")
    ap.add_argument("--out", default="chartinfo-line-markers")
    ap.add_argument("--min-fraction", type=float, default=0.7)
    ap.add_argument("--threshold", type=float, default=0.3)
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--limit", type=int, default=None, help="first N figures (smoke tests)")
    args = ap.parse_args()

    labels = load_labels([TRAIN_ROOT / args.src], args.limit)
    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model = MarkerNet(aux=bool((ck.get("args") or {}).get("aux"))).to(DEVICE).eval()
    model.load_state_dict(ck["model"])
    out_dir = TRAIN_ROOT / args.out
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    kept = []
    for i, lab in enumerate(labels):
        im = open_rgb(lab["_path"])
        dets = detect(model, im, long_side=args.long_side, threshold=args.threshold)
        pts = np.array([p for s in lab["series"] for p in s.get("points_px") or []], float)
        frac = marker_fraction(dets, pts.reshape(-1, 2), max(3.0, 0.01 * max(im.size)))
        if frac >= args.min_fraction:
            lab = {k: v for k, v in lab.items() if k != "_path"}
            lab["extra"] = {**(lab.get("extra") or {}), "vertices_only": False,
                            "marker_check": round(frac, 3)}
            link = out_dir / lab["image"]
            if not link.is_symlink() and not link.exists():
                link.symlink_to(os.path.relpath(TRAIN_ROOT / args.src / lab["image"], link.parent))
            kept.append(lab)
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(labels)} 図、採用 {len(kept)}", flush=True)
    with (out_dir / "labels.jsonl").open("w") as f:
        for lab in kept:
            f.write(json.dumps(lab, ensure_ascii=False) + "\n")
    for name in ("LICENSE", "ATTRIBUTION.md"):
        if (TRAIN_ROOT / args.src / name).exists():
            shutil.copy(TRAIN_ROOT / args.src / name, out_dir / name)
    print(f"採用 {len(kept)} / {len(labels)} 図(頂点の {args.min_fraction:.0%} 以上に検出)"
          f" -> {out_dir}")


if __name__ == "__main__":
    main()
