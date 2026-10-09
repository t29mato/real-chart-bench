"""Run the detector once, at a low peak threshold and several input sizes,
and keep every detection (方式C, docs/design/local-model.md). The
post-processing (threshold, duplicate suppression, frame restriction,
grouping) is then chosen on CPU from this cache (tune_post.py), with one
short GPU hold instead of one per setting.

Sets: the training data's validation split (synthetic: images 5%, first 400
after a fixed shuffle; Starrydata: papers 20%), the stress validation set
(gen_stress_val.py), and the scored benchmark figures -- the last only so
the chosen setting can be scored later without the GPU; nothing is chosen
on them.

  (v2 checkpoint, MPS: add `--only bench` to cache just the scored figures)
  flock /tmp/rcb-gpu.lock ~/.cache/real-chart-bench/detector/venv/bin/python \\
      scripts/train/marker_detector/cache_dets.py --ckpt <best.pt> --out <cache.pkl>
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import REPO, load_labels, open_rgb, split  # noqa: E402
from infer import DEVICE, detect  # noqa: E402
from model import MarkerNet  # noqa: E402
from train import TRAIN_ROOT  # noqa: E402


def val_sets(n_synth: int) -> dict[str, list[dict]]:
    labels = load_labels([TRAIN_ROOT / d for d in ("plotqa", "synth-materials", "starrydata")])
    _, val = split(labels, 0.05, 0.2)
    random.Random(0).shuffle(val)
    synth = [lab for lab in val if lab.get("paper_id") is None][:n_synth]
    real = [lab for lab in val if lab.get("paper_id") is not None]
    stress = load_labels([TRAIN_ROOT / "synth-stress-val"])
    return {"val_synth": synth, "val_real": real, "stress": stress}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-synth", type=int, default=400)
    ap.add_argument("--long-sides", type=int, nargs="+", default=[768, 1024, 1280])
    ap.add_argument("--threshold", type=float, default=0.1)
    ap.add_argument("--only", nargs="*", default=None,
                    help="re-run only these sets, merged into an existing --out cache")
    ap.add_argument("--limit", type=int, default=None, help="figures per set (smoke tests)")
    args = ap.parse_args()

    sets = val_sets(args.n_synth)
    key = json.loads((REPO / "data/llm_run_pixcal/_key.json").read_text())
    sets["bench"] = [
        {"_path": str(REPO / k["image_path"]), "fig": f, "paper_id": k["paper_id"],
         "figure_id": k["figure_id"], "image_path": k["image_path"]}
        for f, k in sorted(key.items())
    ]
    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model = MarkerNet(aux=bool((ck.get("args") or {}).get("aux"))).to(DEVICE).eval()  # v2: aux
    model.load_state_dict(ck["model"])
    from PIL import Image

    detect(model, Image.new("RGB", (800, 600), "white"))
    cache: dict = {"ckpt": args.ckpt, "threshold": args.threshold, "sets": {}, "device": DEVICE}
    if args.only and Path(args.out).exists():
        cache = pickle.loads(Path(args.out).read_bytes())
        sets = {k: v for k, v in sets.items() if k in args.only}
    for name, labs in sets.items():
        t0 = time.time()
        rows = []
        for lab in labs[: args.limit]:
            im = open_rgb(lab["_path"])
            row = {"label": lab, "size": im.size, "dets": {}, "seconds": {}}
            for ls in args.long_sides:
                t1 = time.time()
                ds = detect(model, im, long_side=ls, threshold=args.threshold)
                if DEVICE == "cuda":
                    torch.cuda.synchronize()
                elif DEVICE == "mps":
                    torch.mps.synchronize()
                row["seconds"][ls] = time.time() - t1
                # v2 detections carry size / plot / legend as three more fields
                row["dets"][ls] = [(d.x, d.y, d.score, d.marker, d.embedding)
                                   + ((d.size, d.plot, d.legend) if d.size is not None else ())
                                   for d in ds]
            rows.append(row)
        cache["sets"][name] = rows
        print(f"{name}: {len(rows)} 図 {time.time() - t0:.1f}s", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(cache, f)
    print(f"書き出し: {args.out}", flush=True)


if __name__ == "__main__":
    main()
