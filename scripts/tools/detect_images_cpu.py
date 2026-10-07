"""Raw marker detections for a list of images, on the CPU (no GPU lock), in
the format of export_detector_cache.py: one JSON per image named by its
sha256, low threshold, input long sides 768 / 1024 / 1280. Used for the
development sets of 方式D v3 (docs/design/local-model.md「v3」) -- synthetic
stress cases and CHART-Info line charts -- never for the benchmark.

  RCB_DEVICE=cpu ~/.cache/real-chart-bench/detector/venv/bin/python \\
      scripts/tools/detect_images_cpu.py <list.txt> <out_dir> [--workers 8]

<list.txt>: one image path per line. Images whose JSON exists are skipped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

os.environ.setdefault("RCB_DEVICE", "cpu")
REPO = Path(__file__).resolve().parents[2]
CKPT = Path.home() / ".cache/real-chart-bench/detector/runs/real-b1/best.pt"
LONG_SIDES = (768, 1024, 1280)
THRESHOLD = 0.1

_model = None


def _load():
    global _model
    if _model is None:
        import torch

        sys.path.insert(0, str(REPO / "scripts/train/marker_detector"))
        from model import MarkerNet

        torch.set_num_threads(max(1, int(os.environ.get("RCB_THREADS", "4"))))
        m = MarkerNet().eval()
        m.load_state_dict(torch.load(CKPT, map_location="cpu", weights_only=False)["model"])
        _model = m
    return _model


def one(job: tuple[str, str]) -> str:
    path, out = Path(job[0]), Path(job[1])
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    dst = out / f"{sha}.json"
    if dst.exists():
        return "skip"
    model = _load()
    from infer import detect
    from PIL import Image

    im = Image.open(path).convert("RGB")
    dets = {str(ls): [[d.x, d.y, d.score, d.marker, list(d.embedding)]
                      for d in detect(model, im, long_side=ls, threshold=THRESHOLD,
                                      device="cpu")]
            for ls in LONG_SIDES}
    rec = {"size": list(im.size), "checkpoint": "~/" + str(CKPT.relative_to(Path.home())),
           "raw_threshold": THRESHOLD, "dets": dets}
    dst.write_text(json.dumps(rec) + "\n")
    return "done"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("list")
    ap.add_argument("out")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [p for p in Path(args.list).read_text().split() if p]
    with ProcessPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(one, [(p, str(out)) for p in paths], chunksize=4)):
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(paths)} {r}", flush=True)
    print(f"{len(paths)} images -> {out}")


if __name__ == "__main__":
    main()
