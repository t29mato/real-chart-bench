"""Export the detector's raw detections for the marker_detector tool of 方式D:
one JSON per image, named by the image's sha256, from a cache_dets.py pickle
(scripts/train/marker_detector/cache_dets.py, 方式A (b) weights, low
threshold, input long sides 768 / 1024 / 1280).

  .venv/bin/python scripts/tools/export_detector_cache.py \\
      ~/.cache/real-chart-bench/detector/cache/real-b1-cpu.pkl \\
      ~/.cache/real-chart-bench/orchestrator/dets [--set bench]

Only detections leave: no labels, no ground truth.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.orchestrator_tools import image_sha256  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("--set", default="bench")
    args = ap.parse_args()
    cache = pickle.loads(Path(args.cache).read_bytes())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for row in cache["sets"][args.set]:
        path = Path(row["label"]["_path"])
        rec = {
            "size": list(row["size"]),
            "checkpoint": str(cache["ckpt"]).replace(str(Path.home()), "~"),
            "raw_threshold": cache["threshold"],
            "dets": {str(ls): [[d[0], d[1], d[2], d[3], list(d[4])] for d in dets]
                     for ls, dets in row["dets"].items()},
        }
        (out / f"{image_sha256(path)}.json").write_text(json.dumps(rec) + "\n")
        n += 1
    print(f"{n} images -> {out}")


if __name__ == "__main__":
    main()
