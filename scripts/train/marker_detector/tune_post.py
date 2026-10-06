"""Choose the detector post-processing (方式C) on validation data only:
the training data's validation split (synthetic 400 + Starrydata papers
20%) and the stress validation set (gen_stress_val.py). Reads the detection
cache of cache_dets.py; the benchmark rows in it are not looked at.

Grid: input size, peak threshold, grouping threshold, same-series duplicate
radius (fraction of the long side) and its embedding gate, and the frame
restriction margin and the frame-edge band (frame from the automatic calibration's frame finder,
adapter/auto_axis_calibration.py). Objective: the mean over the three sets
of the per-set mean point F1 (valmetric.py: the benchmark's metric in
pixels, with the label's plot box as the frame).

  .venv/bin/python scripts/train/marker_detector/tune_post.py --cache <pkl> --out <json>
"""

from __future__ import annotations

import argparse
import itertools
import json
import pickle
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from valmetric import series_point_f1  # noqa: E402

from real_chart_bench.adapter.auto_axis_calibration import calibrate_image, load_rgb  # noqa: E402
from real_chart_bench.domain.marker_detection import (  # noqa: E402
    Detection,
    PostConfig,
    postprocess,
)

SETS = ("val_synth", "val_real", "stress")
CACHE: dict = {}


def frame_of(path: str):
    rgb = load_rgb(Path(path))
    c = calibrate_image(rgb) if rgb is not None else None
    return None if c is None else list(c.frame)


def truth(lab):
    return [s["points_px"] for s in lab["series"] if s.get("points_px")]


def score(job):
    ls, cfg = job
    out = {}
    for name in SETS:
        f1s = []
        for row in CACHE["sets"][name]:
            dets = [Detection(x, y, s, m, tuple(e)) for x, y, s, m, e in row["dets"][ls]]
            ans = postprocess(dets, tuple(row["size"]), row.get("frame"), cfg)
            f1s.append(series_point_f1(ans, truth(row["label"]), row["label"]))
        out[name] = sum(f1s) / max(1, len(f1s))
    out["objective"] = sum(out[n] for n in SETS) / len(SETS)
    return ls, cfg, out


def init(cache):
    CACHE.update(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cache = pickle.loads(Path(args.cache).read_bytes())
    frames_path = Path(args.cache).with_suffix(".frames.json")
    if frames_path.exists():
        frames = json.loads(frames_path.read_text())
    else:
        paths = sorted({r["label"]["_path"] for n in SETS for r in cache["sets"][n]})
        with Pool(16) as pool:
            frames = dict(zip(paths, pool.map(frame_of, paths), strict=True))
        frames_path.write_text(json.dumps(frames))
    for n in SETS:
        for r in cache["sets"][n]:
            r["frame"] = frames.get(r["label"]["_path"])
        print(n, len(cache["sets"][n]), "図、枠あり",
              sum(r["frame"] is not None for r in cache["sets"][n]), flush=True)
    cache["sets"] = {n: cache["sets"][n] for n in SETS}  # never the benchmark

    jobs = []
    for ls, t, g in itertools.product((768, 1024, 1280), (0.3, 0.4, 0.5), (0.5, 0.7)):
        for ssf, gate in [(None, 0.5)] + list(
            itertools.product((0.008, 0.012, 0.016, 0.024), (0.5, 1.0))
        ):
            for fm, eb in itertools.product((None, 0.02, 0.05), (None, 0.01, 0.02)):
                jobs.append((ls, PostConfig(threshold=t, group_threshold=g,
                                            same_series_frac=ssf, embed_gate=gate,
                                            frame_margin=fm, edge_band=eb)))
    with Pool(16, initializer=init, initargs=(cache,)) as pool:
        res = pool.map(score, jobs, chunksize=4)
    res.sort(key=lambda r: -r[2]["objective"])

    def row(r):
        ls, cfg, m = r
        return {"long_side": ls, **cfg.__dict__, **{k: round(v, 4) for k, v in m.items()}}

    base = next(r for r in res if r[0] == 768 and r[1] == PostConfig(threshold=0.4,
                                                                    group_threshold=0.5))
    print("方式A の設定:", json.dumps(row(base)))
    for r in res[:10]:
        print(json.dumps(row(r)))
    best = row(res[0])
    Path(args.out).write_text(json.dumps({"chosen": best, "method_a": row(base),
                                          "top20": [row(r) for r in res[:20]],
                                          "n_configs": len(res)}, indent=1))
    print("選択:", json.dumps(best))


if __name__ == "__main__":
    main()
