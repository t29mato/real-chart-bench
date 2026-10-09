"""Choose the post-processing of a MarkerNet v2 checkpoint (方式A v2,
docs/design/local-model.md) on validation data only: the training labels'
deterministic validation split exactly as train.py makes it (plotqa,
synth-materials, synth-stress, chartinfo, chartinfo-line-markers; synthetic images 5%, real figures
by paper 20%) and the stress validation set (gen_stress_val.py, generated
here when missing). The benchmark is never read.

Detections are cached per input size at a low threshold; the grid is
threshold x group threshold x size_factor x plot_min x legend_max (dup_frac
default). Objective: the mean over the three sets of the per-set mean point
F1 (valmetric.py). Writes <ckpt dir>/tuned_v2.json: chosen config, the
v1-equivalent baseline (size_factor/plot_min/legend_max off) and the top 20.

  ~/.cache/real-chart-bench/mps-venv/bin/python scripts/train/marker_detector/tune_v2.py \\
      --ckpt <best.pt>
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import subprocess
import sys
import time
from multiprocessing import get_context
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from valmetric import series_point_f1  # noqa: E402

from real_chart_bench.domain.marker_detection import (  # noqa: E402
    PostConfig,
    postprocess,
)

DATA = ("plotqa", "synth-materials", "synth-stress", "chartinfo", "chartinfo-line-markers")
SETS = ("val_synth", "val_real", "stress")
LONG_SIDES = (768, 1024, 1280)
CACHE: dict = {}


def val_sets(n_synth: int) -> dict[str, list[dict]]:
    from data import load_labels, split
    from train import TRAIN_ROOT

    stress_dir = TRAIN_ROOT / "synth-stress-val"
    if not (stress_dir / "labels.jsonl").exists():
        subprocess.run([sys.executable, str(Path(__file__).with_name("gen_stress_val.py"))],
                       check=True)
    _, val = split(load_labels([TRAIN_ROOT / d for d in DATA]), 0.05, 0.2)
    random.Random(0).shuffle(val)
    return {
        "val_synth": [lab for lab in val if lab.get("paper_id") is None][:n_synth],
        "val_real": [lab for lab in val if lab.get("paper_id") is not None],
        "stress": load_labels([stress_dir]),
    }


def detect_all(ckpt: str, sets: dict, threshold: float, limit: int | None) -> dict:
    import torch
    from data import open_rgb
    from infer import DEVICE, detect
    from model import MarkerNet

    ck = torch.load(ckpt, map_location=DEVICE, weights_only=False)
    model = MarkerNet(aux=bool(ck["args"].get("aux"))).to(DEVICE).eval()
    model.load_state_dict(ck["model"])
    if not model.aux:
        sys.exit("not a v2 (aux) checkpoint")
    out = {}
    for name, labs in sets.items():
        t0 = time.time()
        rows = []
        for lab in labs[:limit]:
            im = open_rgb(lab["_path"])
            dets = {ls: detect(model, im, long_side=ls, threshold=threshold)
                    for ls in LONG_SIDES}
            rows.append({"label": lab, "size": im.size, "dets": dets})
        out[name] = rows
        print(f"{name}: {len(rows)} 図 {time.time() - t0:.0f}s", flush=True)
    return out


def truth(lab):
    return [s["points_px"] for s in lab["series"] if s.get("points_px")]


def score(job):
    ls, cfg = job
    out = {}
    for name in SETS:
        f1s = []
        for row in CACHE[name]:
            ans = postprocess(row["dets"][ls], tuple(row["size"]), None, cfg)
            f1s.append(series_point_f1(ans, truth(row["label"]), row["label"]))
        out[name] = sum(f1s) / max(1, len(f1s))
    out["objective"] = sum(out[n] for n in SETS) / len(SETS)
    return ls, cfg, out


def init(cache):
    CACHE.update(cache)


def row(r):
    ls, cfg, m = r
    return {"long_side": ls, **cfg.__dict__, **{k: round(v, 4) for k, v in m.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", default=None, help="default: <ckpt dir>/tuned_v2.json")
    ap.add_argument("--n-synth", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None, help="figures per set (smoke tests)")
    ap.add_argument("--threshold", type=float, default=0.2, help="cached detections' floor")
    args = ap.parse_args()
    cache = detect_all(args.ckpt, val_sets(args.n_synth), args.threshold, args.limit)

    jobs = [
        (ls, PostConfig(threshold=t, group_threshold=g, size_factor=sf, plot_min=pm,
                        legend_max=lm))
        for ls, t, g, sf, pm, lm in itertools.product(
            LONG_SIDES, (0.3, 0.4, 0.5), (0.5, 0.7), (None, 0.4, 0.6, 0.8, 1.0),
            (None, 0.3, 0.5), (None, 0.5))
    ]
    with get_context("spawn").Pool(16, initializer=init, initargs=(cache,)) as pool:
        res = pool.map(score, jobs, chunksize=4)
    res.sort(key=lambda r: -r[2]["objective"])
    base = max((r for r in res if r[1].size_factor is None and r[1].plot_min is None
                and r[1].legend_max is None), key=lambda r: r[2]["objective"])
    for r in res[:10]:
        print(json.dumps(row(r)))
    out = Path(args.out) if args.out else Path(args.ckpt).parent / "tuned_v2.json"
    out.write_text(json.dumps({
        "chosen": row(res[0]), "baseline": row(base), "top20": [row(r) for r in res[:20]],
        "n_configs": len(res), "sets": {n: len(cache[n]) for n in SETS},
        "ckpt": str(args.ckpt).replace(str(Path.home()), "~")}, indent=1))
    print("baseline(v1 相当、領域・大きさなし):", json.dumps(row(base)))
    print("選択:", json.dumps(row(res[0])), "->", out)


if __name__ == "__main__":
    main()
