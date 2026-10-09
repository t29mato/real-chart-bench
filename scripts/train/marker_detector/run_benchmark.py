"""Run a trained marker detector on the scored figures (condition 2, pixcal).

Writes data/local_model_runs/<run>/pixpts_px.jsonl -- one record per figure in
the local-run shape the scorer reads (adapter/local_vlm_run.py): `parsed` is
[{label, x, y}] in the image file's pixels, converted to values by the scorer
through the person's tick calibration (condition local-cuda-detector-<run>
in scripts/eval/score_llm_predictions.py). Settings come from the
checkpoint's validation tuning, never from the benchmark. Run under the lock:

  flock /tmp/rcb-gpu.lock ~/.cache/real-chart-bench/detector/venv/bin/python \
      scripts/train/marker_detector/run_benchmark.py --ckpt <best.pt> --run <name>

A MarkerNet v2 checkpoint (train args `aux`) is run on MPS with the
post-processing of tune_v2.py (`tuned_v2.json` next to the checkpoint, or
--tuned), which adds the size and region (plot / legend) rules.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import REPO, open_rgb  # noqa: E402
from infer import DEVICE, detect, to_answer  # noqa: E402
from model import MarkerNet  # noqa: E402

from real_chart_bench.domain.marker_detection import PostConfig, postprocess  # noqa: E402

CFG_KEYS = ("threshold", "group_threshold", "dup_frac", "same_series_frac", "embed_gate",
            "edge_band", "frame_margin", "min_points", "size_factor", "plot_min", "legend_max")


def sync():
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    elif DEVICE == "mps":
        torch.mps.synchronize()


def gpu_name():
    if DEVICE == "cuda":
        return torch.cuda.get_device_name(0)
    if DEVICE == "mps":  # torch has no name query for MPS
        proc = platform.processor()
        return "Apple M3 Max" if proc in ("", "arm") else proc
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--run", required=True, help="model id, e.g. markernet-r34-synth")
    ap.add_argument("--name", required=True, help="display name")
    ap.add_argument("--data-note", default="")
    ap.add_argument("--tuned", default=None, help="tuned_v2.json (default: next to the ckpt)")
    ap.add_argument("--limit", type=int, default=None, help="first N figures (smoke tests)")
    args = ap.parse_args()
    # settings chosen on the training data's validation split (tune.py, v2: tune_v2.py)
    v2_json = Path(args.tuned) if args.tuned else Path(args.ckpt).parent / "tuned_v2.json"
    cfg = None
    if v2_json.exists():
        v2 = json.loads(v2_json.read_text())
        cfg = PostConfig(**{k: v2["chosen"][k] for k in CFG_KEYS})
        tuned = {"threshold": cfg.threshold, "group_threshold": cfg.group_threshold,
                 "long_side": v2["chosen"]["long_side"],
                 "val_series_point_f1": v2["chosen"]["objective"], "val_n": v2.get("sets")}
    else:
        tuned = json.loads((Path(args.ckpt).parent / "tuned.json").read_text())
    args.threshold = tuned["threshold"]
    args.group_threshold = tuned["group_threshold"]
    args.long_side = tuned["long_side"]

    key = json.loads((REPO / "data/llm_run_pixcal/_key.json").read_text())
    out_dir = REPO / "data/local_model_runs" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    t_load = time.time()
    ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    model = MarkerNet(aux=bool((ck.get("args") or {}).get("aux"))).to(DEVICE).eval()
    model.load_state_dict(ck["model"])
    load_s = time.time() - t_load
    # warm-up so the first figure's time is not cuDNN autotuning
    from PIL import Image

    detect(model, Image.new("RGB", (800, 600), "white"), long_side=args.long_side)
    sync()

    recs = []
    for fig, k in sorted(key.items())[: args.limit]:
        t0 = time.time()
        rec = {
            "fig": fig,
            "condition": "pixpts_px",
            "error": None,
            "paper_figure": f"{k['paper_id']}-{k['figure_id']}",
            "image_path": k["image_path"],
        }
        try:
            im = open_rgb(str(REPO / k["image_path"]))
            dets = detect(model, im, long_side=args.long_side, threshold=args.threshold)
            if cfg is None:
                rec["parsed"] = to_answer(dets, im.size, group_threshold=args.group_threshold)
                rec["detections"] = [[d.x, d.y, d.score, d.marker] for d in dets]
            else:
                rec["parsed"] = postprocess(dets, im.size, None, cfg)
                rec["detections"] = [[d.x, d.y, d.score, d.marker, d.size, d.plot, d.legend]
                                     for d in dets]
            rec["n_detections"] = len(dets)
        except Exception as e:  # recorded, scored as a total miss
            rec["error"] = repr(e)
            rec["parsed"] = None
        sync()
        rec["seconds"] = round(time.time() - t0, 3)
        recs.append(rec)
        print(
            f"{fig} {rec['paper_figure']}: {len(rec.get('parsed') or [])} 系列 "
            f"{rec.get('n_detections')} 点 {rec['seconds']}s",
            flush=True,
        )
    (out_dir / "pixpts_px.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n")
    env = {
        "kind": "detector",
        "display_name": args.name,
        "architecture": "MarkerNet: ResNet-34 (ImageNet) + U-Net decoder to stride 2, "
        "CenterNet heatmap + offset + shape class + associative embedding"
        + (" + marker size + plot/legend region (v2)" if cfg else ""),
        "checkpoint": str(args.ckpt).replace(str(Path.home()), "~"),
        "checkpoint_step": ck.get("step"),
        "train_args": ck.get("args"),
        "validation_best": tuned["val_series_point_f1"],
        "validation_n": tuned["val_n"],
        "data": args.data_note,
        "post_processing": None if cfg is None else cfg.__dict__,
        "peak_threshold": args.threshold,
        "group_threshold": args.group_threshold,
        "long_side": args.long_side,
        "torch": torch.__version__,
        "gpu": gpu_name(),
        "device": DEVICE,
        "cpu_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "load_seconds": round(load_s, 1),
        "run_dir": "data/llm_run_pixcal",
    }
    (out_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    print(f"書き出し: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
