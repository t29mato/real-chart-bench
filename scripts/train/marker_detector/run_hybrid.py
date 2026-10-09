"""方式C (docs/design/local-model.md「方式C: 分業型」): detector + automatic
axis calibration on the scored figures, both conditions.

- Detections come from the cache of cache_dets.py (the figure's row at the
  chosen input size); post-processing is the setting tune_post.py chose on
  validation data (its JSON), never anything chosen on the benchmark.
- The plot frame and the two axis fits come from
  adapter/auto_axis_calibration.calibrate_image (frame rules + Tesseract).
- Condition 2 (pixcal): pixel answers, converted by the scorer through the
  person's tick calibration -> data/local_model_runs/<run>/pixpts_px.jsonl,
  scored as local-cuda-detector-<run>.
- Condition 1 (noaxis): the same pixels converted by the automatic fits, in
  the printed space (OCR reads the printed labels) ->
  data/local_model_runs/<run>/noaxis.jsonl, scored as local-cuda-hybrid-<run>.
  A figure whose calibration fails gets no answer (scored as a total miss).

  .venv/bin/python scripts/train/marker_detector/run_hybrid.py --cache <pkl> \\
      --tuned <tune_post json> --run <name> --name <display name>

A MarkerNet v2 run: the cache comes from cache_dets.py (--only bench, MPS) and
--tuned is the checkpoint's tuned_v2.json (size / plot / legend rules).
"""

from __future__ import annotations

import argparse
import json
import pickle
import platform
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.auto_axis_calibration import calibrate_image, load_rgb  # noqa: E402
from real_chart_bench.domain.marker_detection import (  # noqa: E402
    Detection,
    PostConfig,
    postprocess,
)

CFG_KEYS = ("threshold", "group_threshold", "dup_frac", "same_series_frac", "embed_gate",
            "edge_band", "frame_margin", "min_points", "size_factor", "plot_min", "legend_max")


def to_values(answer: list[dict], cal) -> list[dict]:
    out = []
    for c in answer:
        out.append({**c,
                    "x": [cal.x_fit.px_to_value(v) for v in c["x"]],
                    "y": [cal.y_fit.px_to_value(v) for v in c["y"]]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--data-note", default="")
    ap.add_argument("--method-a", action="store_true",
                    help="method A's post-processing (the before row), no frame restriction")
    args = ap.parse_args()
    tuned = json.loads(Path(args.tuned).read_text())
    # tuned_v2.json (tune_v2.py) calls the no-v2-rules row "baseline"
    chosen = (tuned.get("method_a") or tuned["baseline"]) if args.method_a else tuned["chosen"]
    cfg = PostConfig(**{k: chosen[k] for k in CFG_KEYS if k in chosen})
    ls = chosen["long_side"]
    cache = pickle.loads(Path(args.cache).read_bytes())
    pix_key = json.loads((REPO / "data/llm_run_pixcal/_key.json").read_text())
    noaxis_key = json.loads((REPO / "data/llm_run_v3/_key.json").read_text())
    noaxis_fig = {(k["paper_id"], k["figure_id"]): f for f, k in noaxis_key.items()}

    out_dir = REPO / "data/local_model_runs" / args.run
    out_dir.mkdir(parents=True, exist_ok=True)
    pix_recs, noaxis_recs = [], []
    for row in cache["sets"]["bench"]:
        lab = row["label"]
        fig = lab["fig"]
        assert pix_key[fig]["figure_id"] == lab["figure_id"]
        t0 = time.time()
        cal = calibrate_image(load_rgb(REPO / lab["image_path"]))
        cal_s = time.time() - t0
        dets = [Detection(d[0], d[1], d[2], d[3], tuple(d[4]), *d[5:]) for d in row["dets"][ls]]
        t1 = time.time()
        ans = postprocess(dets, tuple(row["size"]), cal.frame if cal else None, cfg)
        post_s = time.time() - t1
        det_s = row["seconds"][ls]
        base = {"paper_figure": f"{lab['paper_id']}-{lab['figure_id']}",
                "image_path": lab["image_path"], "error": None}
        pix_recs.append({**base, "fig": fig, "condition": "pixpts_px", "parsed": ans,
                         "n_detections": sum(len(c["x"]) for c in ans),
                         "seconds": round(det_s + post_s + (cal_s if cfg.frame_margin is not None
                                                            else 0.0), 3)})
        rec = {**base, "fig": noaxis_fig[(lab["paper_id"], lab["figure_id"])],
               "condition": "noaxis", "seconds": round(det_s + post_s + cal_s, 3),
               "calibration": None if cal is None else {
                   "frame": [round(v, 1) for v in cal.frame], "y_side": cal.y_side,
                   **{ax: None if f is None else {
                       "scale": f.scale, "family": f.family,
                       "ticks": [[round(p, 1), v] for p, v in f.ticks]}
                      for ax, f in (("x", cal.x_fit), ("y", cal.y_fit))}}}
        if cal is None or not cal.ok:
            rec["parsed"] = None
            rec["error"] = "automatic axis calibration failed"
        else:
            rec["parsed"] = to_values(ans, cal)
        noaxis_recs.append(rec)
        print(f"{fig} {base['paper_figure']}: {len(ans)} 系列, 校正 "
              f"{'ok' if cal and cal.ok else 'なし'}", flush=True)
    (out_dir / "pixpts_px.jsonl").write_text("\n".join(json.dumps(r) for r in pix_recs) + "\n")
    (out_dir / "noaxis.jsonl").write_text("\n".join(json.dumps(r) for r in noaxis_recs) + "\n")
    ck = cache["ckpt"]
    device = cache.get("device", "cpu")  # v1 caches were made on the CPU
    env = {
        "kind": "detector",
        "display_name": args.name,
        "architecture": "MarkerNet: ResNet-34 (ImageNet) + U-Net decoder to stride 2, "
        "CenterNet heatmap + offset + shape class + associative embedding"
        + (" + marker size + plot/legend region (v2)" if "baseline" in tuned else ""),
        "checkpoint": str(ck).replace(str(Path.home()), "~"),
        "data": args.data_note,
        "post_processing": chosen,
        "post_processing_chosen_on": "validation split of the training data + stress "
        "validation set (scripts/train/marker_detector/tune_post.py)",
        "peak_threshold": cfg.threshold,
        "group_threshold": cfg.group_threshold,
        "long_side": ls,
        "validation_best": chosen.get("objective"),
        "calibration":
        "automatic: frame rules + Tesseract tick OCR (adapter/auto_axis_calibration.py)",
        "torch": "see cache_dets.py run",
        "gpu": "Apple M3 Max" if device == "mps" else None,
        "device": device,
        "cpu_threads": 16,
        "python": platform.python_version(),
        "seconds_note": f"detector forward on {device} + post-processing; "
        "noaxis adds the automatic calibration (Tesseract)",
        "run_dir": "data/llm_run_pixcal",
    }
    (out_dir / "env.json").write_text(json.dumps(env, indent=2, ensure_ascii=False) + "\n")
    ok = sum(r["parsed"] is not None for r in noaxis_recs)
    print(f"書き出し: {out_dir}(校正成功 {ok}/{len(noaxis_recs)})", flush=True)


if __name__ == "__main__":
    main()
