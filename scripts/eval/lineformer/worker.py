"""LineFormer inference worker -- runs under LineFormer's own Python 3.10 env
(scripts/eval/lineformer/setup_local.sh), never under this package's env.

Loads the pretrained model once, then for every image in --manifest (JSONL of
{"figure_id", "image_path", "image_key"}) appends one record of raw
pixel-space series to --output. Already-recorded figure_ids are skipped, so an
interrupted run resumes where it stopped.

Progress goes to stdout one line per figure ("[k/N] figure ok series=3
0.4s eta 1m02s") so `tail -f` on the run log shows where the run is.
"""

import argparse
import json
import sys
import time
import traceback


def _fmt_secs(secs: float) -> str:
    secs = int(round(secs))
    return f"{secs // 60}m{secs % 60:02d}s"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineformer-src", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    sys.path.insert(0, args.lineformer_src)
    import cv2
    import infer as lineformer_infer
    import torch

    with open(args.manifest) as f:
        manifest = [json.loads(line) for line in f if line.strip()]
    done = set()
    try:
        with open(args.output) as f:
            done = {json.loads(line)["figure_id"] for line in f if line.strip()}
    except FileNotFoundError:
        pass
    todo = [m for m in manifest if m["figure_id"] not in done]

    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    print(
        f"[start] {len(manifest)} figures, {len(done)} already done, {len(todo)} to run; "
        f"device={args.device} gpu={gpu} torch={torch.__version__}",
        flush=True,
    )
    t_load = time.time()
    lineformer_infer.load_model(args.config, args.checkpoint, args.device)
    print(f"[model] loaded in {time.time() - t_load:.1f}s", flush=True)

    t_run = time.time()
    with open(args.output, "a") as out:
        for i, item in enumerate(todo, 1):
            t0 = time.time()
            record = {
                "figure_id": item["figure_id"],
                "image_key": item["image_key"],
                "width": 0,
                "height": 0,
                "series": [],
                "error": None,
            }
            try:
                img = cv2.imread(item["image_path"])
                if img is None:
                    raise RuntimeError(f"cv2.imread returned None for {item['image_path']!r}")
                record["height"], record["width"] = int(img.shape[0]), int(img.shape[1])
                # to_clean=False: to_clean=True needs a PMC-format annotation
                # dict we never have (design §7.37).
                series_list = lineformer_infer.get_dataseries(img, to_clean=False)
                record["series"] = [
                    [[float(pt["x"]), float(pt["y"])] for pt in points] for points in series_list
                ]
            except Exception:  # noqa: BLE001 -- one figure must not kill the run
                record["error"] = traceback.format_exc()
            record["seconds"] = round(time.time() - t0, 3)
            out.write(json.dumps(record) + "\n")
            out.flush()

            elapsed = time.time() - t_run
            eta = elapsed / i * (len(todo) - i)
            status = "ok" if record["error"] is None else "ERROR"
            print(
                f"[{len(done) + i}/{len(manifest)}] {item['figure_id']} {status} "
                f"series={len(record['series'])} {record['seconds']:.1f}s "
                f"eta {_fmt_secs(eta)}",
                flush=True,
            )
            if record["error"] is not None:
                print(record["error"].rstrip().splitlines()[-1], flush=True)

    print(f"[done] {len(todo)} figures in {_fmt_secs(time.time() - t_run)}", flush=True)


if __name__ == "__main__":
    main()
