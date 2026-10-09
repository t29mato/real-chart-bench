"""Train the marker detector (approach A, docs/design/local-model.md).

Resumable: <out>/last.pt holds model, optimiser, scheduler, step and the
best validation score; rerunning the same command continues. --max-minutes
ends a session cleanly (checkpoint written) so the shared GPU lock is
released between sessions. Run under the lock:

  flock /tmp/rcb-gpu.lock ~/.cache/real-chart-bench/detector/venv/bin/python \
      scripts/train/marker_detector/train.py --data plotqa synth-materials --out <dir>

Validation (the only thing anything is tuned on) is the deterministic split
of the training labels; the benchmark is never read here.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data import CropDataset, collate, load_labels, open_rgb, series_points, split  # noqa: E402
from infer import detect, to_answer  # noqa: E402
from model import MarkerNet, embedding_loss, focal_loss, pick_device, region_loss  # noqa: E402
from valmetric import series_point_f1  # noqa: E402

from real_chart_bench.domain.marker_detection import (  # noqa: E402
    PostConfig,
    pixel_point_f1,
    postprocess,
)

TRAIN_ROOT = Path.home() / ".cache/real-chart-bench/train-data"
WEIGHTS = Path.home() / ".cache/real-chart-bench/detector/weights/resnet34-b627a593.pth"


def evaluate(
    model,
    val,
    n: int,
    require_marker: bool,
    peak_thresholds=(0.2, 0.3, 0.4),
    group_thresholds=(0.5,),
    long_side: int = 1024,
) -> dict:
    """Mean series-matched point F1 (the benchmark's metric in pixel space,
    valmetric.py) over the first n validation labels, per (peak threshold,
    grouping threshold); also pixel F1 ignoring series."""
    model.eval()
    res: dict[str, list[float]] = {}
    for lab in val[:n]:
        im = open_rgb(lab["_path"])
        truth_series = [s for s, _ in series_points(lab, require_marker)]
        truth = [tuple(p) for s in truth_series for p in s]
        dets = detect(model, im, threshold=min(peak_thresholds), long_side=long_side)
        r = max(3.0, 0.01 * max(im.size))
        for t in peak_thresholds:
            kept = [d for d in dets if d.score >= t]
            res.setdefault(f"pix@{t}", []).append(
                pixel_point_f1([(d.x, d.y) for d in kept], truth, r)
            )
            for g in group_thresholds:
                ans = to_answer(kept, im.size, group_threshold=g)
                res.setdefault(f"f1@{t}/{g}", []).append(series_point_f1(ans, truth_series, lab))
                if getattr(model, "aux", False):  # v2 post-processing, fixed default
                    ans = postprocess(kept, im.size, None, PostConfig(
                        threshold=t, group_threshold=g, size_factor=0.6, plot_min=0.5,
                        legend_max=0.5))
                    res.setdefault(f"v2f1@{t}/{g}", []).append(
                        series_point_f1(ans, truth_series, lab))
    model.train()
    return {k: round(sum(v) / max(len(v), 1), 4) for k, v in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True, help="dirs under train-data/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--root", default=str(TRAIN_ROOT))
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--crop", type=int, default=640)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--val-fraction", type=float, default=0.05)
    ap.add_argument("--val-n", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--max-per-dir", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=None)
    ap.add_argument("--require-marker", action="store_true")
    ap.add_argument("--embed-weight", type=float, default=0.5)
    ap.add_argument("--real-val-fraction", type=float, default=None)
    ap.add_argument("--partial-neg-weight", type=float, default=0.25)
    ap.add_argument(
        "--oversample", nargs="*", default=[], help="source=N repeats that source N times"
    )
    ap.add_argument("--init", default=None, help="start from this checkpoint's weights")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--aux", action="store_true", help="v2: size and region heads")
    ap.add_argument("--size-weight", type=float, default=0.5)
    ap.add_argument("--region-weight", type=float, default=0.5)
    ap.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16")
    args = ap.parse_args()
    dev = pick_device()
    amp = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    labels = load_labels([Path(args.root) / d for d in args.data], args.max_per_dir)
    train, val = split(labels, args.val_fraction, args.real_val_fraction)
    random.Random(0).shuffle(val)
    reps = {k: int(v) for k, v in (o.split("=") for o in args.oversample)}
    train = [lab for lab in train for _ in range(reps.get(lab["source"], 1))]
    real_val = [lab for lab in val if lab.get("paper_id") is not None]
    synth_val = [lab for lab in val if lab.get("paper_id") is None]
    print(
        f"学習 {len(train)}(重複込み)/ 検証 合成 {len(synth_val)} 実図 {len(real_val)}"
        f"(論文 {len({lab['paper_id'] for lab in real_val})})",
        flush=True,
    )

    if not WEIGHTS.exists():  # torchvision's ImageNet ResNet-34 (BSD-3)
        WEIGHTS.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(
            "https://download.pytorch.org/models/" + WEIGHTS.name, str(WEIGHTS))
    model = MarkerNet(pretrained_path=str(WEIGHTS), aux=args.aux).to(dev)
    print(f"デバイス {dev}、混合精度 {args.amp}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 500

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, args.steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, best = 0, -1.0
    prior_seconds = 0.0  # GPU wall time of earlier sessions
    t_gpu = time.time()
    log = []
    ck = out / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        step, best, log = s["step"], s["best"], s.get("log", [])
        prior_seconds = s.get("train_seconds", 0.0)
        print(f"再開: step {step}, best {best}", flush=True)
    elif args.init:
        missing, _ = model.load_state_dict(
            torch.load(args.init, map_location=dev, weights_only=False)["model"], strict=False
        )
        if missing:  # a v1 checkpoint has no size/region heads
            print(f"初期値にない層(新規に学習): {sorted({k.split('.')[0] for k in missing})}")
        print(f"初期値: {args.init}", flush=True)

    ds = CropDataset(
        train,
        crop=args.crop,
        require_marker=args.require_marker,
        partial_neg_weight=args.partial_neg_weight,
    )

    def save(path):
        torch.save(
            {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "sched": sched.state_dict(),
                "step": step,
                "best": best,
                "log": log,
                "args": vars(args),
                "train_seconds": prior_seconds + time.time() - t_gpu,
            },
            path,
        )

    model.train()
    stop = False
    while step < args.steps and not stop:
        ds.epoch = step  # new augmentation draw each pass
        dl = DataLoader(
            ds,
            batch_size=args.batch,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate,
            drop_last=True,
            persistent_workers=False,
        )
        t0 = time.time()
        for imgs, heats, pts, neg_w, regions, known in dl:
            imgs, heats, pts = imgs.to(dev), heats.to(dev), pts.to(dev)
            neg_w, regions, known = neg_w.to(dev), regions.to(dev), known.to(dev)
            with torch.autocast(dev, dtype=amp or torch.float32, enabled=amp is not None):
                o = model(imgs)
            bi, gy, gx = pts[:, 0].long(), pts[:, 1].long(), pts[:, 2].long()
            l_heat = focal_loss(o["heat"].float(), heats, neg_weight=neg_w)
            if len(bi):
                off = o["offset"].float()[bi, :, gy, gx]
                l_off = F.l1_loss(off, pts[:, 3:5])
                cls = pts[:, 5].long()
                has = cls >= 0
                l_cls = (
                    F.cross_entropy(o["shape"].float()[bi, :, gy, gx][has], cls[has])
                    if has.any()
                    else l_heat * 0
                )
                sid = pts[:, 6].long()
                l_pull, l_push = embedding_loss(o["embed"], bi, gy, gx, sid)
                sz = pts[:, 7]
                l_size = (
                    F.l1_loss(o["size"].float()[bi, 0, gy, gx][sz >= 0], sz[sz >= 0])
                    if args.aux and (sz >= 0).any()
                    else l_heat * 0
                )
            else:
                l_off = l_cls = l_pull = l_push = l_size = l_heat * 0
            l_reg = region_loss(o["region"].float(), regions, known) if args.aux else l_heat * 0
            loss = (l_heat + l_off + 0.2 * l_cls + args.embed_weight * (l_pull + l_push)
                    + args.size_weight * l_size + args.region_weight * l_reg)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            step += 1
            if step % 50 == 0:
                print(
                    f"step {step} loss {loss.item():.3f} heat {l_heat.item():.3f} "
                    f"off {l_off.item():.3f} cls {l_cls.item():.3f} pull {l_pull.item():.3f} "
                    f"push {l_push.item():.3f} size {l_size.item():.3f} "
                    f"region {l_reg.item():.3f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{(time.time() - t0) / 50:.2f}s/it",
                    flush=True,
                )
                t0 = time.time()
            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(model, synth_val, args.val_n, args.require_marker)
                if real_val:  # the real figures decide when there are any
                    m = {
                        **m,
                        **{
                            f"real_{k}": v
                            for k, v in evaluate(
                                model, real_val, args.val_n, args.require_marker
                            ).items()
                        },
                    }
                pre = "real_" if real_val else ""
                score = max(v for k, v in m.items()
                            if k.startswith(pre + "f1@") or k.startswith(pre + "v2f1@"))
                log.append({"step": step, **m})
                print(f"検証 step {step}: {json.dumps(m)}", flush=True)
                if score > best:
                    best = score
                    save(out / "best.pt")
                save(ck)
            if args.max_minutes and (time.time() - t_start) / 60 > args.max_minutes:
                save(ck)
                print(f"時間切れで中断 step {step}(再実行で再開)", flush=True)
                stop = True
                break
            if step >= args.steps:
                break
    save(ck)
    (out / "train_log.json").write_text(json.dumps(log, indent=1))
    minutes = (prior_seconds + time.time() - t_gpu) / 60
    print(f"完了 step {step} best {best} 学習時間 {minutes:.1f} 分", flush=True)


if __name__ == "__main__":
    main()
