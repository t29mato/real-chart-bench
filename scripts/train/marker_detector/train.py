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
from model import MarkerNet, embedding_loss, focal_loss  # noqa: E402
from valmetric import series_point_f1  # noqa: E402

from real_chart_bench.domain.marker_detection import pixel_point_f1  # noqa: E402

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
    ap.add_argument("--init", default=None, help="start from this checkpoint's weights")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    labels = load_labels([Path(args.root) / d for d in args.data], args.max_per_dir)
    train, val = split(labels, args.val_fraction)
    random.Random(0).shuffle(val)
    print(f"学習 {len(train)} / 検証 {len(val)}", flush=True)

    model = MarkerNet(pretrained_path=str(WEIGHTS)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warm = 500

    def lr_at(step):
        if step < warm:
            return (step + 1) / warm
        return 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, args.steps - warm)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    step, best = 0, -1.0
    log = []
    ck = out / "last.pt"
    if ck.exists():
        s = torch.load(ck, map_location="cuda", weights_only=False)
        model.load_state_dict(s["model"])
        opt.load_state_dict(s["opt"])
        sched.load_state_dict(s["sched"])
        step, best, log = s["step"], s["best"], s.get("log", [])
        print(f"再開: step {step}, best {best}", flush=True)
    elif args.init:
        model.load_state_dict(
            torch.load(args.init, map_location="cuda", weights_only=False)["model"]
        )
        print(f"初期値: {args.init}", flush=True)

    ds = CropDataset(train, crop=args.crop, require_marker=args.require_marker)

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
        for imgs, heats, pts in dl:
            imgs, heats, pts = imgs.cuda(non_blocking=True), heats.cuda(), pts.cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                o = model(imgs)
            bi, gy, gx = pts[:, 0].long(), pts[:, 1].long(), pts[:, 2].long()
            l_heat = focal_loss(o["heat"].float(), heats)
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
            else:
                l_off = l_cls = l_pull = l_push = l_heat * 0
            loss = l_heat + l_off + 0.2 * l_cls + args.embed_weight * (l_pull + l_push)
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
                    f"push {l_push.item():.3f} lr {sched.get_last_lr()[0]:.2e} "
                    f"{(time.time() - t0) / 50:.2f}s/it",
                    flush=True,
                )
                t0 = time.time()
            if step % args.eval_every == 0 or step == args.steps:
                m = evaluate(model, val, args.val_n, args.require_marker)
                score = max(v for k, v in m.items() if k.startswith("f1@"))
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
    print(f"完了 step {step} best {best}", flush=True)


if __name__ == "__main__":
    main()
