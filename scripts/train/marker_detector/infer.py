"""Full-image inference: heatmap peaks -> Detections in image pixels -> series."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from model import MARKER_CLASSES, STRIDE  # noqa: E402

from real_chart_bench.domain.marker_detection import (  # noqa: E402
    Detection,
    Letterbox,
    group_into_series,
    pixel_answer,
    suppress_duplicates,
)


@torch.no_grad()
def detect(
    model,
    im: Image.Image,
    long_side: int = 1024,
    threshold: float = 0.3,
    max_dets: int = 2000,
    device: str = "cuda",
) -> list[Detection]:
    w, h = im.size
    lb = Letterbox.fit(w, h, long_side=long_side, stride=32)
    nw, nh = max(1, round(w * lb.scale)), max(1, round(h * lb.scale))
    canvas = Image.new("RGB", (lb.out_w, lb.out_h), (255, 255, 255))
    canvas.paste(im.resize((nw, nh), Image.BILINEAR), (0, 0))
    x = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float()[None] / 255
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
        out = model(x.to(device))
    heat = out["heat"].float().sigmoid()[0, 0]
    peak = heat == F.max_pool2d(heat[None, None], 3, 1, 1)[0, 0]
    # nothing in the white padding
    valid = torch.zeros_like(heat, dtype=torch.bool)
    valid[: (nh + STRIDE - 1) // STRIDE, : (nw + STRIDE - 1) // STRIDE] = True
    mask = peak & (heat >= threshold) & valid
    ys, xs = torch.nonzero(mask, as_tuple=True)
    scores = heat[ys, xs]
    if len(scores) > max_dets:
        top = scores.topk(max_dets).indices
        ys, xs, scores = ys[top], xs[top], scores[top]
    off = out["offset"].float()[0][:, ys, xs]
    cls = out["shape"].float()[0][:, ys, xs].argmax(0)
    emb = out["embed"].float()[0][:, ys, xs].T
    dets = []
    for k in range(len(scores)):
        gx = (xs[k].item() + off[0, k].item()) * STRIDE
        gy = (ys[k].item() + off[1, k].item()) * STRIDE
        ix, iy = lb.to_image(gx, gy)
        dets.append(
            Detection(
                x=round(ix, 2),
                y=round(iy, 2),
                score=round(scores[k].item(), 4),
                marker=MARKER_CLASSES[cls[k].item()],
                embedding=tuple(round(v, 4) for v in emb[k].tolist()),
            )
        )
    return dets


def to_answer(
    dets: list[Detection],
    image_size,
    dup_radius_frac: float = 0.004,
    group_threshold: float = 0.5,
    min_points: int = 2,
) -> list[dict]:
    radius = max(1.5, dup_radius_frac * max(image_size))
    kept = suppress_duplicates(dets, radius)
    return pixel_answer(group_into_series(kept, group_threshold), min_points=min_points)
