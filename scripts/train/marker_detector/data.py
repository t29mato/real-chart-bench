"""labels.jsonl (docs/design/local-model.md) -> training crops and targets.

Each label directory holds labels.jsonl with image paths relative to it.
Every label passes validate_label and the benchmark-paper exclusion before it
is used; the validation split (domain.marker_detection.is_validation) is
drawn by paper for real figures, by image otherwise.
"""

from __future__ import annotations

import io
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageFilter
from torch.utils.data import Dataset

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from model import MARKER_CLASSES, STRIDE  # noqa: E402

from real_chart_bench.domain.marker_detection import (  # noqa: E402
    Letterbox,
    is_validation,
    split_key,
)
from real_chart_bench.domain.training_data import (  # noqa: E402
    assert_no_benchmark_leak,
    validate_label,
)

CLASS_INDEX = {c: i for i, c in enumerate(MARKER_CLASSES)}


def benchmark_paper_ids() -> set[str]:
    key = json.loads((REPO / "data/llm_run_pixcal/_key.json").read_text())
    reg = json.loads((REPO / "data/verified_pairs/registry.json").read_text())
    rows = reg if isinstance(reg, list) else reg.get("pairs") or reg.get("entries") or []
    ids = {str(v["paper_id"]) for v in key.values()}
    ids |= {str(r.get("paper_id")) for r in rows if isinstance(r, dict) and r.get("paper_id")}
    return ids


def load_labels(dirs: list[Path], max_per_dir: int | None = None) -> list[dict]:
    """All usable labels of the given directories, each with `_path` set to
    its image's absolute path. Invalid lines are counted and skipped."""
    out = []
    for d in dirs:
        f = d / "labels.jsonl"
        if not f.exists():
            print(f"  {f}: なし", flush=True)
            continue
        n_bad, n = 0, 0
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            lab = json.loads(line)
            if validate_label(lab):
                n_bad += 1
                continue
            lab["_path"] = str(d / lab["image"])
            out.append(lab)
            n += 1
            if max_per_dir and n >= max_per_dir:
                break
        print(f"  {f}: {n} 件(不正 {n_bad})", flush=True)
    assert_no_benchmark_leak(out, benchmark_paper_ids())
    return out


def split(labels: list[dict], val_fraction: float) -> tuple[list[dict], list[dict]]:
    tr, va = [], []
    for lab in labels:
        (va if is_validation(split_key(lab), val_fraction) else tr).append(lab)
    return tr, va


def open_rgb(path: str) -> Image.Image:
    im = Image.open(path)
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    return im.convert("RGB")


def series_points(lab: dict, require_marker: bool) -> list[tuple[list, str | None]]:
    out = []
    for s in lab["series"]:
        if require_marker and s.get("marker") is None:
            continue
        pts = s.get("points_px") or []
        if pts:
            out.append((pts, s.get("marker")))
    return out


def _augment(im: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:  # global hue rotation keeps series colours distinct
        hsv = np.array(im.convert("HSV"))
        hsv[..., 0] = (hsv[..., 0].astype(int) + rng.randint(0, 255)) % 256
        im = Image.fromarray(hsv, "HSV").convert("RGB")
    if rng.random() < 0.3:
        im = im.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    if rng.random() < 0.5:
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=rng.randint(30, 95))
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    if rng.random() < 0.3:
        a = np.asarray(im).astype(np.float32)
        a = a * rng.uniform(0.8, 1.1) + rng.uniform(-20, 20)
        im = Image.fromarray(a.clip(0, 255).astype(np.uint8))
    return im


class CropDataset(Dataset):
    """Random fixed-size crops of letterboxed images, with CenterNet targets
    on the stride-2 grid."""

    def __init__(
        self,
        labels,
        crop: int = 640,
        long_side: int = 1024,
        scale_jitter=(0.6, 1.4),
        sigma: float = 1.0,
        require_marker: bool = False,
        seed: int = 0,
        max_points: int = 1024,
    ):
        self.labels = labels
        self.crop, self.long_side, self.scale_jitter = crop, long_side, scale_jitter
        self.sigma, self.require_marker = sigma, require_marker
        self.seed, self.max_points = seed, max_points
        self.epoch = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        rng = random.Random(hash((self.seed, self.epoch, i)))
        lab = self.labels[i]
        im = open_rgb(lab["_path"])
        w, h = im.size
        ls = int(self.long_side * rng.uniform(*self.scale_jitter))
        lb = Letterbox.fit(w, h, long_side=ls, stride=STRIDE)
        nw, nh = max(1, round(w * lb.scale)), max(1, round(h * lb.scale))
        im = im.resize((nw, nh), Image.BILINEAR)
        im = _augment(im, rng)
        c = self.crop
        ox = rng.randint(0, max(0, nw - c))
        oy = rng.randint(0, max(0, nh - c))
        canvas = Image.new("RGB", (c, c), (255, 255, 255))
        canvas.paste(im.crop((ox, oy, min(ox + c, nw), min(oy + c, nh))), (0, 0))
        img = torch.from_numpy(np.asarray(canvas).copy()).permute(2, 0, 1).float() / 255

        gh = gw = c // STRIDE
        heat = np.zeros((gh, gw), np.float32)
        rad = int(math.ceil(3 * self.sigma))
        pts = []  # (gy, gx, dx, dy, cls, sid)
        for sid, (series, marker) in enumerate(series_points(lab, self.require_marker)):
            cls = CLASS_INDEX.get(marker, -1) if marker else -1
            for x, y in series:
                px, py = x * lb.scale - ox, y * lb.scale - oy
                if not (0 <= px < min(c, nw - ox) and 0 <= py < min(c, nh - oy)):
                    continue
                gx, gy = px / STRIDE, py / STRIDE
                ix, iy = min(int(gx), gw - 1), min(int(gy), gh - 1)
                y0, y1 = max(0, iy - rad), min(gh, iy + rad + 1)
                x0, x1 = max(0, ix - rad), min(gw, ix + rad + 1)
                yy, xx = np.mgrid[y0:y1, x0:x1]
                g = np.exp(-((xx - ix) ** 2 + (yy - iy) ** 2) / (2 * self.sigma**2))
                heat[y0:y1, x0:x1] = np.maximum(heat[y0:y1, x0:x1], g)
                pts.append((iy, ix, gx - ix, gy - iy, cls, sid))
        rng.shuffle(pts)
        pts = pts[: self.max_points]
        return img, torch.from_numpy(heat)[None], torch.tensor(pts, dtype=torch.float32).view(-1, 6)


def collate(batch):
    imgs = torch.stack([b[0] for b in batch])
    heats = torch.stack([b[1] for b in batch])
    pts = [torch.cat([torch.full((len(b[2]), 1), float(i)), b[2]], 1) for i, b in enumerate(batch)]
    return imgs, heats, torch.cat(pts) if pts else torch.zeros(0, 7)
