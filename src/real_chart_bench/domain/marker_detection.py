"""Marker detector, approach A of the local extractor (docs/design/local-model.md).

The network (scripts/train/marker_detector/) finds marker centres as heatmap
peaks; everything around it that is plain logic lives here so it is tested:

- Letterbox: the image <-> network-input geometry (resize the long side,
  pad right/bottom up to the stride), so a centre maps back exactly.
- is_validation / split_key: the deterministic validation split carved from
  the training data. Real figures split by paper, so a validation figure
  never shares a paper with a training figure. All tuning happens on this
  split; the benchmark is only ever scored.
- suppress_duplicates / group_into_series / pixel_answer: peaks -> series of
  pixel points in the answer shape the two-stage pixcal scorer converts
  through the person's tick calibration (adapter/tick_plot_areas).
- pixel_point_f1: point F1 in pixels for the validation split.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Letterbox:
    scale: float
    out_w: int
    out_h: int

    @classmethod
    def fit(cls, width: int, height: int, long_side: int, stride: int) -> Letterbox:
        if width <= 0 or height <= 0:
            raise ValueError(f"empty image {width}x{height}")
        scale = long_side / max(width, height)

        def up(v: float) -> int:
            return int(math.ceil(round(v, 6) / stride) * stride)

        return cls(scale=scale, out_w=up(width * scale), out_h=up(height * scale))

    def to_input(self, x: float, y: float) -> tuple[float, float]:
        return x * self.scale, y * self.scale

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        return x / self.scale, y / self.scale


def split_key(label: dict) -> str:
    """What the validation split is drawn over: the paper for a real figure
    (its figures stay together), the image for a synthetic one."""
    if label.get("paper_id") is not None:
        return f"paper:{label['paper_id']}"
    return f"image:{label.get('source')}/{label['image']}"


def is_validation(key: str, fraction: float) -> bool:
    if fraction <= 0:
        return False
    if fraction >= 1:
        return True
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")
    return h / 2**64 < fraction


@dataclass(frozen=True)
class Detection:
    """One marker centre in image pixels, with the network's score, its
    marker-shape class and its grouping embedding."""

    x: float
    y: float
    score: float
    marker: str
    embedding: tuple[float, ...]


def suppress_duplicates(dets: list[Detection], radius: float) -> list[Detection]:
    """Strongest-first: drop any detection strictly closer than `radius` to one
    already kept (two shape classes firing on one marker). Order of the kept
    ones follows the input."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(math.hypot(d.x - k.x, d.y - k.y) >= radius for k in kept):
            kept.append(d)
    keep = {id(d) for d in kept}
    return [d for d in dets if id(d) in keep]


def _dist(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return math.sqrt(sum((p - q) ** 2 for p, q in zip(a, b, strict=True)))


def group_into_series(dets: list[Detection], threshold: float) -> list[list[Detection]]:
    """Leader clustering on the embedding: strongest detection first, each
    joins the nearest group whose running-mean embedding lies within
    `threshold`, else starts a group. Largest group first."""
    groups: list[list[Detection]] = []
    centres: list[list[float]] = []
    for d in sorted(dets, key=lambda d: -d.score):
        best, best_dist = None, threshold
        for i, c in enumerate(centres):
            dist = _dist(d.embedding, tuple(c))
            if dist <= best_dist:
                best, best_dist = i, dist
        if best is None:
            groups.append([d])
            centres.append(list(d.embedding))
        else:
            g = groups[best]
            g.append(d)
            n = len(g)
            centres[best] = [
                c + (e - c) / n for c, e in zip(centres[best], d.embedding, strict=True)
            ]
    return sorted(groups, key=len, reverse=True)


def pixel_answer(groups: list[list[Detection]], min_points: int = 1) -> list[dict]:
    """Series in the scorer's answer shape, in image pixels, points by x."""
    out = []
    for g in groups:
        if len(g) < min_points:
            continue
        pts = sorted(g, key=lambda d: (d.x, d.y))
        out.append(
            {
                "label": f"series_{len(out) + 1}",
                "x": [d.x for d in pts],
                "y": [d.y for d in pts],
            }
        )
    return out


def pixel_point_f1(
    pred: list[tuple[float, float]], truth: list[tuple[float, float]], radius: float
) -> float:
    """F1 of predicted vs true centres; each true point matches at most one
    prediction within `radius` (closest pairs first)."""
    if not pred and not truth:
        return 1.0
    if not pred or not truth:
        return 0.0
    pairs = sorted(
        (math.hypot(p[0] - t[0], p[1] - t[1]), i, j)
        for i, p in enumerate(pred)
        for j, t in enumerate(truth)
        if math.hypot(p[0] - t[0], p[1] - t[1]) <= radius
    )
    used_p, used_t = set(), set()
    for _, i, j in pairs:
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
    tp = len(used_p)
    if tp == 0:
        return 0.0
    prec, rec = tp / len(pred), tp / len(truth)
    return 2 * prec * rec / (prec + rec)


# --- 方式C: post-processing tuned on the validation split ---------------------


def inside_frame(
    dets: list[Detection],
    frame: tuple[float, float, float, float] | None,
    margin_fraction: float,
) -> list[Detection]:
    """Detections inside the plot frame (x0, y0, x1, y1) widened on every side
    by `margin_fraction` of its width / height (markers straddle the frame
    edge). Tick labels, axis titles and arrows outside it go. No frame: all."""
    if frame is None:
        return list(dets)
    x0, y0, x1, y1 = frame
    mx, my = margin_fraction * (x1 - x0), margin_fraction * (y1 - y0)
    return [d for d in dets if x0 - mx <= d.x <= x1 + mx and y0 - my <= d.y <= y1 + my]


def away_from_frame_edges(
    dets: list[Detection],
    frame: tuple[float, float, float, float] | None,
    band_fraction: float,
) -> list[Detection]:
    """Drop detections within band_fraction x the frame's shorter side of any
    of its four lines -- the inward tick marks of a boxed frame fire weak
    peaks there. No frame: all kept."""
    if frame is None:
        return list(dets)
    x0, y0, x1, y1 = frame
    band = band_fraction * min(x1 - x0, y1 - y0)
    return [
        d for d in dets
        if min(abs(d.x - x0), abs(d.x - x1), abs(d.y - y0), abs(d.y - y1)) > band
    ]


def duplicate_radius(
    image_size: tuple[int, int], marker_px: float | None, frac: float, size_factor: float
) -> float:
    """Duplicate-suppression radius: the old floor (frac of the long side,
    at least 1.5 px), raised to size_factor x the figure's marker size when
    that is known -- a large patterned marker fires several peaks."""
    base = max(1.5, frac * max(image_size))
    if marker_px is None:
        return base
    return max(base, size_factor * marker_px)


def suppress_same_series_duplicates(
    dets: list[Detection], radius: float, embed_threshold: float
) -> list[Detection]:
    """Strongest-first: drop a detection closer than `radius` to a kept one
    whose embedding is within `embed_threshold` (the same series) -- the
    extra peaks of one large or patterned marker. A close detection of
    another series (coincident markers) stays. Input order is kept."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: -d.score):
        if all(
            math.hypot(d.x - k.x, d.y - k.y) >= radius
            or _dist(d.embedding, k.embedding) > embed_threshold
            for k in kept
        ):
            kept.append(d)
    keep = {id(d) for d in kept}
    return [d for d in dets if id(d) in keep]


@dataclass(frozen=True)
class PostConfig:
    """Detector post-processing. With same_series_frac and frame_margin left
    None it is method A's pipeline."""

    threshold: float
    group_threshold: float
    dup_frac: float = 0.004
    same_series_frac: float | None = None
    embed_gate: float = 0.5
    frame_margin: float | None = None
    edge_band: float | None = None
    min_points: int = 2


def postprocess(
    dets: list[Detection],
    image_size: tuple[int, int],
    frame: tuple[float, float, float, float] | None,
    cfg: PostConfig,
) -> list[dict]:
    """Detections -> series of pixel points: peak threshold, frame
    restriction (when a margin is set), frame-edge band (when set), duplicate
    suppression, same-series suppression at a wider radius (when set),
    embedding grouping."""
    kept = [d for d in dets if d.score >= cfg.threshold]
    if cfg.frame_margin is not None:
        kept = inside_frame(kept, frame, cfg.frame_margin)
    if cfg.edge_band is not None:
        kept = away_from_frame_edges(kept, frame, cfg.edge_band)
    kept = suppress_duplicates(kept, duplicate_radius(image_size, None, cfg.dup_frac, 0.0))
    if cfg.same_series_frac is not None:
        kept = suppress_same_series_duplicates(
            kept, cfg.same_series_frac * max(image_size), cfg.embed_gate
        )
    return pixel_answer(group_into_series(kept, cfg.group_threshold), min_points=cfg.min_points)
