"""One point per marker from a colour mask (方式D v3, docs/design/local-model.md
「v3: 検証の誤検知と道具の追加」), for the two failures no v2 tool could fix:

- a marker split into pieces by a white cross or bar (the "four squares"
  marker): pieces a few pixels apart whose union is a round, mostly filled
  box are one marker, at the centre of that box
  (verification.group_pieces);
- markers that overlap or touch, drawn as one blob: the marker size is taken
  from the series itself (the typical isolated blob, ``unit_marker``); a blob
  much larger than that is cut into k markers, k from its area and its
  shape, the centres found by k-means on the blob's core (the pixels at
  least 0.6 of a marker radius inside it), which keeps the overlap from
  pulling the centres together.

Line pieces (verification.is_stroke) are not markers. Pure numpy / scipy.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy import ndimage

from real_chart_bench.domain.verification import blob_shape, group_pieces, is_stroke

SPLIT_AREA = 1.4  # a blob this many unit areas (or ...)
SPLIT_SIDE = 1.35  # ... this many unit diameters long is several markers
CORE = 0.6  # the core: pixels at least this share of a marker radius inside the blob
MERGE_GAP = 0.35  # of the typical piece's side (>= 3 px): pieces of one split marker
MAX_CLUSTER = 4.0  # x the unit diameter: larger blobs are not marker clusters
MAX_K = 12
UNJOIN_MAX_R = 6  # px: the widest joining line removed by opening is about 2r


def unit_marker(sizes: Sequence[tuple[float, float, float]]) -> tuple[float, float]:
    """(diameter, area) of a single marker from the blobs of a series, each
    (width, height, area): among the blobs of at least half the median area
    of the larger half (not fragments of covered markers), the median of those no larger than 1.3x
    their median area and roughly as wide as tall (merged blobs are the
    larger, elongated ones)."""
    if not sizes:
        raise ValueError("no blobs")
    # fragments (markers cut by others on top) are not the unit: only blobs
    # of at least half the large ones count
    areas = [a for _, _, a in sizes]
    big = float(np.median([a for a in areas if a >= np.median(areas)]))
    sizes = [s for s in sizes if s[2] >= 0.5 * big] or list(sizes)
    med = float(np.median([a for _, _, a in sizes]))
    keep = [(w, h, a) for w, h, a in sizes
            if a <= 1.3 * med and max(w, h) <= 1.4 * min(w, h)] or list(sizes)
    return (float(np.median([max(w, h) for w, h, _ in keep])),
            float(np.median([a for _, _, a in keep])))


def _kmeans(pts: np.ndarray, k: int, iters: int = 20) -> np.ndarray:
    """k centres of `pts` (n, 2), started evenly along the main axis."""
    mean = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - mean, full_matrices=False)
    proj = (pts - mean) @ vt[0]
    order = np.argsort(proj)
    cents = np.array([pts[order[int((i + 0.5) * len(pts) / k)]] for i in range(k)], float)
    for _ in range(iters):
        d = ((pts[:, None, :] - cents[None, :, :]) ** 2).sum(-1)
        lab = d.argmin(axis=1)
        new = np.array([pts[lab == j].mean(axis=0) if (lab == j).any() else cents[j]
                        for j in range(k)])
        if np.allclose(new, cents, atol=1e-3):
            break
        cents = new
    return cents


def split_blob(blob: np.ndarray, unit_d: float, unit_area: float) -> list[tuple[float, float]]:
    """Centres (x, y) in the crop's pixels of the markers in one blob."""
    filled = ndimage.binary_fill_holes(np.pad(blob, 1))
    dt = ndimage.distance_transform_edt(filled)[1:-1, 1:-1]
    core = dt >= CORE * unit_d / 2
    if not core.any():
        core = dt >= 0.5 * dt.max()
    ys, xs = np.nonzero(core)
    pts = np.column_stack([xs, ys]).astype(float)
    area = float(filled.sum())
    k = max(1, round(area / unit_area))
    if len(pts) > 1:
        c = pts - pts.mean(axis=0)
        _, sv, vt = np.linalg.svd(c, full_matrices=False)
        along = c @ vt[0]
        across = c @ vt[1] if len(vt) > 1 else np.zeros(len(c))
        if (np.ptp(along) - np.ptp(across)) >= 0.25 * unit_d:
            k = max(k, 2)
    k = min(k, MAX_K, len(pts))
    if k <= 1:
        ys, xs = np.nonzero(blob)
        return [(float(xs.mean()), float(ys.mean()))]
    return [(float(x), float(y)) for x, y in _kmeans(pts, k)]


def _disc(r: int) -> np.ndarray:
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return xx * xx + yy * yy <= r * r


def unjoin_lines(mask: np.ndarray, min_diameter: float = 3,
                 max_radius: int = UNJOIN_MAX_R) -> tuple[np.ndarray, int]:
    """Markers joined by a line of their own colour form one long piece. Each
    such piece (a line piece by its shape, verification.is_stroke) is opened
    with the smallest disc that leaves only marker-like pieces: the line goes,
    filled markers stay. Pieces where no disc does that are kept as they
    are. Returns the new mask and how many pieces were opened."""
    lab, _ = ndimage.label(mask, structure=np.ones((3, 3), bool))
    out = mask.copy()
    done = 0
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        comp = lab[sl] == k
        shp = blob_shape(comp)
        if not is_stroke(shp) or shp["thickness"] < 2 * min_diameter / 2 + 2:
            continue  # a plain line (no marker in it) or a small piece
        for r in range(1, max_radius + 1):
            opened = ndimage.binary_opening(np.pad(comp, r + 1), structure=_disc(r))
            opened = opened[r + 1:-(r + 1), r + 1:-(r + 1)]
            sub, n = ndimage.label(opened, structure=np.ones((3, 3), bool))
            if n == 0:
                break
            shapes = [blob_shape(sub[s2] == j)
                      for j, s2 in enumerate(ndimage.find_objects(sub), start=1)]
            if all(not is_stroke(s) for s in shapes):
                region = out[sl]
                region[comp] = False
                region[opened] = True
                done += 1
                break
    return out, done


def marker_centres(
    mask: np.ndarray,
    *,
    min_diameter: float = 3,
    max_diameter: float | None = None,
    diameter: float | None = None,
    merge: bool = True,
    split: bool = True,
    merge_gap: float | None = None,
) -> tuple[list[tuple[float, float]], dict]:
    """Marker centres (x, y) in the mask's pixels, and what was done:
    {"diameter_px", "n_merged", "n_split", "n_markers"}. `diameter` fixes
    the marker size (else taken from the series); `min_diameter` /
    `max_diameter` bound a single marker's side."""
    mask, unjoined = unjoin_lines(np.asarray(mask, bool), min_diameter)
    lab, n = ndimage.label(mask, structure=np.ones((3, 3), bool))
    slices = ndimage.find_objects(lab)
    pieces = []
    for k, sl in enumerate(slices, start=1):
        comp = lab[sl] == k
        shp = blob_shape(comp)
        if shp["side"] < min_diameter / 2 or is_stroke(shp):
            continue
        box = (sl[1].start, sl[0].start, sl[1].stop - 1, sl[0].stop - 1)
        pieces.append((k, box, shp["area"], shp["side"]))
    info = {"diameter_px": None, "n_merged": 0, "n_split": 0, "n_markers": 0,
            "n_unjoined": unjoined}
    if not pieces:
        return [], info
    if merge:
        typical = float(np.median([p[3] for p in pieces]))
        gap = merge_gap if merge_gap is not None else max(3.0, MERGE_GAP * typical)
        groups = group_pieces([(p[1], p[2]) for p in pieces], gap)
    else:
        groups = [[i] for i in range(len(pieces))]
    blobs = []
    for g in groups:
        x0 = min(pieces[i][1][0] for i in g)
        y0 = min(pieces[i][1][1] for i in g)
        x1 = max(pieces[i][1][2] for i in g)
        y1 = max(pieces[i][1][3] for i in g)
        blobs.append({"labels": [pieces[i][0] for i in g], "box": (x0, y0, x1, y1),
                      "w": x1 - x0 + 1, "h": y1 - y0 + 1,
                      "area": sum(pieces[i][2] for i in g)})
    hi = max_diameter if max_diameter is not None else math.inf
    single = [b for b in blobs if min_diameter <= max(b["w"], b["h"]) <= hi]
    if diameter is not None:
        unit_d = float(diameter)
        unit_a = math.pi * unit_d * unit_d / 4
        if single:  # the series' own area for that size, when it has one
            near = [b["area"] for b in single
                    if abs(max(b["w"], b["h"]) - unit_d) <= 0.15 * unit_d]
            if near:
                unit_a = float(np.median(near))
    elif single:
        unit_d, unit_a = unit_marker([(b["w"], b["h"], b["area"]) for b in single])
    else:
        return [], info
    info["diameter_px"] = round(unit_d, 1)
    out: list[tuple[float, float]] = []
    for b in blobs:
        side = max(b["w"], b["h"])
        x0, y0, x1, y1 = b["box"]
        merged = len(b["labels"]) > 1
        big = b["area"] >= SPLIT_AREA * unit_a or side >= SPLIT_SIDE * unit_d
        if split and big and not merged and side <= MAX_CLUSTER * unit_d:
            crop = np.isin(lab[y0:y1 + 1, x0:x1 + 1], b["labels"])
            cs = split_blob(crop, unit_d, unit_a)
            if len(cs) > 1:
                info["n_split"] += 1
            out += [(x0 + cx, y0 + cy) for cx, cy in cs]
            continue
        if not min_diameter <= side <= hi:
            continue
        if merged:
            info["n_merged"] += 1
            out.append(((x0 + x1) / 2, (y0 + y1) / 2))
        else:
            crop = lab[y0:y1 + 1, x0:x1 + 1] == b["labels"][0]
            ys, xs = np.nonzero(crop)
            out.append((float(x0 + xs.mean()), float(y0 + ys.mean())))
    out = sorted((round(x, 2), round(y, 2)) for x, y in out)
    info["n_markers"] = len(out)
    return out, info
