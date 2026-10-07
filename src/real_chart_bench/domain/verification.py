"""verify: objective checks of an extraction on the image (方式D「検証と
やり直し」, docs/design/local-model.md).

The orchestrator calls verify after each extraction. It looks only at the
image and at the answer's points -- never at a ground truth -- and reports
signals a person would check on an overlay:

- per series: do the points sit on ink? ``hit`` is the share of points with
  ink within a small radius, ``null`` the same share with the points shifted
  by a few percent of the frame (the Starrydata gate's test, see
  tick_calibration.ink_contrast); ``contrast = hit - null``. Points on the
  frame lines cannot testify and are left out of both. ``color_consistency``
  is the share of the points whose ink colour agrees with the series'
  main colour. ``duplicates`` (points within half a marker of an earlier
  point of the series) and ``outside`` (points outside the frame);
- ``cross_duplicates``: points sitting on a point of another series;
- ``unexplained``: marker-sized blobs of ink inside the frame that no point
  covers (missed markers, or a missed series). Masked-out regions, OCR'd
  text and legend symbols (a blob just left of a text box on its row) do not
  count;
- for condition 1, ``calibration_signals``: do the tick labels fit the
  chosen scale (residual, the label grid, label pixels on tick marks,
  detected tick marks on the label grid, axis direction).

``verdict`` turns the signals into accept / redo with reasons and a scalar
``score`` (an estimate of point F1 from the signals: a precision proxy from
the hit contrast, a recall proxy from the unexplained blobs). Its thresholds
(``Thresholds``) are measured on development data that is not the benchmark
(scripts/eval/orchestrator/verify_dev.py); the redo decision is that
measurement, not prompt wording.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from real_chart_bench.domain.digitizer_tools import _line, build_mask, to_hex
from real_chart_bench.domain.tick_calibration import (
    away_from_frame_lines,
    frame_band,
    ink_contrast,
    ink_radius,
    null_shift,
    without_frame_lines,
)

INK_LEVEL = 48  # a channel this far from the background is ink (grid greys are not)
COLOR_MATCH = 60.0  # RGB distance within which two ink colours are the same
OUTSIDE_MARGIN = 0.02  # of the frame's size: markers straddle the frame
MARKER_MAX_FRACTION = 0.1  # a component larger than this share of the frame is not a marker
DEFAULT_MARKER_FRACTION = 0.02  # marker diameter when no point sits on a marker-sized blob
BLOB_MIN, BLOB_MAX, BLOB_THIN = 0.5, 2.5, 0.3  # marker-like: x the marker diameter
LEGEND_GAP = 4.0  # a blob this many diameters left of a text box is its legend symbol
LOOKALIKE_NCC = 0.7  # a place this similar to a series' median patch looks like its points
LOOKALIKE_HALF = 0.6  # template half-size, x the marker diameter
DET_SUPPORT, DET_CONFIDENT = 0.2, 0.5  # detector peak scores: supports a point / is a marker
DET_RADIUS = 0.015  # of the frame's long side: a peak this close is the same marker
# v3 (docs/design/local-model.md「v3: 検証の誤検知と道具の追加」): what is not a marker
ELONG_STROKE = 2.5  # a piece this elongated (sqrt of its moments' ratio) is a line piece
ELONG_THIN = 1.8  # ... or this elongated and thin
THIN_STROKE = 0.35  # thin: stroke width (after filling holes) <= this x its side
GROUP_GAP = 0.008  # of the frame's short side (>= 3 px): pieces of one split marker
GROUP_MAX_ASPECT, GROUP_MIN_FILL = 1.5, 0.6  # a split marker's pieces fill a round box
GROUP_MIN_FILL_MANY = 0.4  # ... three or more pieces (quarters)
CAP_ALIGN = 4.0  # a thin piece within this many diameters of a point, on its row or column
LOOKALIKE_DENSITY = 0.6  # a look-alike carries at least this share of the points' ink
HOLLOW_REACH = 0.03  # of the frame's short side: the marker under a point, for its size
ASYMMETRY_STROKE = 0.15  # a thin piece whose ink centre is off its box centre (T, L)


# ------------------------------------------------------------------ ink


def estimate_background(rgb: np.ndarray, step: int = 16) -> tuple[int, int, int]:
    """The most common colour (binned by `step`), as the mean of its bin."""
    px = rgb[..., :3].reshape(-1, 3).astype(np.int64)
    q = px // step
    key = q[:, 0] * 4096 + q[:, 1] * 64 + q[:, 2]
    vals, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    top = cnt.argmax()
    mean = px[inv.reshape(-1) == top].mean(axis=0)
    return tuple(int(round(v)) for v in mean)


def ink_mask(rgb: np.ndarray, background: Sequence[int] | None = None,
             level: int = INK_LEVEL) -> np.ndarray:
    """True where some channel differs from the background by more than
    `level` -- markers, lines, text; not paper, JPEG noise or light grids."""
    bg = np.array(background if background is not None else estimate_background(rgb))
    diff = np.abs(rgb[..., :3].astype(np.int16) - bg.astype(np.int16))
    return (diff > level).any(axis=-1)


# ------------------------------------------------------------------ helpers


def _points(s: dict) -> list[tuple[float, float]]:
    return [(float(x), float(y)) for x, y in zip(s.get("x", []), s.get("y", []), strict=True)]


def _ink_colour_at(rgb, ink, bg, x, y, r) -> np.ndarray | None:
    """The ink pixel near (x, y) farthest from the background colour."""
    h, w = ink.shape
    cx, cy = int(round(x)), int(round(y))
    ya, yb, xa, xb = max(0, cy - r), min(h, cy + r + 1), max(0, cx - r), min(w, cx + r + 1)
    if ya >= yb or xa >= xb:
        return None
    win = ink[ya:yb, xa:xb]
    if not win.any():
        return None
    px = rgb[ya:yb, xa:xb, :3][win].astype(float)
    d = ((px - bg) ** 2).sum(axis=1)
    return px[d.argmax()]


def _colour_cluster(colours: list[np.ndarray]) -> tuple[float, np.ndarray | None]:
    """(share of the largest group of mutually close colours, its median)."""
    if not colours:
        return 0.0, None
    c = np.array(colours)
    d = np.sqrt(((c[:, None, :] - c[None, :, :]) ** 2).sum(-1))
    near = d <= COLOR_MATCH
    best = near.sum(axis=1).argmax()
    members = c[near[best]]
    return float(near[best].sum() / len(c)), np.median(members, axis=0)


def _components(ink: np.ndarray):
    lab, n = ndimage.label(ink, structure=np.ones((3, 3), bool))
    return lab, n, ndimage.find_objects(lab)


def _label_near(lab: np.ndarray, x: float, y: float, r: int) -> int:
    h, w = lab.shape
    cx, cy = int(round(x)), int(round(y))
    ya, yb, xa, xb = max(0, cy - r), min(h, cy + r + 1), max(0, cx - r), min(w, cx + r + 1)
    if ya >= yb or xa >= xb:
        return 0
    win = lab[ya:yb, xa:xb]
    ys, xs = np.nonzero(win)
    if len(ys) == 0:
        return 0
    k = ((ys + ya - cy) ** 2 + (xs + xa - cx) ** 2).argmin()
    return int(win[ys[k], xs[k]])


def _cross_run(lab: np.ndarray, k: int, x: float, y: float, limit: int) -> int | None:
    """min(horizontal, vertical) run of component k through (x, y); None
    when (x, y) is not on it."""
    h, w = lab.shape
    cx, cy = int(round(x)), int(round(y))
    if not (0 <= cx < w and 0 <= cy < h) or lab[cy, cx] != k:
        return None
    runs = []
    for dx, dy in ((1, 0), (0, 1)):
        n = 1
        for sgn in (1, -1):
            step = 1
            while step <= limit:
                xx, yy = cx + sgn * dx * step, cy + sgn * dy * step
                if not (0 <= xx < w and 0 <= yy < h) or lab[yy, xx] != k:
                    break
                n += 1
                step += 1
        runs.append(n)
    return min(runs)


def _marker_diameter(lab, slices, points, r: int, frame, pieces=None, groups=None,
                     owner=None) -> float:
    fx0, fy0, fx1, fy1 = frame
    short = min(fx1 - fx0, fy1 - fy0)
    sizes = []
    limit = MARKER_MAX_FRACTION * short
    for x, y in points:
        # v3: a hollow marker's ring may lie farther than the ink radius
        k = _label_near(lab, x, y, max(r, int(round(HOLLOW_REACH * short))))
        if not k:
            continue
        if owner is not None and k in owner and len(groups[owner[k]]) > 1:
            gx0, gy0, gx1, gy1 = _group_box(pieces, groups[owner[k]])
            sizes.append(max(gx1 - gx0 + 1, gy1 - gy0 + 1))  # a split marker: all of it
            continue
        sl = slices[k - 1]
        side = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start)
        if side <= limit:
            sizes.append(side)
            continue
        # a marker joined to a line or another marker: the shorter of the
        # horizontal and vertical ink runs through its centre (filled markers)
        run = _cross_run(lab, k, x, y, int(limit))
        if run and run <= limit:
            sizes.append(run)
    if sizes:
        return float(np.median(sizes))
    return max(4.0, DEFAULT_MARKER_FRACTION * short)


# ------------------------------------------------------------------ shapes (v3)


def blob_shape(comp: np.ndarray) -> dict:
    """Shape of one connected piece of ink (a boolean crop): its side (longer
    bbox side), area, elongation (sqrt of the ratio of its second moments; 1
    for a disc, a square or a "+", large for a line piece) and thickness
    (twice the largest inscribed radius after filling holes: about the side
    for a filled or hollow marker, the stroke width for lines, "+" and "x")."""
    comp = np.asarray(comp, bool)
    ys, xs = np.nonzero(comp)
    if len(ys) == 0:
        return {"side": 0, "area": 0, "elongation": 1.0, "thickness": 0.0}
    h, w = int(ys.max() - ys.min() + 1), int(xs.max() - xs.min() + 1)
    cov = np.cov(np.vstack([xs, ys]).astype(float)) if len(ys) > 1 else np.zeros((2, 2))
    cov = cov + np.eye(2) / 12.0  # a pixel's own extent
    ev = np.linalg.eigvalsh(cov)
    filled = ndimage.binary_fill_holes(np.pad(comp, 1))
    thick = 2.0 * float(ndimage.distance_transform_edt(filled).max())
    side = int(max(h, w))
    off = math.hypot(xs.mean() - (xs.min() + xs.max()) / 2, ys.mean() - (ys.min() + ys.max()) / 2)
    return {"side": side, "area": int(len(ys)),
            "elongation": float(math.sqrt(ev[1] / max(ev[0], 1e-9))), "thickness": thick,
            "asymmetry": float(off / side)}


def is_stroke(shape: dict) -> bool:
    """A piece of a line (a connecting segment, an error bar or its cap, a
    fit-curve fragment), not a marker: clearly elongated, or thin and
    somewhat elongated, or thin and lopsided (a bar with its cap: a T or an
    L). "+" / "x" markers are thin but neither elongated nor lopsided."""
    e, side = shape["elongation"], max(1, shape["side"])
    thin = shape["thickness"] <= THIN_STROKE * side
    # a row of touching markers is elongated too, but as thick as a marker
    return ((e >= ELONG_STROKE and shape["thickness"] <= 0.25 * side)
            or (thin and e >= ELONG_THIN)
            or (thin and shape.get("asymmetry", 0.0) >= ASYMMETRY_STROKE))


def _box_gap(a, b) -> float:
    dx = max(0.0, b[0] - a[2] - 1, a[0] - b[2] - 1)
    dy = max(0.0, b[1] - a[3] - 1, a[1] - b[3] - 1)
    return max(dx, dy)


def group_pieces(pieces: Sequence[tuple[Sequence[float], int]], gap: float) -> list[list[int]]:
    """Pieces ((x0, y0, x1, y1) inclusive box, ink area) that are one marker
    split by a white cross or bar: pieces at most `gap` px apart whose union
    is a round box (aspect <= GROUP_MAX_ASPECT) mostly filled with their ink
    (>= GROUP_MIN_FILL). Other pieces stay alone. Groups of indices, in the
    order of their first piece."""
    n = len(pieces)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _box_gap(pieces[i][0], pieces[j][0]) <= gap:
                parent[find(i)] = find(j)
    comps: dict[int, list[int]] = {}
    for i in range(n):
        comps.setdefault(find(i), []).append(i)
    out: list[list[int]] = []
    for members in comps.values():
        if len(members) == 1:
            out.append(members)
            continue
        x0 = min(pieces[k][0][0] for k in members)
        y0 = min(pieces[k][0][1] for k in members)
        x1 = max(pieces[k][0][2] for k in members)
        y1 = max(pieces[k][0][3] for k in members)
        bw, bh = x1 - x0 + 1, y1 - y0 + 1
        fill = sum(pieces[k][1] for k in members) / (bw * bh)
        # two pieces: halves (fill ~0.75), not two markers touching corner to
        # corner (<= 0.5); three or four: quarters of a disc fill about half
        need = GROUP_MIN_FILL if len(members) == 2 else GROUP_MIN_FILL_MANY
        if max(bw, bh) / min(bw, bh) <= GROUP_MAX_ASPECT and fill >= need:
            out.append(sorted(members))
        else:
            out += [[k] for k in members]
    return sorted(out, key=lambda g: g[0])


def _pieces(lab, slices, short: float):
    """Marker-candidate pieces (components no larger than a quarter of the
    frame's short side) with their shape, and the groups of split markers
    among the pieces that are not line pieces. Returns (pieces, groups,
    owner): pieces[k-1] for label k (None when too large), groups as lists
    of labels, owner[label] = group index."""
    limit = 0.25 * short
    pieces: list[dict | None] = []
    for k, sl in enumerate(slices, start=1):
        if sl is None:
            pieces.append(None)
            continue
        h, w = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        if max(h, w) > limit:
            pieces.append(None)
            continue
        shp = blob_shape(lab[sl] == k)
        shp["box"] = (sl[1].start, sl[0].start, sl[1].stop - 1, sl[0].stop - 1)
        shp["stroke"] = is_stroke(shp)
        pieces.append(shp)
    cand = [k for k, p in enumerate(pieces, start=1) if p is not None and not p["stroke"]]
    gap = max(3.0, GROUP_GAP * short)
    grouped = group_pieces([(pieces[k - 1]["box"], pieces[k - 1]["area"]) for k in cand], gap)
    groups = [[cand[i] for i in g] for g in grouped]
    owner = {k: gi for gi, g in enumerate(groups) for k in g}
    return pieces, groups, owner


def _group_box(pieces, g):
    boxes = [pieces[k - 1]["box"] for k in g]
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _is_cap(shape: dict, cx: float, cy: float, pts, d: float, r: int) -> bool:
    """A thin piece on a point's column or row, a few diameters away: an
    error bar's cap (or a detached piece of the bar)."""
    if shape["thickness"] > THIN_STROKE * max(1, shape["side"]) or shape["elongation"] < 1.4:
        return False  # not thin, or as round as a "+" / "x" marker
    tol = max(r, 0.25 * d)
    reach = CAP_ALIGN * d
    return any((abs(cx - x) <= tol and abs(cy - y) <= reach)
               or (abs(cy - y) <= tol and abs(cx - x) <= reach) for x, y in pts)


def _near_text(bx0, by0, bx1, by1, boxes, d) -> bool:
    for tx0, ty0, tx1, ty1 in boxes:
        if bx1 >= tx0 - 2 and bx0 <= tx1 + 2 and by1 >= ty0 - 2 and by0 <= ty1 + 2:
            return True  # inside / touching the text
        cy = (by0 + by1) / 2
        if ty0 - 0.25 * d <= cy <= ty1 + 0.25 * d and 0 <= tx0 - bx1 <= LEGEND_GAP * d:
            return True  # a legend symbol: the text is just right of it
    return False


# ------------------------------------------------------------------ look-alikes


def _ncc_map(img: np.ndarray, tmpl: np.ndarray) -> np.ndarray:
    """Normalised cross-correlation of a (h, w, c) image with a (k, k, c)
    template, at every template-centre position (same size as img); 0 where
    the image window is flat."""
    from scipy.signal import fftconvolve

    k = tmpl.shape[0]
    n = k * k * tmpl.shape[2]
    t0 = tmpl - tmpl.mean(axis=(0, 1), keepdims=True)
    tnorm = float(np.sqrt((t0**2).sum()))
    if tnorm < 1e-6:
        return np.zeros(img.shape[:2], np.float32)
    num = np.zeros(img.shape[:2], np.float64)
    var = np.zeros(img.shape[:2], np.float64)
    for c in range(img.shape[2]):
        ch = img[..., c].astype(np.float64)
        num += fftconvolve(ch, t0[::-1, ::-1, c], mode="same")
        s1 = ndimage.uniform_filter(ch, size=k, mode="constant") * k * k
        s2 = ndimage.uniform_filter(ch * ch, size=k, mode="constant") * k * k
        var += np.maximum(0.0, s2 - s1 * s1 / (k * k))
    den = np.sqrt(var) * tnorm
    out = np.where(den > 1e-6 * n, num / np.maximum(den, 1e-12), 0.0)
    return out.astype(np.float32)


def _templates(diff: np.ndarray, pts, half: int) -> np.ndarray | None:
    h, w = diff.shape[:2]
    patches = []
    for x, y in pts[:60]:
        cx, cy = int(round(x)), int(round(y))
        if half <= cx < w - half and half <= cy < h - half:
            patches.append(diff[cy - half:cy + half + 1, cx - half:cx + half + 1])
    if len(patches) < 3:
        return None
    return np.median(np.stack(patches), axis=0)


def _ink_share(ink: np.ndarray, x: float, y: float, half: int) -> float:
    h, w = ink.shape
    cx, cy = int(round(x)), int(round(y))
    win = ink[max(0, cy - half):min(h, cy + half + 1), max(0, cx - half):min(w, cx + half + 1)]
    return float(win.mean()) if win.size else 0.0


def lookalikes(diff: np.ndarray, clean_ink: np.ndarray, series_pts: list, all_pts: list,
               frame, band: int, d: float, r: int, region: np.ndarray | None,
               text_boxes) -> tuple[list[float | None], list[dict]]:
    """Per series: the share of its points that look like the series' typical
    point (NCC >= LOOKALIKE_NCC with the median patch); and the places in the
    frame that look like a series' points but carry no point of the answer."""
    h, w = diff.shape[:2]
    half = max(3, int(round(LOOKALIKE_HALF * d)))
    fx0, fy0, fx1, fy1 = frame
    xa, xb = max(0, int(fx0) - half), min(w, int(fx1) + half + 1)
    ya, yb = max(0, int(fy0) - half), min(h, int(fy1) + half + 1)
    sub = diff[ya:yb, xa:xb]
    nms = max(3, int(round(d)))
    hd = max(2, int(round(0.5 * d)))
    selfs: list[float | None] = []
    found: list[dict] = []
    for si, pts in enumerate(series_pts[:12]):
        t = _templates(diff, pts, half)
        if t is None:
            selfs.append(None)
            continue
        ncc = _ncc_map(sub, t)
        own = 0
        for x, y in pts:
            cx, cy = int(round(x)) - xa, int(round(y)) - ya
            win = ncc[max(0, cy - r):cy + r + 1, max(0, cx - r):cx + r + 1]
            own += bool(win.size) and float(win.max()) >= LOOKALIKE_NCC
        selfs.append(own / len(pts))
        # v3: a look-alike carries about as much ink as the series' points (a
        # line or a fit curve through the template window carries less)
        own_ink = float(np.median([_ink_share(clean_ink, x, y, hd) for x, y in pts]))
        peaks = (ncc >= LOOKALIKE_NCC) & (ncc == ndimage.maximum_filter(ncc, size=nms))
        for py, px in zip(*np.nonzero(peaks), strict=True):
            x, y = float(px + xa), float(py + ya)
            if not (fx0 + band < x < fx1 - band and fy0 + band < y < fy1 - band):
                continue
            iy, ix = int(y), int(x)
            if region is not None and not region[iy, ix]:
                continue
            if not clean_ink[max(0, iy - r):iy + r + 1, max(0, ix - r):ix + r + 1].any():
                continue
            if any(math.hypot(x - a, y - b) < d for a, b in all_pts):
                continue
            if any(math.hypot(x - f["x"], y - f["y"]) < d for f in found):
                continue
            if _near_text(x - half, y - half, x + half, y + half, text_boxes, d):
                continue
            if _ink_share(clean_ink, x, y, hd) < LOOKALIKE_DENSITY * own_ink:
                continue
            found.append({"x": round(x, 1), "y": round(y, 1), "series": si,
                          "ncc": round(float(ncc[py, px]), 3)})
    return selfs, found


# ------------------------------------------------------------------ verify


def verify(
    rgb: np.ndarray,
    series: Sequence[dict],
    frame: Sequence[float] | None,
    *,
    mask: dict | None = None,
    text_boxes: Sequence[Sequence[float]] = (),
    calibration: dict | None = None,
    tick_marks: dict | None = None,
    detections: Sequence[tuple[float, float, float]] | None = None,
) -> dict:
    """Signals about pixel `series` ({"label", "x", "y"} in image pixels) on
    `rgb`. `frame` [x0, y0, x1, y1] is the plot frame (None: the whole
    image); `mask` a mask spec (digitizer_tools.build_mask) of the region
    examined; `text_boxes` OCR'd words [x0, y0, x1, y1]; `calibration` and
    `tick_marks` ({"x": [px...], "y": [px...]}) add calibration_signals;
    `detections` (x, y, score) -- the marker detector's raw peaks -- add
    detector_agreement."""
    h, w = rgb.shape[:2]
    frame = [float(v) for v in frame] if frame else [0.0, 0.0, w - 1.0, h - 1.0]
    fx0, fy0, fx1, fy1 = frame
    bg = estimate_background(rgb)
    bg_arr = np.array(bg, float)
    ink = ink_mask(rgb, bg)
    r = ink_radius(frame)
    band = frame_band(frame, r)
    clean = without_frame_lines(ink, frame, band)
    lab, _, slices = _components(clean)
    all_pts = [p for s in series for p in _points(s)]
    pieces, groups, gowner = _pieces(lab, slices, min(fx1 - fx0, fy1 - fy0))
    d = _marker_diameter(lab, slices, all_pts, r, frame, pieces, groups, gowner)
    dup_r = max(2.0, 0.5 * d)
    mx, my = OUTSIDE_MARGIN * (fx1 - fx0), OUTSIDE_MARGIN * (fy1 - fy0)
    shift = null_shift(frame, r)

    out_series = []
    owner: list[tuple[float, float, int]] = []
    cross = 0
    for i, s in enumerate(series):
        pts = _points(s)
        testify = away_from_frame_lines(pts, frame, band)
        if testify:
            hit, null = ink_contrast(testify, clean, radius=r, shift_px=shift)
        else:
            hit = null = None
        colours = [c for c in (_ink_colour_at(rgb, clean, bg_arr, x, y, r) for x, y in pts)
                   if c is not None]
        share, colour = _colour_cluster(colours)
        dups = 0
        for k, (x, y) in enumerate(pts):
            if any(math.hypot(x - a, y - b) < dup_r for a, b in pts[:k]):
                dups += 1
            if any(j != i and math.hypot(x - a, y - b) < dup_r for a, b, j in owner):
                cross += 1
        owner += [(x, y, i) for x, y in pts]
        outside = sum(1 for x, y in pts
                      if not (fx0 - mx <= x <= fx1 + mx and fy0 - my <= y <= fy1 + my))
        out_series.append({
            "index": i, "label": s.get("label"), "n": len(pts), "testifying": len(testify),
            "hit": None if hit is None else round(hit, 4),
            "null": None if null is None else round(null, 4),
            "contrast": None if hit is None else round(hit - null, 4),
            "color": None if colour is None else to_hex([int(round(v)) for v in colour]),
            "color_consistency": round(share, 4) if colours else None,
            "duplicates": dups, "outside": outside,
        })

    # unexplained marker-like blobs inside the frame
    region = build_mask(h, w, mask) if mask else None
    series_cols = [np.array([int(c["color"][k:k + 2], 16) for k in (1, 3, 5)], float)
                   for c in out_series if c["color"]]
    blobs = []
    # v3: line pieces (segments, error bars and caps, fit-curve fragments)
    # are not markers; the pieces of a marker split by a white cross are one
    for g in groups:
        bx0, by0, bx1, by1 = _group_box(pieces, g)
        bw, bh = bx1 - bx0 + 1, by1 - by0 + 1
        if not (BLOB_MIN * d <= max(bw, bh) <= BLOB_MAX * d and min(bw, bh) >= BLOB_THIN * d):
            continue
        sl = (slice(by0, by1 + 1), slice(bx0, bx1 + 1))
        comp = np.isin(lab[sl], g)
        if len(g) > 1:
            cx, cy = (bx0 + bx1) / 2, (by0 + by1) / 2
        else:
            ys, xs = np.nonzero(comp)
            cx, cy = float(xs.mean() + bx0), float(ys.mean() + by0)
        if not (fx0 + band < cx < fx1 - band and fy0 + band < cy < fy1 - band):
            continue
        if region is not None and not region[int(round(cy)), int(round(cx))]:
            continue
        if any(bx0 - r <= x <= bx1 + r and by0 - r <= y <= by1 + r for x, y in all_pts):
            continue
        if len(g) == 1 and _is_cap(pieces[g[0] - 1], cx, cy, all_pts, d, r):
            continue
        if _near_text(bx0, by0, bx1, by1, text_boxes, d):
            continue
        col = np.median(rgb[sl][comp][:, :3].astype(float), axis=0)
        same = any(np.sqrt(((col - c) ** 2).sum()) <= COLOR_MATCH for c in series_cols)
        blobs.append({"x": round(cx, 1), "y": round(cy, 1), "size": int(max(bw, bh)),
                      "color": to_hex([int(round(v)) for v in col]), "same_color": same})
    n_pts = len(all_pts)
    diff = np.abs(rgb[..., :3].astype(np.float32) - bg_arr.astype(np.float32)) / 255.0
    selfs, alikes = lookalikes(diff, clean, [_points(s) for s in series], all_pts, frame,
                               band, d, r, region, text_boxes)
    for s, v in zip(out_series, selfs + [None] * len(out_series), strict=False):
        s["self_match"] = None if v is None else round(v, 4)
    missed = [(b["x"], b["y"]) for b in blobs]
    for a in alikes:
        if all(math.hypot(a["x"] - x, a["y"] - y) >= d for x, y in missed):
            missed.append((a["x"], a["y"]))
    out = {
        "kind": "verify",
        "frame": [round(v, 1) for v in frame],
        "marker_diameter_px": round(d, 1),
        "ink_radius_px": r,
        "n_series": len(series),
        "n_points": n_pts,
        "series": out_series,
        "cross_duplicates": cross,
        "unexplained": {
            "n": len(blobs),
            "same_color": sum(b["same_color"] for b in blobs),
            "share": round(len(blobs) / (len(blobs) + n_pts), 4) if blobs or n_pts else 0.0,
            "blobs": blobs[:30],
        },
        "lookalikes": {"n": len(alikes), "points": alikes[:30]},
        "missed": {"n": len(missed),
                   "share": round(len(missed) / (len(missed) + n_pts), 4)
                   if missed or n_pts else 0.0},
    }
    if detections is not None:
        out["detector"] = detector_agreement(all_pts, detections, frame, band, region,
                                             text_boxes, d)
    if calibration is not None:
        out["calibration"] = calibration_signals(calibration, tick_marks)
    return out


def detector_agreement(all_pts, detections, frame, band: int, region, text_boxes,
                       d: float) -> dict:
    """The answer against the marker detector's raw peaks (an independent
    marker map): ``support`` = share of the answer's points with a peak of
    score >= DET_SUPPORT nearby; ``coverage`` = share of the confident peaks
    (score >= DET_CONFIDENT, inside the frame, not masked, not text) with an
    answer point nearby; ``f1`` their harmonic mean. Near = 1.5% of the
    frame's long side."""
    fx0, fy0, fx1, fy1 = frame
    rad = max(3.0, DET_RADIUS * max(fx1 - fx0, fy1 - fy0))
    sup = [(x, y) for x, y, s in detections if s >= DET_SUPPORT]
    conf = []
    for x, y, s in detections:
        if s < DET_CONFIDENT or not (fx0 + band < x < fx1 - band and fy0 + band < y < fy1 - band):
            continue
        if region is not None and not region[int(y), int(x)]:
            continue
        if _near_text(x - d / 2, y - d / 2, x + d / 2, y + d / 2, text_boxes, d):
            continue
        if any(math.hypot(x - a, y - b) < rad for a, b in conf):
            continue  # one marker, two peaks
        conf.append((x, y))

    def near(p, pool):
        return any(math.hypot(p[0] - a, p[1] - b) <= rad for a, b in pool)

    support = sum(near(p, sup) for p in all_pts) / len(all_pts) if all_pts else 0.0
    uncovered = [c for c in conf if not near(c, all_pts)]
    coverage = 1 - len(uncovered) / len(conf) if conf else 1.0
    f1 = 0.0 if support + coverage == 0 else 2 * support * coverage / (support + coverage)
    return {"support": round(support, 4), "coverage": round(coverage, 4), "f1": round(f1, 4),
            "n_confident": len(conf),
            "uncovered": [{"x": round(x, 1), "y": round(y, 1)} for x, y in uncovered[:30]]}


# ------------------------------------------------------------------ calibration


def _t(v: float, scale: str) -> float | None:
    if scale == "log":
        return math.log10(v) if v > 0 else None
    return float(v)


def _one_digit(v: float) -> bool:
    if v <= 0:
        return False
    m = v / 10 ** math.floor(math.log10(v) + 1e-9)
    return abs(m - round(m)) <= 1e-6 * max(1.0, m)


MARKS_AT_LABELS = 0.5  # share of the labels on detected marks for the marks to count
LATTICE_TOL = 0.2  # a mark spacing within this share of a whole number of minor steps
LOG_TOL, LOG_SHIFT = 0.02, 0.03  # decades: a mark on d x 10^k; the labels' offset searched


def _lattice_share(marks: Sequence[float], label_step_px: float) -> float:
    """Linear axis: the share of consecutive mark spacings that are a whole
    number of minor steps, the minor step being the label step divided by
    any number of subdivisions (Origin's "10 minor ticks" makes 11). The
    subdivision is read from the marks' own median spacing (and +-1); the
    test uses spacings only, so a slightly wrong slope does not add up."""
    gaps = [b - a for a, b in zip(marks, marks[1:], strict=False) if b - a > 0.5]
    if not gaps or label_step_px <= 0:
        return 0.0
    n0 = max(1, round(label_step_px / float(np.median(gaps))))
    best = 0.0
    for n in {max(1, n0 - 1), n0, n0 + 1}:
        step = label_step_px / n
        ok = sum(1 for g in gaps
                 if round(g / step) >= 1 and abs(g / step - round(g / step)) <= LATTICE_TOL)
        best = max(best, ok / len(gaps))
    return best


def _log_marks_on_grid(marks: Sequence[float], slope: float, intercept: float) -> float:
    """Log axis: the share of marks on d x 10^k (d = 1..9), allowing the
    labels' line to be off by up to LOG_SHIFT decades (label centres a pixel
    or two off their ticks)."""
    ts = [(m - intercept) / slope for m in marks]
    logs = [math.log10(k) for k in range(1, 11)]
    best = 0.0
    for delta in np.linspace(-LOG_SHIFT, LOG_SHIFT, 13):
        ok = 0
        for t in ts:
            f = (t + delta) - math.floor(t + delta)
            ok += min(abs(f - lk) for lk in logs) <= LOG_TOL
        best = max(best, ok / len(ts))
    return best


def _axis_signals(axis: dict, which: str, marks: Sequence[float] | None) -> dict:
    scale = axis["scale"]
    try:
        slope, intercept = _line(axis)
    except (ValueError, ZeroDivisionError, KeyError, TypeError):
        return {"ok": False, "n_ticks": len(axis.get("ticks") or [])}
    ticks = [(float(p), _t(float(v), scale)) for p, v in axis["ticks"]]
    ticks = [(p, t) for p, t in ticks if t is not None]
    pxs = [p for p, _ in ticks]
    span = (max(pxs) - min(pxs)) if len(pxs) > 1 else 0.0
    res = max(abs(p - (slope * t + intercept)) for p, t in ticks) if ticks else 0.0
    residual = res / span if span > 0 else 1.0
    # the label grid, in pixel order: whole multiples of the smallest value
    # step (labels may be missing, v3); on a log axis, labels are d x 10^k
    order = [t for _, t in sorted(ticks)]
    steps = [b - a for a, b in zip(order, order[1:], strict=False)]
    grid_dev = 0.0
    base = None
    if scale == "log" and steps:
        base = float(np.median(steps))
        grid_dev = sum(1 for _, v in axis["ticks"] if not _one_digit(float(v))) / len(ticks)
    elif steps:
        nonzero = [s for s in steps if s != 0]
        base = min(nonzero, key=abs) if nonzero else 0.0
        if base == 0 or len(nonzero) < len(steps):
            grid_dev = 1.0
        else:
            for s in steps:
                q = s / base
                grid_dev = max(grid_dev, 1.0 if round(q) <= 0 else abs(q - round(q)))
    direction_ok = slope > 0 if which == "x" else slope < 0
    marks = sorted(float(m) for m in marks or [])
    on_marks = None
    if len(marks) >= 2 and ticks:
        tol = max(3.0, 0.01 * span)
        on_marks = sum(1 for p in pxs if min(abs(p - m) for m in marks) <= tol) / len(pxs)
    on_grid = None
    # v3: the detected marks speak for the scale only when they are this
    # axis's tick marks (the labels sit on them)
    if len(marks) >= 3 and base and on_marks is not None and on_marks >= MARKS_AT_LABELS:
        if scale == "log":
            on_grid = _log_marks_on_grid(marks, slope, intercept)
        else:
            on_grid = _lattice_share(marks, abs(slope * base))
    return {"ok": True, "scale": scale, "n_ticks": len(ticks), "residual": round(residual, 5),
            "grid_dev": round(grid_dev, 4), "direction_ok": direction_ok,
            "marks_on_grid": None if on_grid is None else round(on_grid, 4),
            "labels_on_marks": None if on_marks is None else round(on_marks, 4)}


def calibration_signals(cal: dict, tick_marks: dict | None) -> dict:
    """Self-consistency of a calibration ({"x": {"scale", "ticks", "fit"?},
    "y": ...}) and the tick marks detected on the frame (pixels)."""
    marks = tick_marks or {}
    return {ax: _axis_signals(cal[ax], ax, marks.get(ax)) if isinstance(cal.get(ax), dict)
            else {"ok": False, "n_ticks": 0} for ax in ("x", "y")}


# ------------------------------------------------------------------ verdict


@dataclass(frozen=True)
class Thresholds:
    """Accept / redo limits, measured on the development split
    (scripts/eval/orchestrator/verify_dev.py; docs/design/local-model.md
    「検証とやり直し」). Only ``min_score`` / ``min_score_image`` and the
    calibration limits decide; the rest only raise hints that say what to fix."""

    min_score: float = 0.75  # score with the detector's peaks (detector_agreement)
    min_score_image: float = 0.85  # score from the image signals alone (no detector cache)
    max_residual: float = 0.02  # calibration: of the labelled tick span
    max_grid_dev: float = 0.5
    min_marks_on_grid: float = 0.5
    min_labels_on_marks: float = 0.0
    # hints
    min_contrast: float = 0.35  # per series, over its testifying points
    max_unexplained_share: float = 0.25  # missed blobs + look-alikes / (them + points)
    max_duplicate_share: float = 0.2
    max_outside_share: float = 0.2

    def as_dict(self) -> dict:
        return asdict(self)


def _missed(signals: dict) -> dict:
    return signals.get("missed") or signals["unexplained"]


def image_score(signals: dict) -> float:
    """A point-F1-like estimate from the image signals alone: precision from
    each series' hit contrast (normalised by 1 - null), discounted by
    duplicates and points outside the frame; recall from the missed blobs /
    look-alikes."""
    n = signals["n_points"]
    if n == 0:
        return 0.0
    num = 0.0
    for s in signals["series"]:
        if s["n"] == 0:
            continue
        if s["hit"] is None:
            q = 0.5
        else:
            q = min(1.0, max(0.0, (s["hit"] - s["null"]) / max(1e-6, 1 - s["null"])))
        q *= 1 - (s["duplicates"] + s["outside"]) / s["n"]
        num += s["n"] * max(0.0, q)
    num -= signals.get("cross_duplicates", 0)
    p = max(0.0, num / n)
    tp = p * n
    u = _missed(signals)["n"]
    rec = tp / (tp + u) if tp + u > 0 else 0.0
    return 0.0 if p + rec == 0 else 2 * p * rec / (p + rec)


def score(signals: dict) -> float:
    """The scalar that decides: with the detector's peaks, their agreement
    (detector_agreement f1) times the share of marker-like places the answer
    explains (1 - missed share); else image_score. The form was chosen on the
    development split as the best within-figure ranking of tool results."""
    if signals["n_points"] == 0:
        return 0.0
    det = signals.get("detector")
    if det is None:
        return image_score(signals)
    return det["f1"] * (1 - _missed(signals)["share"])


def hints(signals: dict, th: Thresholds) -> list[str]:
    """What looks wrong, for the orchestrator to fix (does not decide)."""
    out = []
    if signals["n_points"] == 0:
        return ["no points"]
    for s in signals["series"]:
        if s["n"] == 0:
            continue
        if s["contrast"] is not None and s["contrast"] < th.min_contrast:
            out.append(f"series {s['index']}: points not on markers "
                       f"(hit {s['hit']:.2f} vs shifted {s['null']:.2f})")
        if s["duplicates"] / s["n"] > th.max_duplicate_share:
            out.append(f"series {s['index']}: {s['duplicates']} duplicate points")
        if s["outside"] / s["n"] > th.max_outside_share:
            out.append(f"series {s['index']}: {s['outside']} points outside the frame")
    if signals.get("cross_duplicates"):
        out.append(f"{signals['cross_duplicates']} points sit on a point of another series")
    m = _missed(signals)
    if m["share"] > th.max_unexplained_share:
        u = signals["unexplained"]
        la = signals.get("lookalikes", {}).get("n", 0)
        out.append(f"{m['n']} unexplained marker-like places ({u['n']} blobs, "
                   f"{u['same_color']} in a series colour; {la} look like answered "
                   "points) -- markers missed?")
    det = signals.get("detector")
    if det is not None:
        if det["support"] < 0.8:
            out.append(f"{1 - det['support']:.0%} of the points have no detector peak "
                       "(text, lines, legend?)")
        if det["coverage"] < 0.8:
            out.append(f"{len(det['uncovered'])} confident detector peaks have no point "
                       "(missed markers or series?)")
    return out


def calibration_reasons(cal_signals: dict | None, th: Thresholds) -> list[str]:
    reasons = []
    for ax, c in (cal_signals or {}).items():
        if not c.get("ok"):
            reasons.append(f"calibration {ax}: unusable")
            continue
        if not c["direction_ok"]:
            reasons.append(f"calibration {ax}: values run the wrong way")
        if c["residual"] > th.max_residual:
            reasons.append(f"calibration {ax}: tick labels off the fitted line "
                           f"(residual {c['residual']:.3f})")
        if c["grid_dev"] > th.max_grid_dev:
            reasons.append(f"calibration {ax}: tick values not on one grid")
        if c["marks_on_grid"] is not None and c["marks_on_grid"] < th.min_marks_on_grid:
            reasons.append(f"calibration {ax}: tick marks fall between grid values "
                           f"({c['marks_on_grid']:.2f}) -- wrong scale?")
        if c["labels_on_marks"] is not None and c["labels_on_marks"] < th.min_labels_on_marks:
            reasons.append(f"calibration {ax}: labels not at tick marks "
                           f"({c['labels_on_marks']:.2f})")
    return reasons


def points_accept(signals: dict, th: Thresholds) -> bool:
    limit = th.min_score if signals.get("detector") is not None else th.min_score_image
    return signals["n_points"] > 0 and score(signals) >= limit


def verdict(signals: dict, th: Thresholds | None = None) -> dict:
    """accept / redo, from `signals` (verify's output): the score against its
    measured threshold, and the calibration checks. ``reasons`` say why a
    redo; ``hints`` what looks wrong either way."""
    th = th or Thresholds()
    sc = score(signals)
    reasons = []
    if not points_accept(signals, th):
        limit = th.min_score if signals.get("detector") is not None else th.min_score_image
        reasons.append("no points" if signals["n_points"] == 0
                       else f"score {sc:.2f} below {limit}")
    reasons += calibration_reasons(signals.get("calibration"), th)
    return {"accept": not reasons, "reasons": reasons, "hints": hints(signals, th),
            "score": round(sc, 4)}


