"""Axis calibration from locally OCR'd tick labels, and the checks that turn a
Starrydata curve plus a real figure into a training label
(docs/design/local-model.md, "データ: 実図(Starrydata)").

Why this exists: the benchmark's own axis reads came from Claude/GPT, and a
model's output may not become a training label (local-model.md rule 2). Here
the only inputs are OCR text (Tesseract, run by an adapter) with the pixel
where it sat, and Starrydata's hand-digitized values. Every number in a label
traces to one of the two.

Pipeline, all pure:
1. parse_tick_label: OCR text -> the numbers it can mean. A lost superscript
   ("10^3" read as "103") is kept as both readings; the fit picks.
2. fit_axis: robust (px, value) line per axis, linear or log10, at least
   three agreeing ticks, the misread ones dropped as outliers.
3. candidate_transforms: Starrydata stores SI (V/K, S/m, K); a figure prints
   uV/K, S/cm, degC, 1000/T or log10(sigma). Each candidate maps a stored
   value to the printed one; nothing is fitted -- a whole-decade change of
   unit, K->degC, a reciprocal temperature or a printed log10.
4. inside_fraction / ink_contrast: the self-consistency gate. Projected
   points must land inside the plot frame and on ink (the markers) far more
   often than the same points shifted a little -- a calibration or a pairing
   that is wrong puts them on white paper.
"""

from __future__ import annotations

import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

_MINUS = str.maketrans({"−": "-", "–": "-", "—": "-", "‐": "-", "‑": "-"})
_SUPERSCRIPT = str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789")
_PLAIN = re.compile(r"^-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_POW10_EXPLICIT = re.compile(r"^10\^(-?\d{1,2})$")
_POW10_OCR = re.compile(r"^10(-?\d{1,2})$")
_SUPERSCRIPT_CHARS = set("⁻⁰¹²³⁴⁵⁶⁷⁸⁹")

MIN_TICKS = 3
# Tick-to-line tolerance: 2% of the axis's tick span, at least 2 px. OCR
# boxes centre within a pixel or two of the tick; a misread label (a dropped
# digit, a wrong decimal point) is off by a whole tick spacing or more.
_TOL_FRACTION = 0.02
_TOL_FLOOR_PX = 2.0


@dataclass(frozen=True)
class TickReading:
    kind: str  # "plain" | "pow10"
    value: float


_SCI = re.compile(r"^(-?\d+\.?\d*)\s*[x×X*]\s*10\^?(-?\d{1,2})$")


def parse_tick_label(text: str) -> list[TickReading]:
    """Every number an OCR'd tick label can mean; [] when it is not one.
    v3: a label printed as a x 10^n (OCR: "3.5x104", "3.5×10^4", "1.0×10⁴")
    reads as its value."""
    s = text.strip().translate(_MINUS)
    sci = _SCI.match(s.translate(_SUPERSCRIPT).replace(" ", ""))
    if sci:
        return [TickReading("plain", float(sci.group(1)) * 10.0 ** int(sci.group(2)))]
    if any(c in _SUPERSCRIPT_CHARS for c in s):
        s = s.translate(_SUPERSCRIPT)
        m = re.match(r"^10(-?\d{1,2})$", s)
        return [TickReading("pow10", 10.0 ** int(m.group(1)))] if m else []
    m = _POW10_EXPLICIT.match(s)
    if m:
        return [TickReading("pow10", 10.0 ** int(m.group(1)))]
    out: list[TickReading] = []
    if _PLAIN.match(s):
        out.append(TickReading("plain", float(s)))
    m = _POW10_OCR.match(s)
    if m:
        out.append(TickReading("pow10", 10.0 ** int(m.group(1))))
    return out


@dataclass(frozen=True)
class AxisFit:
    """px = slope * t(value) + intercept, t = identity or log10."""

    scale: str  # "linear" | "log"
    slope: float
    intercept: float
    ticks: tuple[tuple[float, float], ...]  # (px, value) inliers, as printed
    residual_px: float
    family: str = "plain"

    def _t(self, value: float) -> float | None:
        if self.scale == "log":
            return math.log10(value) if value > 0 else None
        return value

    def value_to_px(self, value: float) -> float | None:
        t = self._t(value)
        return None if t is None else self.slope * t + self.intercept

    def px_to_value(self, px: float) -> float:
        t = (px - self.intercept) / self.slope
        return 10**t if self.scale == "log" else t


def _fit_family(points: list[tuple[float, float]], scale: str, direction: int,
                plausible_log: bool = False):
    if scale == "log":
        points = [(px, v) for px, v in points if v > 0]
        tv = [math.log10(v) for _, v in points]
    else:
        tv = [v for _, v in points]
    if len(points) < MIN_TICKS:
        return None
    pxs = [p for p, _ in points]
    tol = max(_TOL_FLOOR_PX, _TOL_FRACTION * (max(pxs) - min(pxs)))
    best = None
    for i in range(len(points)):
        for j in range(i + 1, len(points)):
            if tv[i] == tv[j] or pxs[i] == pxs[j]:
                continue
            a = (pxs[j] - pxs[i]) / (tv[j] - tv[i])
            if (a > 0) != (direction > 0):
                continue
            b = pxs[i] - a * tv[i]
            inl = [k for k in range(len(points)) if abs(a * tv[k] + b - pxs[k]) <= tol]
            if len({tv[k] for k in inl}) != len(inl):
                continue
            if best is None or len(inl) > len(best):
                best = inl
    if best is None or len(best) < MIN_TICKS:
        return None
    if scale == "linear":
        best = _on_label_grid(best, tv)
        if len(best) < MIN_TICKS:
            return None
    elif plausible_log and not _plausible_log([points[k][1] for k in best]):
        return None
    x = np.array([tv[k] for k in best])
    y = np.array([pxs[k] for k in best])
    a, b = np.polyfit(x, y, 1)
    if (a > 0) != (direction > 0):
        return None
    resid = float(np.max(np.abs(a * x + b - y)))
    ticks = tuple(sorted((pxs[k], points[k][1]) for k in best))
    return AxisFit(scale, float(a), float(b), ticks, resid)


_GRID_TOL = 0.02


def _one_significant_digit(v: float) -> bool:
    if v <= 0:
        return False
    m = v / 10 ** math.floor(math.log10(v) + 1e-9)
    return abs(m - round(m)) <= 1e-6 * max(1.0, m)


def _plausible_log(values: Sequence[float]) -> bool:
    """A log axis spans at least a decade, or prints d x 10^k labels (1, 2,
    5, 10, 20 ...); labels such as 30, 25, 20, 14 over a third of a decade
    are misreads of a linear axis (v3)."""
    vals = [v for v in values if v > 0]
    if not vals:
        return False
    span = math.log10(max(vals)) - math.log10(min(vals))
    return span >= 0.99 or all(_one_significant_digit(v) for v in vals)


def touches_border(box: Sequence[float], size: Sequence[int], margin: float = 1.0) -> bool:
    """Whether a word box (x0, y0, x1, y1) touches the image border (w, h): a
    tick label cut by the border reads as another number ("350" -> "50"), so
    it is left out and the axis may become unreadable rather than guessed."""
    x0, y0, x1, y1 = box
    w, h = size
    return x0 <= margin or y0 <= margin or x1 >= w - 1 - margin or y1 >= h - 1 - margin


_NUMBER_TEXT = re.compile(r"^(-?)(\d+)(\.\d+)?$")


def apply_glyphs(text: str, glyphs: dict) -> str:
    """The OCR text with what the label's ink shows and the OCR dropped (from
    axis_frame.label_glyphs): a leading minus sign, a decimal point after
    `dot_after` digits. Other text is returned unchanged."""
    m = _NUMBER_TEXT.match(text.strip().translate(_MINUS))
    if not m:
        return text
    sign, digits, frac = m.group(1), m.group(2), m.group(3) or ""
    if glyphs.get("minus") and not sign:
        sign = "-"
    k = glyphs.get("dot_after")
    if not frac and k is not None and 0 < k < len(digits):
        digits, frac = digits[:k], "." + digits[k:]
    return f"{sign}{digits}{frac}"


def _on_label_grid(idx: list[int], tv: list[float]) -> list[int]:
    """Linear-axis labels sit on one grid v0 + k*step. Keep the largest set
    of ticks that does: a misread digit ("809" for 800) can pass the pixel
    test but not this one. Labels may be missing, so k need not be
    consecutive."""
    vals = sorted({tv[k] for k in idx})
    steps = {b - a for a, b in zip(vals, vals[1:], strict=False) if b - a > 0}
    best: list[int] = idx[:0]
    for step in steps:
        for anchor in vals:
            keep = [k for k in idx
                    if abs((tv[k] - anchor) / step - round((tv[k] - anchor) / step)) <= _GRID_TOL]
            kv = [tv[k] for k in keep]
            # at most half the grid positions between the end labels unlabelled
            if (max(kv) - min(kv)) / step + 1 > 2 * len(keep):
                continue
            if len(keep) > len(best):
                best = keep
    return best


def fit_axis(
    readings: Sequence[tuple[float, Sequence[TickReading]]], *, direction: int,
    allow_reversed: bool = False, plausible_log: bool = False,
) -> AxisFit | None:
    """Best (px, value) line through OCR'd ticks. direction=+1 when pixels grow
    with value (x axis), -1 when they shrink (y axis, image y points down).

    Ranked by agreeing ticks; on a tie a powers-of-ten reading beats a plain
    one ("102 103 104" fits a linear axis perfectly, but nobody prints that)
    and linear beats log."""
    dirs = (direction, -direction) if allow_reversed else (direction,)
    for d in dirs:  # v3 (allow_reversed): a reversed axis when nothing else fits
        cands = []
        for family in ("pow10", "plain"):
            pts = [(float(px), r.value) for px, rs in readings for r in rs if r.kind == family]
            for scale in ("linear", "log"):
                if family == "pow10" and scale == "linear":
                    continue
                fit = _fit_family(pts, scale, d, plausible_log)
                if fit is not None:
                    fit = AxisFit(fit.scale, fit.slope, fit.intercept, fit.ticks,
                                  fit.residual_px, family)
                    rank = (len(fit.ticks), family == "pow10", scale == "linear",
                            -fit.residual_px)
                    cands.append((rank, fit))
        if cands:
            return max(cands, key=lambda c: c[0])[1]
    return None


@dataclass(frozen=True)
class Transform:
    """Stored (Starrydata, SI) value -> value as printed on the axis."""

    kind: str  # scale | k_to_degc | reciprocal | log10 | shift
    param: int = 0

    @property
    def name(self) -> str:
        return {
            "scale": f"x1e{self.param}",
            "k_to_degc": "K->degC",
            "reciprocal": f"1e{self.param}/v",
            "log10": f"log10(v)+{self.param}",
            "shift": f"v+{self.param}",
        }[self.kind]

    def apply(self, v: float) -> float | None:
        if self.kind == "scale":
            return v * 10.0**self.param
        if self.kind == "k_to_degc":
            return v - 273.15
        if self.kind == "reciprocal":
            return 10.0**self.param / v if v != 0 else None
        if self.kind == "log10":
            return math.log10(v) + self.param if v > 0 else None
        if self.kind == "shift":
            return v + self.param
        raise ValueError(self.kind)


def candidate_transforms(prop: str, unit: str) -> list[Transform]:
    """The printed forms a stored quantity can take in this corpus. A
    quantity Starrydata already stores as a logarithm only moves by whole
    decades (S/m -> S/cm is -2)."""
    if prop.strip().lower().startswith("log"):
        return [Transform("shift", n) for n in range(-6, 7)]
    out = [Transform("scale", k) for k in range(-12, 13)]
    out += [Transform("log10", k) for k in range(-12, 13)]
    if unit.strip() == "K":
        out.append(Transform("k_to_degc"))
        out += [Transform("reciprocal", k) for k in range(0, 5)]
    return out


def inside_fraction(
    points: Sequence[tuple[float, float]], bbox: Sequence[float], *, margin_px: float
) -> float:
    if not points:
        return 0.0
    x0, y0, x1, y1 = bbox
    n = sum(
        1 for x, y in points
        if x0 - margin_px <= x <= x1 + margin_px and y0 - margin_px <= y <= y1 + margin_px
    )
    return n / len(points)


def _hit_rate(points, ink: np.ndarray, radius: int, dx: float, dy: float) -> float:
    h, w = ink.shape
    hits = 0
    for x, y in points:
        cx, cy = int(round(x + dx)), int(round(y + dy))
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        if ink[max(0, cy - radius) : cy + radius + 1, max(0, cx - radius) : cx + radius + 1].any():
            hits += 1
    return hits / len(points) if points else 0.0


def ink_contrast(
    points: Sequence[tuple[float, float]],
    ink: np.ndarray,
    *,
    radius: int,
    shift_px: float | None = None,
) -> tuple[float, float]:
    """(hit, null): the share of points with ink within `radius` px, and the
    mean share for the same points shifted by `shift_px` in eight directions.
    hit >> null means the points sit on the drawn markers; hit ~ null means
    ink is everywhere (or nowhere) and the points explain nothing."""
    d = shift_px if shift_px is not None else 4 * radius + 4
    hit = _hit_rate(points, ink, radius, 0, 0)
    shifts = [(d, 0), (-d, 0), (0, d), (0, -d), (d, d), (d, -d), (-d, d), (-d, -d)]
    null = sum(_hit_rate(points, ink, radius, sx, sy) for sx, sy in shifts) / len(shifts)
    return hit, null


def _join_words(words):
    """Merge horizontally adjacent boxes on one line ("10" + raised "-3")."""
    ws = sorted(words, key=lambda w: w[1])
    out: list[list] = []
    for t, x0, y0, x1, y1 in ws:
        h = y1 - y0
        for o in out:
            gap = x0 - o[3]
            overlap = min(y1, o[4]) - max(y0, o[2])
            if -1 <= gap <= max(2.0, 0.4 * (o[4] - o[2])) and overlap > 0.2 * min(h, o[4] - o[2]):
                o[0] += t
                o[3] = max(o[3], x1)
                o[2], o[4] = min(o[2], y0), max(o[4], y1)
                break
        else:
            out.append([t, x0, y0, x1, y1])
    return out


def _axis_labels(
    words: Sequence[tuple[str, float, float, float, float]],
    frame: Sequence[float],
    axis: str,
    *,
    ticks: Sequence[float],
) -> list[tuple[float, str, list[TickReading]]]:
    """(px, text, readings) for the numeric labels of one axis of `frame`.

    x: labels in a band under the x axis, at their horizontal centre.
    y: labels in the column hugging the y axis on its left, at their
    vertical centre. A label within snapping distance of a detected tick
    mark takes the tick's pixel."""
    x0, y0, x1, y1 = frame
    fw, fh = x1 - x0, y1 - y0
    words = [w[:5] for w in words]
    if axis == "x":
        band = [w for w in words
                if y1 - 3 <= w[2] <= y1 + 0.2 * fh + 10 and x0 - 0.1 * fw <= (w[1] + w[3]) / 2
                <= x1 + 0.1 * fw]
        joined = _join_words(band)
        if joined:  # keep only the first text line under the axis
            top = min(w[2] for w in joined if parse_tick_label(w[0])) if any(
                parse_tick_label(w[0]) for w in joined) else 0
            joined = [w for w in joined if w[2] <= top + 0.6 * (w[4] - w[2]) + 2]
        centres = [((w[1] + w[3]) / 2, w[0]) for w in joined]
    else:
        band = [w for w in words
                if w[3] <= x0 + 3 and y0 - 0.1 * fh <= (w[2] + w[4]) / 2 <= y1 + 0.1 * fh]
        joined = [w for w in _join_words(band) if parse_tick_label(w[0])]
        if joined:  # the label column: right edges near the rightmost one
            right = max(w[3] for w in joined)
            joined = [w for w in joined if right - w[3] <= max(8.0, 0.6 * (w[3] - w[1]) + 6)]
        centres = [((w[2] + w[4]) / 2, w[0]) for w in joined]
    spacing = np.diff(sorted(ticks)) if len(ticks) > 1 else np.array([])
    snap = max(3.0, 0.3 * float(np.median(spacing))) if spacing.size else 0.0
    out = []
    for c, text in sorted(centres):
        rs = parse_tick_label(text)
        if not rs:
            continue
        if ticks:
            near = min(ticks, key=lambda t: abs(t - c))
            if abs(near - c) <= snap:
                c = near
        out.append((float(c), text, rs))
    return out


def readings_for_axis(
    words: Sequence[tuple[str, float, float, float, float]],
    frame: Sequence[float],
    axis: str,
    *,
    ticks: Sequence[float],
) -> list[tuple[float, list[TickReading]]]:
    """(px, readings) for the numeric labels of one axis of `frame`.

    x: labels in a band under the x axis, at their horizontal centre.
    y: labels in the column hugging the y axis on its left, at their
    vertical centre. A label within snapping distance of a detected tick
    mark takes the tick's pixel."""
    return [(c, rs) for c, _, rs in _axis_labels(words, frame, axis, ticks=ticks)]


def labels_for_axis(
    words: Sequence[tuple[str, float, float, float, float]],
    frame: Sequence[float],
    axis: str,
    *,
    ticks: Sequence[float],
) -> list[tuple[float, str]]:
    """The same labels as readings_for_axis, as (px, OCR text)."""
    return [(c, t) for c, t, _ in _axis_labels(words, frame, axis, ticks=ticks)]


# Gate constants for one axis's transform (see local-model.md for how they
# were set on the held-out handful): nearly every stored point must project
# inside the frame (2% margin -- markers straddle the frame edge), and the
# stored values must span a real share of the frame, or the frame is not the
# one they were digitized from.
MIN_INSIDE = 0.9
FRAME_MARGIN_FRACTION = 0.02
# x: same band as pairing_checks' coverage rule (GT_span / L, p5 = 0.376
# over 202 verified axes). At 0.2 three wrong pairings passed on the
# held-out handful, each with x squeezed into a fifth of the frame. y stays
# at 0.2: Starrydata often digitizes only some series, so the stored y
# values legitimately cover a small part of the axis (22807 Fig. 5c).
MIN_COVERAGE_X = 0.35
MIN_COVERAGE_Y = 0.2
NULL_SHIFT_FRACTION = 0.04
# Ink search radius: 1% of the frame's shorter side, at least 2 px. A hollow
# marker's centre is paper; its ring sits about a marker radius away, and
# markers scale with the frame (2 px missed open circles in 600 px frames).
INK_RADIUS_FRACTION = 0.01


def ink_radius(frame: Sequence[float]) -> int:
    x0, y0, x1, y1 = frame
    return max(2, int(round(INK_RADIUS_FRACTION * min(x1 - x0, y1 - y0))))


def frame_band(frame: Sequence[float], radius: int) -> int:
    """Half-width erased around each frame line: the line, inward ticks
    (up to ~1.5% of the frame) and the search radius."""
    x0, y0, x1, y1 = frame
    return radius + max(2, int(round(0.015 * min(x1 - x0, y1 - y0))))


def null_shift(frame: Sequence[float], radius: int) -> float:
    x0, y0, x1, y1 = frame
    return max(4 * radius + 4, NULL_SHIFT_FRACTION * min(x1 - x0, y1 - y0))


@dataclass(frozen=True)
class Projection:
    x_transform: Transform
    y_transform: Transform
    points_px: list  # per curve: [(px, py), ...]
    points_value: list  # per curve: [(x, y), ...] as printed
    inside: float
    hit: float
    null: float

    @property
    def contrast(self) -> float:
        return self.hit - self.null


def _axis_options(values, fit: AxisFit, lo_px: float, hi_px: float, transforms, min_cov):
    span = hi_px - lo_px
    margin = FRAME_MARGIN_FRACTION * span
    out = []
    for t in transforms:
        pv = [t.apply(v) for v in values]
        px = [None if p is None else fit.value_to_px(p) for p in pv]
        ok = [p for p in px if p is not None]
        if len(ok) < max(2, 0.9 * len(values)):
            continue
        inside = sum(1 for p in ok if lo_px - margin <= p <= hi_px + margin) / len(values)
        if inside < MIN_INSIDE or (max(ok) - min(ok)) / span < min_cov:
            continue
        out.append((t, pv, px, inside))
    return out


def without_frame_lines(ink: np.ndarray, frame: Sequence[float], band: int) -> np.ndarray:
    """Ink with the frame's own lines (and the inward ticks on them) erased:
    a point projected onto an axis line is not evidence of a marker. Points
    stored as y=0 on a y-from-zero axis, or a wrong transform that squashes
    everything onto an axis, otherwise score as hits."""
    out = ink.copy()
    h, w = out.shape
    x0, y0, x1, y1 = (int(round(v)) for v in frame)
    xa, xb = max(0, x0 - band), min(w, x1 + band + 1)
    ya, yb = max(0, y0 - band), min(h, y1 + band + 1)
    for r in (y0, y1):
        out[max(0, r - band) : min(h, r + band + 1), xa:xb] = False
    for c in (x0, x1):
        out[ya:yb, max(0, c - band) : min(w, c + band + 1)] = False
    return out


def away_from_frame_lines(points, frame: Sequence[float], band: int) -> list:
    """The points that can testify: those not within `band` px of a frame
    line. A real point at y=0 on a from-zero axis sits on the line, where
    erased ink can neither confirm nor refute it, so it is not scored."""
    x0, y0, x1, y1 = frame
    return [(x, y) for x, y in points
            if min(abs(x - x0), abs(x - x1)) > band and min(abs(y - y0), abs(y - y1)) > band]


def scored_contrast(points, ink_clean: np.ndarray, frame, radius: int) -> tuple[float, float]:
    """(hit, null) over the testifying points; (0, 0) when fewer than half of
    the points can testify -- a squashed-onto-the-axis projection proves
    nothing."""
    band = frame_band(frame, radius)
    pts = away_from_frame_lines(points, frame, band)
    if not points or len(pts) < 0.5 * len(points):
        return 0.0, 0.0
    return ink_contrast(pts, ink_clean, radius=radius, shift_px=null_shift(frame, radius))


def best_projection(
    curves: Sequence[tuple[Sequence[float], Sequence[float]]],
    x_fit: AxisFit,
    y_fit: AxisFit,
    frame: Sequence[float],
    ink: np.ndarray,
    x_transforms: Sequence[Transform],
    y_transforms: Sequence[Transform],
    *,
    radius: int | None = None,
) -> Projection | None:
    """The (x, y) transform pair under which the stored curves sit in the
    frame and on the ink; None when no pair keeps them in the frame.

    The null shift scales with the frame (4% of its shorter side): a big
    marker or a thick connecting line drawn in a big frame must not count
    as "ink everywhere"."""
    x0, y0, x1, y1 = frame
    radius = ink_radius(frame) if radius is None else radius
    ink = without_frame_lines(ink, frame, frame_band(frame, radius))
    xs = [v for c in curves for v in c[0]]
    ys = [v for c in curves for v in c[1]]
    xo = _axis_options(xs, x_fit, x0, x1, x_transforms, MIN_COVERAGE_X)
    yo = _axis_options(ys, y_fit, y0, y1, y_transforms, MIN_COVERAGE_Y)
    best = None
    for tx, xpv, xpx, xin in xo:
        for ty, ypv, ypx, yin in yo:
            pts = [(a, b) for a, b in zip(xpx, ypx, strict=True)
                   if a is not None and b is not None]
            hit, null = scored_contrast(pts, ink, frame, radius)
            cand = (hit - null, hit, tx, ty, xpv, xpx, ypv, ypx, min(xin, yin), null)
            if best is None or cand[:2] > best[:2]:
                best = cand
    if best is None:
        return None
    _, hit, tx, ty, xpv, xpx, ypv, ypx, inside, null = best
    points_px, points_value, i = [], [], 0
    for cx, _ in curves:
        n = len(cx)
        points_px.append([(xpx[k], ypx[k]) for k in range(i, i + n)
                          if xpx[k] is not None and ypx[k] is not None])
        points_value.append([(xpv[k], ypv[k]) for k in range(i, i + n)
                             if xpx[k] is not None and ypx[k] is not None])
        i += n
    return Projection(tx, ty, points_px, points_value, inside, hit, null)


# --- 方式C: automatic calibration of a benchmark figure ----------------------


def axis_agreement(fit: AxisFit | None, person_ticks: Sequence[dict], scale: str) -> float:
    """How far an automatic axis fit is from the person's calibration: at each
    of the person's two ticks, |auto value - person value| as a fraction of
    the person's tick span (in decades on a log axis); the worse tick counts.
    inf when there is no fit or the scales differ."""
    if fit is None or fit.scale != scale:
        return math.inf
    (a, b) = person_ticks[:2]

    def t(v: float) -> float:
        return math.log10(v) if scale == "log" else v

    span = abs(t(b["value"]) - t(a["value"]))
    if span == 0:
        return math.inf
    err = 0.0
    for p in (a, b):
        auto = (p["px"] - fit.intercept) / fit.slope  # already in t-space
        err = max(err, abs(auto - t(p["value"])) / span)
    return err


_PIECE = re.compile(r"-?(0|[1-9]\d*)(\.\d+)?")


def split_merged_labels(text: str, *, min_pieces: int = 3) -> list[str] | None:
    """Tesseract reads tightly spaced tick labels as one word
    ("300320340360"). Split such a word into at least `min_pieces` numbers
    that form one arithmetic progression, or None when it does not split
    that way (a single label, or digits that are no tick run). Words shorter
    than 5 characters are never split ("123" is more likely a label)."""
    s = text.strip().translate(_MINUS)
    if len(s) < 5:
        return None

    def pieces_at(i: int):
        for m_end in range(len(s), i, -1):
            m = _PIECE.fullmatch(s, i, m_end)
            if m:
                yield s[i:m_end]

    def walk(i: int, got: list[str], step: float | None):
        if i == len(s):
            return list(got) if len(got) >= min_pieces else None
        for p in pieces_at(i):
            v = float(p)
            if got:
                d = v - float(got[-1])
                if d == 0:
                    continue
                if step is not None and abs(d - step) > 1e-9 * max(1.0, abs(step)):
                    continue
                res = walk(i + len(p), [*got, p], d)
            else:
                res = walk(i + len(p), [p], None)
            if res:
                return res
        return None

    return walk(0, [], None)


_DECADE = re.compile(r"^10(\^?(-?\d{1,2}))?$")


def decade_readings(
    labels: Sequence[tuple[float, str]], *, direction: int
) -> list[tuple[float, list[TickReading]]]:
    """Log-axis labels printed 10^n, whose small raised exponents OCR mostly
    drops ("10") or garbles ("109" for 10^3). Labels that start with "10"
    and sit on one even spacing (a label may be missing) are consecutive
    decades; every exponent that was read votes for the offset (an explicit
    "10^n" counts twice, a merged "10n" once) and the winner numbers them
    all. [] unless at least three such labels, even spacing and a clear
    winner. direction as in fit_axis."""
    cands = []
    for px, text in labels:
        s = text.strip().translate(_MINUS).translate(_SUPERSCRIPT)
        m = _DECADE.match(s)
        if not m:
            continue
        vote = None
        if m.group(2) is not None:
            vote = (int(m.group(2)), 2 if m.group(1).startswith("^") else 1)
        cands.append((float(px), vote))
    if len(cands) < 3:
        return []
    cands.sort(key=lambda c: c[0] * direction)
    pos = [c[0] for c in cands]
    diffs = [abs(b - a) for a, b in zip(pos, pos[1:], strict=False)]
    base = min(diffs)
    if base <= 0:
        return []
    ranks = [0]
    for d in diffs:
        k = d / base
        if abs(k - round(k)) > 0.1:
            return []
        ranks.append(ranks[-1] + round(k))
    votes: dict[int, float] = {}
    for r, (_, v) in zip(ranks, cands, strict=True):
        if v is None:
            continue
        votes[v[0] - r] = votes.get(v[0] - r, 0) + v[1]
        # an unsigned exponent may have lost its minus sign: half a vote for
        # that reading too; it wins only where the signed votes disagree
        if v[0] > 0:
            votes[-v[0] - r] = votes.get(-v[0] - r, 0) + v[1] / 2
    if not votes:
        return []
    top = sorted(votes.values(), reverse=True)
    if len(top) > 1 and top[0] == top[1]:
        return []
    offset = max(votes, key=lambda o: votes[o])
    out = [(px, [TickReading("pow10", 10.0 ** (r + offset))])
           for r, (px, _) in zip(ranks, cands, strict=True)]
    return sorted(out)
