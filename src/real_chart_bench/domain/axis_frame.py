"""Plot frames and tick marks on a binarised figure image (True = ink).

Pure numpy, no codec: the pixel half of the no-LLM axis calibration
(docs/design/local-model.md, "データ: 実図(Starrydata)"); tick_calibration.py
is the number half. Works on a whole multi-panel figure or a page render:
every bottom-left corner where a long horizontal ink line meets a long
vertical one is a plot frame, whatever the panel layout around it.

Out of scope, by design: frames with no drawn axis lines (seaborn-style
despined plots with offset spines), broken axes, and twin right-hand y axes
(only the left axis is calibrated, so curves read against a right axis do
not pair and are left out rather than guessed).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Frame = tuple[float, float, float, float]  # (x0, y0, x1, y1): y-axis x, top, right, x-axis y

_MIN_FRAME_PX = 60
_MIN_FRAME_FRACTION = 0.05
_JOIN_PX = 3  # a line's rows/columns may wobble this much (antialiasing, JPEG)
_MAX_TICK_WIDTH = 4
_MIN_TICK_LEN = 2
# A plot interior is mostly paper. Micrographs, photos and filled maps are
# dark all over and make a frame at every pair of crossing edges.
_MAX_INTERIOR_INK = 0.3
# Axis lines are thin; a thick dark band is a bar, a border or a photo edge.
_MAX_LINE_THICK_PX = 6
_MAX_LINE_THICK_FRACTION = 0.01


@dataclass
class _Line:
    pos: float  # row (horizontal) or column (vertical), centre of thickness
    start: int
    end: int
    thick: int


def _runs(row: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    padded = np.concatenate(([False], row, [False]))
    d = np.diff(padded.astype(np.int8))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(s), int(e) - 1) for s, e in zip(starts, ends, strict=True) if e - s >= min_len]


def _lines(dark: np.ndarray, min_len: int) -> list[_Line]:
    """Long straight runs of ink along axis 1 (rows), merged across rows."""
    open_: list[list] = []  # [first_row, last_row, start, end]
    done: list[list] = []
    for r in range(dark.shape[0]):
        segs = _runs(dark[r], min_len)
        still = []
        for ln in open_:
            match = next(
                (s for s in segs
                 if abs(s[0] - ln[2]) <= _JOIN_PX and abs(s[1] - ln[3]) <= _JOIN_PX),
                None,
            )
            if match is not None and ln[1] == r - 1:
                segs.remove(match)
                ln[1] = r
                ln[2], ln[3] = min(ln[2], match[0]), max(ln[3], match[1])
                still.append(ln)
            else:
                done.append(ln)
        still += [[r, r, s, e] for s, e in segs]
        open_ = still
    done += open_
    return [_Line((a + b) / 2, s, e, b - a + 1) for a, b, s, e in done]


def detect_axis_frames(dark: np.ndarray) -> list[Frame]:
    """Every (x0, y0, x1, y1) plot frame: a vertical y-axis line whose bottom
    meets the left end of a horizontal x-axis line."""
    h, w = dark.shape
    min_w = max(_MIN_FRAME_PX, int(_MIN_FRAME_FRACTION * w))
    min_h = max(_MIN_FRAME_PX, int(_MIN_FRAME_FRACTION * h))
    max_thick = max(_MAX_LINE_THICK_PX, int(_MAX_LINE_THICK_FRACTION * min(h, w)))
    hor = [ln for ln in _lines(dark, min_w) if ln.thick <= max_thick]
    ver = [ln for ln in _lines(dark.T, min_h) if ln.thick <= max_thick]
    frames: list[Frame] = []
    for hl in hor:
        for vl in ver:
            tol = _JOIN_PX + max(hl.thick, vl.thick)
            reach = tol + 15  # an outward tick at the origin extends either line
            if not (hl.start - tol <= vl.pos <= hl.start + reach):
                continue
            if not (hl.pos - tol <= vl.end <= hl.pos + reach):
                continue
            f = (vl.pos, float(vl.start), float(hl.end), hl.pos)
            if f[2] - f[0] < min_w or f[3] - f[1] < min_h:
                continue
            inner = dark[int(f[1]) + max_thick : int(f[3]) - max_thick,
                         int(f[0]) + max_thick : int(f[2]) - max_thick]
            if inner.size == 0 or inner.mean() > _MAX_INTERIOR_INK:
                continue
            if not any(all(abs(a - b) <= 5 for a, b in zip(f, g, strict=True)) for g in frames):
                frames.append(f)
    return frames


def _axis_band(profile_rows: np.ndarray, centre: int) -> tuple[int, int]:
    """First/last index around `centre` whose line coverage exceeds 0.8."""
    lo = hi = centre
    while lo - 1 >= 0 and profile_rows[lo - 1] > 0.8:
        lo -= 1
    while hi + 1 < len(profile_rows) and profile_rows[hi + 1] > 0.8:
        hi += 1
    return lo, hi


def _ticks_along(dark: np.ndarray, line_row: float, a: int, b: int, max_len: int) -> list[float]:
    """Tick columns on a horizontal axis line at `line_row` spanning
    columns a..b (callers transpose for a vertical axis)."""
    h = dark.shape[0]
    c = int(round(line_row))
    cov = np.array([dark[r, a : b + 1].mean() if 0 <= r < h else 0.0
                     for r in range(c - 4, c + 5)])
    if cov.max() <= 0.8:
        return []
    k = int(np.argmax(cov))
    lo, hi = _axis_band(cov, k)
    top, bottom = c - 4 + lo, c - 4 + hi

    def run(x: int, step: int, start: int) -> int:
        n, r = 0, start
        while 0 <= r < h and dark[r, x] and n <= max_len:
            n += 1
            r += step
        return n

    cols = []
    for x in range(a + _JOIN_PX + 1, b - _JOIN_PX):
        below, above = run(x, 1, bottom + 1), run(x, -1, top - 1)
        if _MIN_TICK_LEN <= below <= max_len or _MIN_TICK_LEN <= above <= max_len:
            # a tall thing (a bar, a curve) standing on the axis is not a tick
            if below <= max_len and above <= max_len:
                cols.append(x)
    ticks, group = [], []
    for x in cols + [None]:
        if group and (x is None or x != group[-1] + 1):
            if len(group) <= _MAX_TICK_WIDTH:
                ticks.append(sum(group) / len(group))
            group = []
        if x is not None:
            group.append(x)
    return ticks


def detect_ticks(dark: np.ndarray, frame: Frame) -> tuple[list[float], list[float]]:
    """Tick positions (x pixels on the x axis, y pixels on the y axis), the
    frame's own corners excluded. Inward and outward ticks both count."""
    x0, y0, x1, y1 = frame
    max_len = max(4, int(0.04 * min(dark.shape)))
    xt = _ticks_along(dark, y1, int(round(x0)), int(round(x1)), max_len)
    yt = _ticks_along(dark.T, x0, int(round(y0)), int(round(y1)), max_len)
    return xt, yt


def _outward(dark: np.ndarray, line: float, a: int, b: int, max_len: int) -> int:
    c = int(round(line))
    h = dark.shape[0]
    rows = [r for r in range(c, min(h, c + 5)) if dark[r, a : b + 1].mean() > 0.8]
    if not rows:
        return 0
    bottom = max(rows)
    best = 0
    for x in range(a + _JOIN_PX + 1, b - _JOIN_PX):
        n, r = 0, bottom + 1
        while r < h and dark[r, x] and n <= max_len:
            n, r = n + 1, r + 1
        if n <= max_len:
            best = max(best, n)
    return best


def outward_tick_extent(dark: np.ndarray, frame: Frame) -> tuple[int, int]:
    """How far ticks stick out below the x axis and left of the y axis, so
    the OCR strips can start past them (a tick read as "1" or "-" corrupts a
    label)."""
    x0, y0, x1, y1 = frame
    max_len = max(4, int(0.04 * min(dark.shape)))
    below = _outward(dark, y1, int(round(x0)), int(round(x1)), max_len)
    # the y axis, mirrored so "outward" (left) becomes "down"
    flipped = dark[:, ::-1].T
    w = dark.shape[1]
    left = _outward(flipped, w - 1 - x0, int(round(y0)), int(round(y1)), max_len)
    return below, left


# --- 方式C: right-hand y axes, merged labels, raised exponents ------------------------------


def mirror_frame(frame: Frame, width: int) -> Frame:
    """The frame in the left-right mirrored image (and back: an involution)."""
    x0, y0, x1, y1 = frame
    return (width - 1 - x1, y0, width - 1 - x0, y1)


def detect_right_axis_frames(dark: np.ndarray) -> list[Frame]:
    """Frames whose only y axis is drawn on the right (a bottom-right L), in
    image coordinates: x1 is the y axis. Found as bottom-left frames of the
    mirrored image."""
    w = dark.shape[1]
    return [mirror_frame(f, w) for f in detect_axis_frames(dark[:, ::-1])]


def split_at_widest_gaps(
    ink: np.ndarray, k: int, min_ratio: float | None = None
) -> list[tuple[int, int]] | None:
    """Column spans (x0, x1 exclusive) of k labels run together in one OCR
    word: cut at the k-1 widest blank gaps between inked columns. None when
    there are fewer than k-1 gaps, or (min_ratio) when the narrowest cut gap
    is not min_ratio times wider than the widest gap left uncut -- then the
    ink holds a different number of labels."""
    cols = np.flatnonzero(ink.any(axis=0))
    if cols.size == 0 or k < 1:
        return None
    gaps = [(int(b - a - 1), int(a), int(b)) for a, b in zip(cols[:-1], cols[1:], strict=True)
            if b - a > 1]
    if len(gaps) < k - 1:
        return None
    ranked = sorted(gaps, key=lambda g: -g[0])
    if min_ratio is not None and k >= 2 and len(ranked) >= k:
        if ranked[k - 2][0] < min_ratio * ranked[k - 1][0]:
            return None
    cuts = ranked[: k - 1]
    cuts.sort(key=lambda g: g[1])
    spans, start = [], int(cols[0])
    for _, a, b in cuts:
        spans.append((start, a + 1))
        start = b
    spans.append((start, int(cols[-1]) + 1))
    return spans


def exponent_span(ink: np.ndarray) -> tuple[int, int, int, int] | None:
    """Box (x0, y0, x1, y1; ends exclusive) of the raised exponent in the ink
    of a "10^n" label: the columns right of the base whose ink stops well
    above the base's baseline. None when nothing is raised."""
    cols = np.flatnonzero(ink.any(axis=0))
    if cols.size == 0:
        return None
    rows = np.flatnonzero(ink.any(axis=1))
    lowest = {int(c): int(np.flatnonzero(ink[:, c])[-1]) for c in cols}
    baseline = max(lowest.values())
    height = baseline - int(rows[0]) + 1
    base = [c for c in lowest if lowest[c] >= baseline - 0.15 * height]
    raised = [c for c in lowest if c > max(base) and lowest[c] < baseline - 0.25 * height]
    if not raised:
        return None
    x0, x1 = min(raised), max(raised) + 1
    r = np.flatnonzero(ink[:, x0:x1].any(axis=1))
    return (x0, int(r[0]), x1, int(r[-1]) + 1)


def label_glyphs(ink: np.ndarray) -> dict:
    """What a tick label's ink (a boolean crop of one label) shows that the
    OCR may drop (方式D v3): a leading minus sign -- a short, thin, wide
    piece at mid-height left of every digit -- and a decimal point -- a small
    piece on the baseline between digits. ``dot_after`` counts the digits
    left of the point (touching digits are counted by their width)."""
    from scipy import ndimage

    out: dict = {"minus": False, "dot_after": None}
    lab, _ = ndimage.label(np.asarray(ink, bool), structure=np.ones((3, 3), bool))
    boxes = [(sl[1].start, sl[0].start, sl[1].stop, sl[0].stop)
             for sl in ndimage.find_objects(lab) if sl is not None]
    if not boxes:
        return out
    hmax = max(b[3] - b[1] for b in boxes)
    tall = [b for b in boxes if b[3] - b[1] >= 0.5 * hmax]
    top, base = min(b[1] for b in tall), max(b[3] for b in tall)
    height = base - top
    first = min(b[0] for b in tall)
    for b in boxes:
        if b in tall:
            continue
        x0, y0, x1, y1 = b
        w, h = x1 - x0, y1 - y0
        cy = (y0 + y1) / 2
        if (x1 <= first and first - x1 <= 0.8 * height and h <= 0.25 * height
                and w >= 1.5 * h and w >= 0.25 * height
                and abs(cy - (top + base) / 2) <= 0.2 * height):
            out["minus"] = True
        if (x0 > first and h <= 0.3 * height and w <= 0.35 * height and w <= 2 * h + 1
                and base - y1 <= 0.2 * height):
            cx = (x0 + x1) / 2
            out["dot_after"] = sum(max(1, round((t[2] - t[0]) / (0.6 * height)))
                                   for t in tall if (t[0] + t[2]) / 2 < cx)
    return out
