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
