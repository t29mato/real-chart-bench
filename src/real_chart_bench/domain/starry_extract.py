"""Symbol Extract and Line Extract, ported from starry-digitizer
(docs/design/local-model.md, 方式D: 司令塔 + 道具).

Ported from starry-digitizer 2.0.0-dev (a13c927), the compiled
``library-build/dist/core.js`` classes named "Symbol Extract" and
"Line Extract" and their base class (matchColor / isOnMask); the source is
``application/strategies/extractStrategies/`` of t29mato/starry-digitizer.

    MIT License

    Copyright (c) 2021 MATO Tomoya

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.

What is kept exactly (tests/domain/test_starry_extract.py checks it against
outputs of the original JavaScript, tests/fixtures/starry_extract_js_cases.json):

- colour match: squared RGB distance / (255^2 * 3) * 100 < distance_pct,
  in the same floating-point order;
- the mask: when a mask is drawn, only pixels on it are examined
  (``isOnMask``: RGBA (255, 255, 0, a > 0)); here the mask is a boolean array,
  True = on the mask, and ``mask_from_rgba`` converts a drawn RGBA mask;
- Symbol Extract: seeds in row-major order, 8-connected flood fill of
  matching pixels, kept when the equivalent diameter sqrt(area / pi) * 2 is
  within [min, max]; centre = mean pixel index + 0.5, rounded as JS
  ``toFixed(1)``;
- Line Extract: seeds in column-major order, the same flood fill limited to
  |dx| <= dx_px and |dy| <= dy_px from the seed, no size filter.

The colour test is vectorised once per image (it is a pure per-pixel
function, so this does not change the result); the flood fill itself runs
in the original order.
"""

from __future__ import annotations

import math
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

Point = tuple[float, float]


def js_to_fixed_1(value: float) -> float:
    """``parseFloat(value.toFixed(1))``: decimal rounding of the exact binary
    value, a tie going away from zero (JS picks the larger n; for a negative
    number toFixed rounds its magnitude)."""
    d = Decimal(value).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return float(d)


def match_color(rgb: tuple[int, int, int], target: tuple[int, int, int], pct: float) -> bool:
    """The original ``matchColor`` for one pixel."""
    s = 0
    for a, b in zip(rgb, target, strict=True):
        s = s + (a - b) ** 2
    return float(s) / (255.0**2 * 3) * 100 < pct


def match_color_mask(rgb: np.ndarray, target: tuple[int, int, int], pct: float) -> np.ndarray:
    """``match_color`` for every pixel of an (H, W, 3) image."""
    d = rgb[..., :3].astype(np.int64) - np.asarray(target, dtype=np.int64)
    s = (d * d).sum(axis=-1).astype(np.float64)
    return s / (255.0**2 * 3) * 100 < pct


def mask_from_rgba(mask_rgba: np.ndarray) -> np.ndarray:
    """A mask drawn as in starry-digitizer (``isOnMask``): yellow, opaque."""
    m = mask_rgba
    return (m[..., 0] == 255) & (m[..., 1] == 255) & (m[..., 2] == 0) & (m[..., 3] > 0)


def _candidates(rgb: np.ndarray, target, pct: float, mask: np.ndarray | None) -> np.ndarray:
    ok = match_color_mask(rgb, tuple(int(v) for v in target), pct)
    if mask is not None:
        if mask.shape != ok.shape:
            raise ValueError(f"mask {mask.shape} does not match image {ok.shape}")
        ok &= mask.astype(bool)
    return ok


def _flood(
    ok: bytearray,
    seen: bytearray,
    h: int,
    w: int,
    sx: int,
    sy: int,
    dx_lim: int | None,
    dy_lim: int | None,
) -> tuple[int, int, int]:
    """Flood fill from a matching seed (already marked seen). Returns
    (number of pixels, sum of x, sum of y). Neighbour order as the original:
    y - 1 .. y + 1 outer, x - 1 .. x + 1 inner."""
    qx, qy = [sx], [sy]
    i = 0
    while i < len(qx):
        cx, cy = qx[i], qy[i]
        for ny in (cy - 1, cy, cy + 1):
            if ny < 0 or ny >= h:
                continue
            if dy_lim is not None and abs(ny - sy) > dy_lim:
                continue
            row = ny * w
            for nx in (cx - 1, cx, cx + 1):
                if nx < 0 or nx >= w:
                    continue
                if dx_lim is not None and abs(nx - sx) > dx_lim:
                    continue
                k = row + nx
                if seen[k]:
                    continue
                if ok[k]:
                    qx.append(nx)
                    qy.append(ny)
                    seen[k] = 1
        i += 1
    return len(qx), sum(qx), sum(qy)


def _prepare(rgb, target, pct, mask):
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected an (H, W, 3) image, got {rgb.shape}")
    h, w = rgb.shape[:2]
    cand = _candidates(rgb, target, pct, mask)
    # every pixel that is not a candidate behaves as already visited: a
    # masked-out pixel is marked visited up front by the original, a
    # non-matching one is skipped (and marked) when it comes up as a seed
    # and never joins a fill.
    return h, w, cand


def symbol_extract(
    rgb: np.ndarray,
    target: tuple[int, int, int],
    distance_pct: float = 1.0,
    mask: np.ndarray | None = None,
    min_diameter_px: float = 5,
    max_diameter_px: float = 100,
) -> list[Point]:
    """starry-digitizer's Symbol Extract: one point per blob of the target
    colour whose equivalent diameter is within [min, max] px."""
    h, w, cand = _prepare(rgb, target, distance_pct, mask)
    ok = bytearray(cand.ravel().astype(np.uint8).tobytes())
    seen = bytearray(h * w)
    out: list[Point] = []
    for k in np.flatnonzero(cand).tolist():  # row-major, as o (y) outer, h (x) inner
        if seen[k]:
            continue
        seen[k] = 1
        y, x = divmod(k, w)
        n, sx, sy = _flood(ok, seen, h, w, x, y, None, None)
        diameter = math.sqrt(n / math.pi) * 2
        if min_diameter_px <= diameter <= max_diameter_px:
            out.append((js_to_fixed_1(sx / n + 0.5), js_to_fixed_1(sy / n + 0.5)))
    return out


def line_extract(
    rgb: np.ndarray,
    target: tuple[int, int, int],
    distance_pct: float = 1.0,
    mask: np.ndarray | None = None,
    dx_px: int = 10,
    dy_px: int = 10,
) -> list[Point]:
    """starry-digitizer's Line Extract: pixels of the target colour sampled
    column by column into patches of at most (2 dx + 1) x (2 dy + 1) around
    each seed, one point per patch."""
    h, w, cand = _prepare(rgb, target, distance_pct, mask)
    ok = bytearray(cand.ravel().astype(np.uint8).tobytes())
    seen = bytearray(h * w)
    out: list[Point] = []
    # column-major: o (x) outer, h (y) inner
    for kt in np.flatnonzero(cand.T).tolist():
        x, y = divmod(kt, h)
        k = y * w + x
        if seen[k]:
            continue
        seen[k] = 1
        n, sx, sy = _flood(ok, seen, h, w, x, y, int(dx_px), int(dy_px))
        out.append((js_to_fixed_1(sx / n + 0.5), js_to_fixed_1(sy / n + 0.5)))
    return out
