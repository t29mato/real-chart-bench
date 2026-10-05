"""Pure helpers for the materials-style synthetic chart generator
(scripts/train/gen_synth_materials.py, docs/design/local-model.md).

No matplotlib import here: the generator renders, these helpers only turn
matplotlib's vocabulary (marker codes, display coordinates, tick label text)
into the training label format.
"""

from __future__ import annotations

import re

_MARKERS = {
    "o": "circle",
    "s": "square",
    "^": "triangle", "v": "triangle", "<": "triangle", ">": "triangle",
    "D": "diamond", "d": "diamond",
    "x": "cross", "+": "cross", "X": "cross", "P": "cross",
    "*": "other", "p": "other", "h": "other", "H": "other", "8": "other",
}

_MATHDEFAULT = re.compile(r"^\$\\mathdefault\{(.*)\}\$$")
_POW10 = re.compile(r"^(?:([0-9.]+)\\times)?10\^\{?([\-−]?[0-9]+)\}?$")


def marker_class(code: str) -> str:
    """matplotlib marker code -> the label's marker class."""
    try:
        return _MARKERS[code]
    except KeyError:
        raise ValueError(f"no label class for matplotlib marker {code!r}") from None


def display_to_image(
    x: float, y: float, *, height: float, scale: float = 1.0
) -> tuple[float, float]:
    """matplotlib display coordinates (origin bottom-left, y up) -> image
    pixels (origin top-left, y down). Pixel i spans [i, i+1) in both.

    scale: the figure was drawn `scale` times larger (supersampled) and the
    image is that drawing shrunk by `scale`; height is the drawing's height.
    """
    return x / scale, (height - y) / scale


def parse_tick_text(text: str) -> float | None:
    """The value a printed tick label shows ('300', '−1.5', '$10^{-3}$',
    '$2\\times10^{-1}$'); None when it is not a number."""
    s = text.strip()
    m = _MATHDEFAULT.match(s)
    if m:
        s = m.group(1).strip()
    s = s.replace("−", "-").replace(",", "")
    p = _POW10.match(s)
    if p:
        mant = float(p.group(1)) if p.group(1) else 1.0
        return mant * 10.0 ** int(p.group(2).replace("−", "-"))
    try:
        return float(s)
    except ValueError:
        return None


def inside_any(point: tuple[float, float], boxes: list[tuple[float, float, float, float]]) -> bool:
    """Whether an image point lies in any (x0, y0, x1, y1) box."""
    x, y = point
    return any(x0 <= x <= x1 and y0 <= y <= y1 for x0, y0, x1, y1 in boxes)
