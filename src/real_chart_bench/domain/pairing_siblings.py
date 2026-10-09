"""Sibling Starrydata figures (docs/design/pairing-automation.md 12.9).

One printed plot is sometimes digitized twice in Starrydata: as sigma and as
log(sigma) (the log record stores 100*log10(sigma) - 200, a unit-conversion
artefact), or as two figure records of the same curves. Both records describe
the same image, so pairing must propose that image once, for the group, not
once per record. Pure functions over the figures' numbers; no I/O.

Two figures are siblings only when all of these hold (conservative: a false
sibling would hide a real pairing behind another figure's tile):
- same paper and same normalised figure reference ("8a sigma" ~ "8a");
- same x quantity and unit; same y quantity (a "log(...)" prefix is ignored)
  -- two different quantities that both rise with temperature are co-monotone
  but not one plot twice;
- at least half of the smaller figure's curves have a partner with identical
  x values whose y values are identical, or (only when exactly one side is a
  log quantity) strictly monotonically related over at least three points.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

_REL_TOL = 1e-9
_MIN_MONOTONE_POINTS = 3
_MIN_MATCHED_SHARE = 0.5


@dataclass(frozen=True)
class SiblingFigure:
    paper_id: str
    figure_id: str
    reference: str
    prop_x: str
    unit_x: str
    prop_y: str
    unit_y: str
    curves: tuple[tuple[tuple[float, ...], tuple[float, ...]], ...]  # (xs, ys) per curve


def normalise_reference(reference: str) -> str:
    """"Fig. 8(a)", "8a sigma" -> "8a": drop a leading "Fig", keep the first word
    (Starrydata's figure_name often carries the quantity after the number)."""
    words = re.sub(r"^fig(?:ure)?\.?\s*", "", reference.strip().lower()).split()
    return "".join(ch for ch in (words[0] if words else "") if ch.isalnum())


_LOG = re.compile(r"^log(?:10)?\s*\((.*)\)$", re.IGNORECASE)


def _quantity(prop: str) -> tuple[str, bool]:
    """(quantity name, is a logarithm of it)."""
    s = prop.strip()
    m = _LOG.match(s)
    return (m.group(1).strip().lower(), True) if m else (s.lower(), False)


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= _REL_TOL * max(1.0, abs(a), abs(b))


def _same(a: Sequence[float], b: Sequence[float]) -> bool:
    return len(a) == len(b) and all(_close(p, q) for p, q in zip(a, b, strict=True))


def _monotone_related(ya: Sequence[float], yb: Sequence[float]) -> bool:
    """yb is a strictly monotone function of ya (increasing or decreasing)."""
    if len(ya) != len(yb) or len(ya) < _MIN_MONOTONE_POINTS:
        return False
    pairs = sorted(zip(ya, yb, strict=True))
    sign = 0
    for (a0, b0), (a1, b1) in zip(pairs, pairs[1:], strict=False):
        if _close(a0, a1):
            if not _close(b0, b1):
                return False
            continue
        if _close(b0, b1):
            return False
        s = 1 if b1 > b0 else -1
        if sign and s != sign:
            return False
        sign = s
    return sign != 0


def _curves_match(a: SiblingFigure, b: SiblingFigure, log_pair: bool) -> bool:
    small, large = (a, b) if len(a.curves) <= len(b.curves) else (b, a)
    if not small.curves:
        return False
    used: set[int] = set()
    matched = 0
    for xs, ys in small.curves:
        for j, (xs2, ys2) in enumerate(large.curves):
            if j in used or not _same(xs, xs2):
                continue
            if _same(ys, ys2) or (log_pair and _monotone_related(ys, ys2)):
                used.add(j)
                matched += 1
                break
    return matched >= _MIN_MATCHED_SHARE * len(small.curves)


def are_siblings(a: SiblingFigure, b: SiblingFigure) -> bool:
    if a.paper_id != b.paper_id or a.figure_id == b.figure_id:
        return False
    ref = normalise_reference(a.reference)
    if not ref or ref != normalise_reference(b.reference):
        return False
    if (a.prop_x.strip().lower(), a.unit_x.strip()) != (b.prop_x.strip().lower(),
                                                        b.unit_x.strip()):
        return False
    qa, log_a = _quantity(a.prop_y)
    qb, log_b = _quantity(b.prop_y)
    if qa != qb:
        return False
    if log_a == log_b and a.unit_y.strip() != b.unit_y.strip():
        return False
    return _curves_match(a, b, log_pair=log_a != log_b)


def sibling_groups(figures: Sequence[SiblingFigure]) -> list[tuple[str, ...]]:
    """Connected groups (size >= 2) of sibling figures, ids in numeric order;
    groups ordered by their first id."""
    parent = {f.figure_id: f.figure_id for f in figures}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i, a in enumerate(figures):
        for b in figures[i + 1:]:
            if are_siblings(a, b):
                parent[find(a.figure_id)] = find(b.figure_id)
    groups: dict[str, list[str]] = {}
    for f in figures:
        groups.setdefault(find(f.figure_id), []).append(f.figure_id)
    out = [tuple(sorted(g, key=int)) for g in groups.values() if len(g) > 1]
    return sorted(out, key=lambda g: int(g[0]))
