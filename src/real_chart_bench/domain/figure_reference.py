"""Figure-reference normalization (design §2.3: "図番号表記ゆれ").

Starrydata's ``figure_name`` column and the figure captions we'll extract
from paper PDFs use inconsistent notation for the same figure/panel — e.g.
``"2(a)"``, ``"2a"``, ``"Figure 6(a)"``, ``"Fig 9(a)"``, ``"7_b"`` were all
observed for real panels in the Phase 2 pilot (design §7.9). This module
reduces any of those spellings to a canonical ``FigureReference`` so pairing
can compare them for equality instead of failing on notation drift.

Pure string logic — no I/O.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Optional "Figure "/"Fig "/"Fig. " prefix, digits, then an optional panel
# letter that may be parenthesized, underscore-separated, or bare.
_PATTERN = re.compile(
    r"^(?:fig(?:ure)?\.?\s+)?(?P<number>\d+)\s*(?:[\(_]?\s*(?P<panel>[a-zA-Z])\)?)?$",
    re.IGNORECASE,
)


class UnparseableFigureReferenceError(ValueError):
    """Raised when a raw figure reference doesn't match any known notation
    (e.g. it refers to a table, a supplementary figure, or free text)."""


@dataclass(frozen=True)
class FigureReference:
    number: str
    panel: str | None = None


def normalize_figure_reference(raw: str) -> FigureReference:
    match = _PATTERN.match(raw.strip())
    if not match:
        raise UnparseableFigureReferenceError(f"Cannot parse figure reference: {raw!r}")

    panel = match.group("panel")
    return FigureReference(number=match.group("number"), panel=panel.lower() if panel else None)
