"""Port for splitting a composite multi-panel figure image into labeled
sub-panel images (2026-08-16 技術調査, design §7.10/§7.11).

Deliberately generic (raw bytes in, raw bytes + label out) so this is a
clean, standalone port other consumers — e.g. deep-digitizer, per 司令塔's
2026-08-16 instruction that this becomes a shared component — can implement
or consume without pulling in real-chart-bench's own domain vocabulary
(Curve, FigureRecord, etc. are intentionally absent from this module).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class SplitPanel:
    label: str  # "a", "b", "c", ... in row-major reading order
    image_bytes: bytes


class PanelSplitterPort(Protocol):
    def split(self, image_bytes: bytes) -> list[SplitPanel]:
        """Returns one SplitPanel per detected panel. A composite image
        with no confidently-detected internal grid returns a single
        SplitPanel(label="a", ...) wrapping the original image unchanged —
        callers should not assume len(result) > 1 implies a real split
        happened, but they can always safely use result[0] as a fallback
        "whole image" panel."""
        ...
