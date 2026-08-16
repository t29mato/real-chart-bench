"""I/O adapter for the VerifiedPairing registry (design §7.19).

The registry file (data/verified_pairs/registry.json) is the audit trail of
manual numeric cross-verification described in domain/verified_pairing.py.
This module only knows how to turn JSON on disk into VerifiedPairing value
objects — it does not decide which entries are usable (that's
usecase/real_image_gate.py).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.verified_pairing import VerificationStatus, VerifiedPairing


def _parse_range(raw: list[float] | None) -> tuple[float, float] | None:
    if raw is None:
        return None
    return (float(raw[0]), float(raw[1]))


def _parse_entry(raw: dict[str, Any]) -> VerifiedPairing:
    return VerifiedPairing(
        paper_id=raw["paper_id"],
        figure_id=raw["figure_id"],
        image_path=raw["image_path"],
        panel_label=raw["panel_label"],
        x_range=_parse_range(raw["x_range"]),
        y_range=_parse_range(raw["y_range"]),
        status=VerificationStatus(raw["status"]),
        verified_at=raw["verified_at"],
        evidence=raw["evidence"],
        x_scale=ScaleType(raw["x_scale"]) if "x_scale" in raw else ScaleType.LINEAR,
    )


def parse_registry(raw_entries: list[dict[str, Any]]) -> list[VerifiedPairing]:
    return [_parse_entry(entry) for entry in raw_entries]


def load_registry(path: Path) -> list[VerifiedPairing]:
    raw_entries = json.loads(path.read_text())
    return parse_registry(raw_entries)
