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
from real_chart_bench.domain.verified_pairing import (
    GtSuspectStatus,
    RejectionCategory,
    RejectionEvidence,
    VerificationStatus,
    VerifiedPairing,
)


def _parse_range(raw: list[float] | None) -> tuple[float, float] | None:
    if raw is None:
        return None
    return (float(raw[0]), float(raw[1]))


def _parse_rejection_evidence(raw: dict[str, Any] | None) -> RejectionEvidence | None:
    if raw is None:
        return None
    return RejectionEvidence(
        axis_range_mismatch=raw.get("axis_range_mismatch"),
        point_count_mismatch=raw.get("point_count_mismatch"),
        y_value_offset_magnitude=raw.get("y_value_offset_magnitude"),
        missing_series=raw.get("missing_series"),
    )


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
        y_scale=ScaleType(raw["y_scale"]) if "y_scale" in raw else ScaleType.LINEAR,
        excluded_reason=raw.get("excluded_reason"),
        license_id=raw.get("license_id"),
        rejection_category=(
            RejectionCategory(raw["rejection_category"])
            if raw.get("rejection_category") is not None
            else None
        ),
        gt_suspect_status=(
            GtSuspectStatus(raw["gt_suspect_status"])
            if raw.get("gt_suspect_status") is not None
            else None
        ),
        rejection_evidence=_parse_rejection_evidence(raw.get("rejection_evidence")),
    )


def parse_registry(raw_entries: list[dict[str, Any]]) -> list[VerifiedPairing]:
    return [_parse_entry(entry) for entry in raw_entries]


def load_registry(path: Path) -> list[VerifiedPairing]:
    raw_entries = json.loads(path.read_text())
    return parse_registry(raw_entries)


def _serialize_range(value: tuple[float, float] | None) -> list[float] | None:
    if value is None:
        return None
    return [value[0], value[1]]


def _serialize_rejection_evidence(evidence: RejectionEvidence | None) -> dict[str, Any] | None:
    if evidence is None:
        return None
    return {
        "axis_range_mismatch": evidence.axis_range_mismatch,
        "point_count_mismatch": evidence.point_count_mismatch,
        "y_value_offset_magnitude": evidence.y_value_offset_magnitude,
        "missing_series": evidence.missing_series,
    }


def serialize_entry(
    pairing: VerifiedPairing, base: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Serialises a VerifiedPairing back to a JSON-able dict.

    When ``base`` is given (typically the raw dict this pairing was parsed
    from via ``_parse_entry``), the result is a copy of ``base`` with only
    the fields VerifiedPairing models updated in place. This preserves the
    original key order and any keys the domain model doesn't know about
    (e.g. ``figure_reference``, which isn't part of VerifiedPairing) --
    important for registry.json, where a migration that adds
    rejection_category to a handful of entries should not reshuffle or drop
    anything else in the file. Without ``base``, a fresh dict is built in a
    fixed canonical field order.

    A field whose value is None and that was absent from ``base`` is
    omitted from the output (not written as an explicit null), keeping
    already-migrated and not-yet-migrated entries visually consistent.
    """
    out: dict[str, Any] = dict(base) if base is not None else {}

    out["paper_id"] = pairing.paper_id
    out["figure_id"] = pairing.figure_id
    out["image_path"] = pairing.image_path
    out["panel_label"] = pairing.panel_label
    out["x_range"] = _serialize_range(pairing.x_range)
    out["y_range"] = _serialize_range(pairing.y_range)
    out["x_scale"] = pairing.x_scale.value
    out["y_scale"] = pairing.y_scale.value
    out["status"] = pairing.status.value
    out["verified_at"] = pairing.verified_at
    out["evidence"] = pairing.evidence

    for key, value in (
        ("license_id", pairing.license_id),
        ("excluded_reason", pairing.excluded_reason),
    ):
        if value is not None or key in out:
            out[key] = value

    if pairing.rejection_category is not None:
        out["rejection_category"] = pairing.rejection_category.value
    elif "rejection_category" in out:
        del out["rejection_category"]

    if pairing.gt_suspect_status is not None:
        out["gt_suspect_status"] = pairing.gt_suspect_status.value
    elif "gt_suspect_status" in out:
        del out["gt_suspect_status"]

    serialized_evidence = _serialize_rejection_evidence(pairing.rejection_evidence)
    if serialized_evidence is not None:
        out["rejection_evidence"] = serialized_evidence
    elif "rejection_evidence" in out:
        del out["rejection_evidence"]

    return out


def serialize_registry(
    pairings: list[VerifiedPairing], raw_entries: list[dict[str, Any]] | None = None
) -> list[dict[str, Any]]:
    if raw_entries is None:
        return [serialize_entry(p) for p in pairings]
    return [
        serialize_entry(p, base=raw) for p, raw in zip(pairings, raw_entries, strict=True)
    ]
