"""VerifiedPairing: an image<->ground-truth pairing that has undergone
manual numeric cross-verification (design §7.19, 司令塔ゲート指示 2026-08-16).

司令塔 decision: real-image evaluation is gated on verification, not on
scale — "量より信頼性。ベンチマークの信用が資産" (reliability over quantity;
the benchmark's credibility is the asset). Only REJECTED entries stay
too, deliberately not deleted: they're an audit trail of due diligence
already performed, so a future worker doesn't re-investigate (and
potentially wrongly accept) the same candidate.

Pure value object — no I/O. Loading the registry file is an adapter
concern (see adapter/verified_pairing_registry.py); the pass/fail gate
itself is a usecase concern (see usecase/real_image_gate.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from real_chart_bench.domain.curve import ScaleType


class VerificationStatus(Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass(frozen=True)
class VerifiedPairing:
    paper_id: str
    figure_id: str
    image_path: str | None
    panel_label: str | None
    x_range: tuple[float, float] | None
    y_range: tuple[float, float] | None
    status: VerificationStatus
    verified_at: str
    evidence: str
    x_scale: ScaleType = ScaleType.LINEAR
    y_scale: ScaleType = ScaleType.LINEAR  # design §7.25
    # design §7.30 (HQ license audit request 2026-08-22): the paper-level
    # license is already recorded in data/manifest/v0/papers.json, but a
    # pairing's own license basis (needed to justify committing a derived
    # crop under data/verified_pairs/crops/, which is active redistribution)
    # was only reachable by a cross-reference, not self-contained/auditable
    # from the registry alone. Recording it here directly (same raw
    # identifier string as papers.json's license_id, e.g. "cc-by") makes
    # each entry independently auditable.
    license_id: str | None = None
    # None: fully includable in the real-image evaluation suite (the normal
    # case). Non-None: the pairing itself IS correct (status stays VERIFIED,
    # it is not a REJECTED/wrong pairing) but the current harness cannot
    # correctly score it. HQ decision 2026-08-19: such pairings are excluded
    # from the real-image suite until the relevant harness gap is closed,
    # tracked as a separate feature task rather than blocking the
    # verified-pair count. (Originally introduced for log-y axis charts,
    # §7.22 -- now that y_scale exists, §7.25, that specific reason no
    # longer applies, but the field stays as the general escape hatch for
    # future harness gaps of the same shape.)
    excluded_reason: str | None = None
