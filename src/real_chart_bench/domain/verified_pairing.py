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
