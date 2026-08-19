"""Gate: only VerifiedPairing entries with status=VERIFIED may enter the
real-image evaluation suite (design §7.19, 司令塔ゲート指示 2026-08-16:
"量より信頼性。ベンチマークの信用が資産" — reliability over quantity).

An unverified or explicitly REJECTED candidate is excluded even if it would
otherwise look like a plausible match — see domain/verified_pairing.py for
why rejected entries are kept (not deleted) in the registry.

A VERIFIED pairing with excluded_reason set (design §7.22, HQ decision
2026-08-19) is also excluded from the suite even though its pairing is
correct — the current harness cannot score it yet (e.g. log-y axis charts).
This is a separate concept from REJECTED: the pairing is trustworthy, it's
just not includable in scoring until the relevant harness gap closes.
"""

from __future__ import annotations

from real_chart_bench.domain.verified_pairing import VerificationStatus, VerifiedPairing


def select_verified_pairings(registry: list[VerifiedPairing]) -> list[VerifiedPairing]:
    return [
        p
        for p in registry
        if p.status is VerificationStatus.VERIFIED and p.excluded_reason is None
    ]


def is_verified(registry: list[VerifiedPairing], *, paper_id: str, figure_id: str) -> bool:
    return any(
        p.paper_id == paper_id
        and p.figure_id == figure_id
        and p.status is VerificationStatus.VERIFIED
        for p in registry
    )
