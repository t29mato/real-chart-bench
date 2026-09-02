"""Selects the confirmed ground-truth issues to export for upstream
Starrydata cleansing (design §7.48, 戦略メモ「柱G」).

司令塔 rule (owner instruction, repeated here because it is the whole point
of this module): an ``llm_flagged`` GT_SUSPECT entry is a *suspicion*, not a
confirmed error -- VLM readings are themselves error-prone. Only an entry a
human has actually looked at and confirmed
(``GtSuspectStatus.HUMAN_CONFIRMED``) may ever be reported as a ground-truth
error. ``select_confirmed_gt_issues`` below enforces this by delegating
entirely to ``VerifiedPairing.is_confirmed_gt_error`` -- it does not
re-implement the check via a hand-written ``gt_suspect_status ==
GtSuspectStatus.something`` comparison, which is exactly the kind of
call-site mistake that property exists to make impossible.

``scripts/export/gt_issues.py`` is the only intended caller: it loads
registry.json (+ ground_truth.json + papers.json for enrichment), calls the
two functions here, and hands the result to
adapter/gt_issues_export.py for CSV/JSON rendering. Pure transform -- no
file I/O -- so the selection logic is unit-testable on its own, same shape
as usecase/real_image_gate.py and usecase/build_leaderboard.py.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from real_chart_bench.domain.verified_pairing import (
    GtSuspectStatus,
    RejectionCategory,
    RejectionEvidence,
    VerifiedPairing,
)


@dataclass(frozen=True)
class GtIssueProperty:
    """One Starrydata (prop_x, prop_y) pairing + units observed on the
    figure's ground-truth curve(s), as recorded in
    data/verified_pairs/ground_truth.json."""

    prop_x: str
    unit_x: str
    prop_y: str
    unit_y: str


@dataclass(frozen=True)
class GtIssueRow:
    """One confirmed ground-truth issue, ready for CSV/JSON rendering.

    ``gt_suspect_status`` is always ``GtSuspectStatus.HUMAN_CONFIRMED`` by
    construction (see ``select_confirmed_gt_issues``) -- carried explicitly
    on the row anyway so the exported file is self-describing and a
    downstream reader doesn't have to trust that invariant blindly.

    ``properties`` can be empty: ground_truth.json only stores curves for
    VERIFIED (includable) pairings, so a REJECTED gt_suspect entry's
    properties will not be found there. This is a known data-shape
    limitation, not a bug -- see the lookup in select_confirmed_gt_issues.
    """

    paper_id: str
    figure_id: str
    doi: str | None
    properties: tuple[GtIssueProperty, ...]
    gt_suspect_status: GtSuspectStatus
    evidence: str
    rejection_evidence: RejectionEvidence | None
    verified_at: str


@dataclass(frozen=True)
class GtSuspectReviewSummary:
    """Review-lifecycle counts across every GT_SUSPECT-flagged entry in the
    registry, regardless of pairing status (VERIFIED or REJECTED) -- used
    to report the "how many are still awaiting human review" figure the
    script must show loudly even when human_confirmed is 0 (task hard
    requirement 4)."""

    total_gt_suspect: int
    human_confirmed: int
    human_rejected: int
    awaiting_human_review: int


def _properties_for_figure(
    figure_id: str, ground_truth: Mapping[str, Sequence[Mapping[str, Any]]]
) -> tuple[GtIssueProperty, ...]:
    seen: list[GtIssueProperty] = []
    for curve in ground_truth.get(figure_id, []):
        prop = GtIssueProperty(
            prop_x=curve.get("prop_x") or "",
            unit_x=curve.get("unit_x") or "",
            prop_y=curve.get("prop_y") or "",
            unit_y=curve.get("unit_y") or "",
        )
        if prop not in seen:
            seen.append(prop)
    return tuple(seen)


def select_confirmed_gt_issues(
    registry: Sequence[VerifiedPairing],
    *,
    ground_truth: Mapping[str, Sequence[Mapping[str, Any]]] | None = None,
    papers_by_id: Mapping[str, Mapping[str, Any]] | None = None,
) -> list[GtIssueRow]:
    """Returns one GtIssueRow per registry entry with
    ``is_confirmed_gt_error`` True, sorted by (paper_id, figure_id) for a
    stable diff between export runs.

    ``ground_truth`` (data/verified_pairs/ground_truth.json, keyed by
    figure_id) and ``papers_by_id`` (data/manifest/v0/papers.json, keyed by
    paper_id, for DOI enrichment) are both optional and default to empty --
    the human_confirmed filter itself needs neither.
    """
    ground_truth = ground_truth or {}
    papers_by_id = papers_by_id or {}

    rows = [
        GtIssueRow(
            paper_id=pairing.paper_id,
            figure_id=pairing.figure_id,
            doi=(papers_by_id.get(pairing.paper_id) or {}).get("doi"),
            properties=_properties_for_figure(pairing.figure_id, ground_truth),
            gt_suspect_status=pairing.gt_suspect_status,
            evidence=pairing.evidence,
            rejection_evidence=pairing.rejection_evidence,
            verified_at=pairing.verified_at,
        )
        for pairing in registry
        if pairing.is_confirmed_gt_error
    ]
    return sorted(rows, key=lambda r: (r.paper_id, r.figure_id))


def summarize_gt_suspect_review(registry: Sequence[VerifiedPairing]) -> GtSuspectReviewSummary:
    suspects = [p for p in registry if p.rejection_category is RejectionCategory.GT_SUSPECT]
    confirmed = sum(1 for p in suspects if p.is_confirmed_gt_error)
    human_rejected = sum(
        1 for p in suspects if p.gt_suspect_status is GtSuspectStatus.HUMAN_REJECTED
    )
    return GtSuspectReviewSummary(
        total_gt_suspect=len(suspects),
        human_confirmed=confirmed,
        human_rejected=human_rejected,
        awaiting_human_review=len(suspects) - confirmed - human_rejected,
    )
