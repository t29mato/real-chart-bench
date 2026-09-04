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

import dataclasses
from dataclasses import dataclass
from enum import Enum

from real_chart_bench.domain.curve import ScaleType


class VerificationStatus(Enum):
    VERIFIED = "verified"
    REJECTED = "rejected"


class RejectionCategory(Enum):
    """Why a candidate pairing was rejected -- or, for a VERIFIED entry, the
    one case where the pairing itself is fine but the ground truth backing
    it is still under suspicion (design §7.48, 戦略メモ「柱G」).

    - PAIRING: wrong figure matched, or a panel boundary/orientation crop
      error -- our bug in the matching/cropping step.
    - IMAGE: the image side could not be used, on our side of the
      pipeline -- unreadable due to resolution/scan quality, or (the same
      "our side, not the data's" failure, just further upstream) no usable
      image could even be found/extracted for the figure.
    - GT_SUSPECT: the Starrydata ground truth itself looks wrong (human
      digitization, axis calibration, or unit-conversion error) -- a
      dataset-side problem, not a pairing/image problem on our side.

    A VERIFIED entry's rejection_category, if set at all, may only be
    GT_SUSPECT: PAIRING/IMAGE describe defects that would make the pairing
    itself untrustworthy, which is a contradiction for a VERIFIED entry.
    GT_SUSPECT is the one category that is orthogonal to pairing
    correctness -- a figure can be correctly matched and cropped, and the
    Starrydata curve digitized against it can still be wrong. See
    VerifiedPairing.__post_init__.
    """

    PAIRING = "pairing"
    IMAGE = "image"
    GT_SUSPECT = "gt_suspect"


class GtSuspectStatus(Enum):
    """Review lifecycle for a GT_SUSPECT flag (design §7.48).

    - LLM_FLAGGED: an LLM/automated check raised the suspicion. Nothing more.
    - HUMAN_CONFIRMED: a human looked at the source figure and confirmed the
      ground truth is wrong.
    - HUMAN_REJECTED: a human looked and the LLM was wrong -- the GT is fine.

    CRITICAL (owner rule): LLM_FLAGGED alone must NEVER be reported as "a GT
    error" -- VLM readings are themselves error-prone. Only HUMAN_CONFIRMED
    counts as a confirmed GT error. Read via ``is_confirmed_gt_error``
    (below) or ``VerifiedPairing.is_confirmed_gt_error`` rather than
    comparing against this enum by hand at call sites, so the rule can't be
    silently gotten wrong by a future ``== GtSuspectStatus.LLM_FLAGGED``-style
    check that means to ask "is this a GT error" but forgets the distinction.
    """

    LLM_FLAGGED = "llm_flagged"
    HUMAN_CONFIRMED = "human_confirmed"
    HUMAN_REJECTED = "human_rejected"

    @property
    def is_confirmed_gt_error(self) -> bool:
        return self is GtSuspectStatus.HUMAN_CONFIRMED


class TickRangeProvenance(Enum):
    """Where a promoted x_tick_range/y_tick_range came from (design §7.57).

    The only member today is OWNER_REVIEWED: VerifiedPairing.promote_tick_range
    refuses to attach a tick range unless the source reading in
    axis_pixel_candidates.json carries status "owner_reviewed" -- see that
    function's docstring, and GtSuspectStatus above for the sibling
    llm_flagged/human_confirmed discipline this mirrors. Modelled as an enum
    (rather than a bare bool) so that if a second, differently-sourced review
    tier is ever introduced, it adds a member here instead of overloading
    what "True" means.
    """

    OWNER_REVIEWED = "owner_reviewed"


@dataclass(frozen=True)
class RejectionEvidence:
    """Structured findings for *what* disagreed between a candidate image
    and the Starrydata ground truth, alongside (not instead of) the
    free-text ``evidence`` string on VerifiedPairing, which is never
    deleted. Every field is optional: populate only what the evidence text
    actually supports for a given entry -- leave the rest ``None`` rather
    than infer/guess a number that was never actually derived.

    - axis_range_mismatch: the candidate's printed/calibrated axis range
      does not agree with the GT curve's x/y range.
    - point_count_mismatch: the GT curve's point count doesn't fit the
      chart (e.g. a dense continuous trace where the chart shows discrete
      markers, or vice versa).
    - y_value_offset_magnitude: how far off the GT's y-values are from the
      chart's, expressed as a unitless ratio (e.g. 100.0 for "off by two
      orders of magnitude") -- deliberately unit-agnostic since the two
      quantities being compared are themselves not always in the same unit.
    - missing_series: the GT curve doesn't correspond to any series
      actually visible on the (correctly identified) chart/panel.
    """

    axis_range_mismatch: bool | None = None
    point_count_mismatch: bool | None = None
    y_value_offset_magnitude: float | None = None
    missing_series: bool | None = None


@dataclass(frozen=True)
class VerifiedPairing:
    paper_id: str
    figure_id: str
    image_path: str | None
    panel_label: str | None
    # x_range / y_range: the extent of the drawn axis FRAME (the plot box),
    # not the printed tick labels -- design §7.57. This is deliberate: GT
    # data routinely lies outside the outermost printed tick (ordinary
    # plotting margin), so a tick-valued calibration would put real data
    # outside the calibrated range. These names are kept as-is for external
    # consumers keyed on them (see docs/interop/README.md) even though
    # "frame_range" would now be the more accurate name. For the printed
    # tick values themselves -- the only axis-reading ground truth a model
    # reading the chart could ever produce, since the frame extent is not
    # printed anywhere -- see x_tick_range / y_tick_range below.
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
    # design §7.48 (戦略メモ「柱G」): distinguishes *why* a REJECTED entry was
    # rejected (pairing/image/gt_suspect), and doubles as a flag a VERIFIED
    # entry can carry to say "the pairing is correct but the GT is
    # suspect" -- see RejectionCategory's docstring for why GT_SUSPECT is
    # the one value allowed on a VERIFIED entry. Deliberately NOT
    # hard-required on every REJECTED entry at construction time (see
    # __post_init__ and needs_rejection_classification below): a handful of
    # already-rejected registry entries have evidence text that genuinely
    # does not point at a single category, and guessing one to satisfy a
    # hard invariant would be worse than leaving it explicitly pending --
    # the registry must also stay loadable while that human review is
    # pending, rather than refusing to parse mid-migration data.
    rejection_category: RejectionCategory | None = None
    # Required iff rejection_category is GT_SUSPECT, forbidden otherwise --
    # enforced in __post_init__.
    gt_suspect_status: GtSuspectStatus | None = None
    # Structured counterpart to the free-text `evidence` string above (kept
    # as-is). Optional and independent of rejection_category: e.g. a
    # PAIRING rejection can still note a point_count_mismatch that helped
    # spot the wrong match.
    rejection_evidence: RejectionEvidence | None = None
    # x_tick_range / y_tick_range: the printed tick-label extent read off
    # the chart (e.g. axis_pixel_candidates.json's x_min_label/x_max_label),
    # as opposed to x_range/y_range above which is the drawn frame extent
    # (design §7.57). Optional and, deliberately, promoted only for the
    # minority of entries whose axis reading has been human-reviewed -- see
    # promote_tick_range below and TickRangeProvenance. None for an entry
    # whose axis reading is still an unreviewed LLM candidate.
    x_tick_range: tuple[float, float] | None = None
    y_tick_range: tuple[float, float] | None = None
    # Required iff x_tick_range or y_tick_range is set, forbidden otherwise
    # -- enforced in __post_init__. See TickRangeProvenance.
    tick_range_source: TickRangeProvenance | None = None

    def __post_init__(self) -> None:
        if (
            self.status is VerificationStatus.VERIFIED
            and self.rejection_category is not None
            and self.rejection_category is not RejectionCategory.GT_SUSPECT
        ):
            raise ValueError(
                "a VERIFIED entry's rejection_category may only be GT_SUSPECT "
                "(the pairing itself is correct, only the GT is in question) "
                "-- PAIRING/IMAGE describe defects that would make the "
                "pairing itself untrustworthy, i.e. it should be REJECTED"
            )

        is_gt_suspect = self.rejection_category is RejectionCategory.GT_SUSPECT
        if is_gt_suspect and self.gt_suspect_status is None:
            raise ValueError(
                "gt_suspect_status is required when rejection_category is GT_SUSPECT"
            )
        if not is_gt_suspect and self.gt_suspect_status is not None:
            raise ValueError(
                "gt_suspect_status is only allowed when rejection_category is GT_SUSPECT "
                f"(got rejection_category={self.rejection_category})"
            )

        # design §7.57: a tick range is a refinement of the frame range for
        # the same axis, so it cannot exist for an axis that has no frame
        # range at all (in practice this never arises: every entry that has
        # a reviewed axis_pixel_candidates.json reading is VERIFIED and
        # VERIFIED entries always carry both frame ranges) -- illegal rather
        # than merely undocumented, so a future migration bug fails loudly
        # instead of silently producing an axis with a tick range but no
        # frame to have refined.
        if self.x_tick_range is not None and self.x_range is None:
            raise ValueError("x_tick_range requires x_range (frame extent) to be set")
        if self.y_tick_range is not None and self.y_range is None:
            raise ValueError("y_tick_range requires y_range (frame extent) to be set")

        has_tick_range = self.x_tick_range is not None or self.y_tick_range is not None
        if has_tick_range and self.tick_range_source is None:
            raise ValueError(
                "tick_range_source is required when x_tick_range or y_tick_range is set"
            )
        if not has_tick_range and self.tick_range_source is not None:
            raise ValueError(
                "tick_range_source is only allowed when x_tick_range or y_tick_range is set"
            )

    @property
    def needs_rejection_classification(self) -> bool:
        """True for a REJECTED entry that has not (yet) been assigned a
        rejection_category -- i.e. pending human adjudication. See the
        rejection_category field comment for why this is a query rather
        than a construction-time error."""
        return self.status is VerificationStatus.REJECTED and self.rejection_category is None

    @property
    def is_confirmed_gt_error(self) -> bool:
        """True only when this entry is flagged GT_SUSPECT *and* a human has
        confirmed the error (GtSuspectStatus.HUMAN_CONFIRMED). An
        LLM_FLAGGED-only entry is never a confirmed GT error -- see
        GtSuspectStatus.is_confirmed_gt_error."""
        return self.gt_suspect_status is not None and self.gt_suspect_status.is_confirmed_gt_error


def promote_tick_range(
    pairing: VerifiedPairing,
    *,
    x_tick_range: tuple[float, float] | None,
    y_tick_range: tuple[float, float] | None,
    candidate_status: str,
) -> VerifiedPairing:
    """Return a copy of ``pairing`` with printed-tick axis ranges attached
    (design §7.57).

    ``candidate_status`` is the raw ``status`` string carried by the
    matching entry in ``axis_pixel_candidates.json``
    ("owner_reviewed" / "llm_candidate" / "excluded"). Promotion is refused
    -- ``ValueError`` -- unless it is exactly "owner_reviewed": promoting an
    unreviewed LLM axis reading into the verified registry, and then
    scoring the v1 axis-reading task against it, is exactly the failure
    design §7.48's llm_flagged/human_confirmed discipline (see
    GtSuspectStatus) exists to prevent. ``registry.json`` is verified data;
    axis_pixel_candidates.json stays the raw LLM output and audit trail.

    Pure: does not read axis_pixel_candidates.json or any other file --
    the caller (an adapter/script) is responsible for looking up the
    matching candidate entry and passing its already-parsed fields in.
    """
    if candidate_status != "owner_reviewed":
        raise ValueError(
            "refusing to promote a tick range whose source axis reading is "
            f"not owner_reviewed (got candidate_status={candidate_status!r}) "
            "-- only human-reviewed axis readings may enter the verified "
            "registry (design §7.57, mirrors §7.48's llm_flagged discipline)"
        )
    return dataclasses.replace(
        pairing,
        x_tick_range=x_tick_range,
        y_tick_range=y_tick_range,
        tick_range_source=TickRangeProvenance.OWNER_REVIEWED,
    )
