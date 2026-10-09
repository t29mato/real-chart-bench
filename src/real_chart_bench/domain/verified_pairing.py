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
from dataclasses import dataclass, field
from enum import Enum

from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.dataset_subset import DatasetSubset, distribution_dir_name
from real_chart_bench.domain.licensing import licence_admitted_for


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


class FigureKind(Enum):
    """How a figure's data is drawn on the page (design §7.59).

    A binary, deliberately: it answers only "are there discrete markers",
    never "how were the points connected".

    - MARKERS: discrete markers are drawn; the markers are the data,
      whether or not any stroke also connects or fits them. A bare
      connecting polyline, a smooth fitted curve through the markers, or no
      stroke at all are all MARKERS as long as markers are present -- the
      stroke's nature is not part of this taxonomy.
    - LINE_ONLY: no markers anywhere; the stroke itself is the data (e.g. a
      continuous trace with nothing marking individual observations).

    History: three earlier rounds tried to classify figures as
    line/scatter/mixed, hinging on whether a drawn stroke *connects* the
    data points (a plain polyline) or is a *fitted curve* over them (design
    §7.59 has the failure of each round). The owner cut the distinction
    entirely -- "the plot points are the experimental values ... if there
    are plot markers, it should be recognised as markers" -- because
    connector-vs-fit turned out too hard to call reliably from the image
    alone, and a controlled comparison found it made no material
    difference to scores either way (§7.59). "scatter" is also avoided as
    a name: in materials science it implies many different samples, while
    a single sample's measurement sweep rendered as markers is not a
    statistical scatter.

    Optional on VerifiedPairing: older entries predate this taxonomy and
    carry no figure_kind at all -- None, not a member, is how "not yet
    classified" is represented.
    """

    MARKERS = "markers"
    LINE_ONLY = "line_only"


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


_SHA256_HEX_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")
_QUARTER_TURNS = frozenset({0, 90, 180, 270})


@dataclass(frozen=True)
class CropRecipe:
    """How the committed crop under data/verified_pairs/crops/ is cut out of
    its source image (figure-fetch-distribution §3.2, scaling-verification
    第0段).

    Until 第0段 this was recorded nowhere: §7.21 called the crops
    "再現不可能な手動生成物" (irreproducible manual artefacts). That is not a
    documentation gap but a correctness one -- tick_calibration.json's
    pixel coordinates are defined against the crop's exact bytes, so main
    condition 2 (人が軸を校正) is only reproducible if the crop is.

    - source_image_path: the image the rectangle is taken from -- the
      figure as extracted from the paper's PDF. Recorded explicitly
      because it is NOT derivable from the pairing's own image_path, which
      names the crop; several crops also come from a source that no entry
      points at directly (a two-panel image split into two crops).
    - box: ``(x0, y0, x1, y1)`` in the source's pixel coordinates,
      half-open on the far edge -- the numpy / PIL ``Image.crop``
      convention, so the recipe is executable without a translation step.
    - rotation_deg: counter-clockwise quarter turn applied *after* cutting
      (numpy.rot90 convention). 0 for every entry recovered so far.
      Mirrors are deliberately NOT expressible: see
      domain/crop_recovery.py's CropOrientation for the orientation fixes
      of paper 2.4 that fall outside this vocabulary and are reported
      rather than recorded.
    """

    source_image_path: str
    box: tuple[int, int, int, int]
    rotation_deg: int = 0

    def __post_init__(self) -> None:
        if not self.source_image_path:
            raise ValueError("source_image_path must name the image the crop is cut from")
        if len(self.box) != 4 or any(type(value) is not int for value in self.box):
            raise ValueError(f"box must be four ints (x0, y0, x1, y1), got {self.box!r}")
        x0, y0, x1, y1 = self.box
        if x0 < 0 or y0 < 0:
            raise ValueError(f"box origin must be non-negative, got {self.box!r}")
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"box must be non-empty and half-open, got {self.box!r}")
        if self.rotation_deg not in _QUARTER_TURNS:
            raise ValueError(
                "rotation_deg must be one of 0/90/180/270 (a quarter turn is the "
                f"only orientation change a crop recipe can express), got {self.rotation_deg!r}"
            )

    @property
    def width(self) -> int:
        """Width of the rectangle in the source image."""
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        """Height of the rectangle in the source image."""
        return self.box[3] - self.box[1]

    @property
    def final_width(self) -> int:
        """Width of the saved crop, i.e. after ``rotation_deg`` is applied."""
        return self.height if self.rotation_deg in (90, 270) else self.width

    @property
    def final_height(self) -> int:
        return self.width if self.rotation_deg in (90, 270) else self.height


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
    # design §7.84: the second y axis, printed on the right, for a figure that
    # draws some of its series against it. The x axis is shared, so only the y
    # extent and scale are recorded. None on every entry today -- the dataset
    # has no dual-y figure yet. y2_scale is None exactly when y2_range is.
    y2_range: tuple[float, float] | None = None
    y2_scale: ScaleType | None = None
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
    # design §7.59: how this figure's data is drawn (markers vs a bare
    # stroke). None for entries verified before this taxonomy existed --
    # "not yet classified", not a third value of the enum itself.
    figure_kind: FigureKind | None = None
    # design §7.59: observation-quality tags carried over from the
    # markers/line_only classification pass, e.g. "inset", "error_bars",
    # "dense_overlap" -- see the migration script / docs/design §7.59 for
    # the full set. Deliberately excludes "fitting_line": that tag encoded
    # exactly the connector-vs-fit judgement the owner abandoned as
    # unreliable, and carrying it forward would invite rebuilding that
    # distinction on data nobody trusts. Empty tuple (not None) when an
    # entry has been through the pass but has no tags -- mirrors
    # GroundTruthCurve.quality_flags's tuple[str, ...] convention.
    figure_tags: tuple[str, ...] = field(default_factory=tuple)
    # scaling-verification 第0段 / figure-fetch-distribution §3.2: the
    # deterministic recipe for a hand-made crop, and the sha256 of the crop
    # file the recipe must reproduce. Both-or-neither, the same discipline as
    # the y2 pair above (§7.84): a box with no hash is a claim nobody can
    # check, and a hash with no box says the bytes matter but not how to get
    # them. None on the 16 entries that point straight at an extracted image
    # (no crop to reproduce) and on every crop whose source image is not in
    # the repository, so the box could not be recovered and verified.
    crop: CropRecipe | None = None
    final_sha256: str | None = None
    # design §7.88.1: which distribution this figure belongs to. Follows the
    # licence (never set by hand): enforced in __post_init__ whenever
    # license_id names a redistributable licence. core for every entry
    # written before the nc subset existed.
    subset: DatasetSubset = DatasetSubset.CORE

    def __post_init__(self) -> None:
        self._check_subset()
        self._check_invariants()

    def _check_subset(self) -> None:
        # raises when the licence admits the figure to the other subset; the
        # subset follows the licence and is never set by hand (design §7.88)
        licence_admitted_for(self.license_id, self.subset)
        # a committed figure (a path with a directory) must sit in its own
        # subset's distribution directory -- the nc figures ship separately
        if self.image_path and "/" in self.image_path:
            own = f"data/{distribution_dir_name(self.subset)}/"
            if not self.image_path.startswith(own):
                raise ValueError(
                    f"{self.subset.value} figure {self.image_path!r} must be committed "
                    f"under {own}"
                )

    def _check_invariants(self) -> None:
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

        # design §7.84: the second y axis is a second axis of the same figure,
        # so it needs the first one to exist; and its range and scale come as
        # a pair -- a scale alone describes an axis that is not there, and a
        # range alone would leave the scale to be guessed.
        if self.y2_range is not None and self.y_range is None:
            raise ValueError("y2_range requires y_range (the first y axis) to be set")
        if self.y2_scale is not None and self.y2_range is None:
            raise ValueError("y2_scale requires y2_range (the second axis's extent)")
        if self.y2_range is not None and self.y2_scale is None:
            raise ValueError("y2_range requires y2_scale (linear or log, never guessed)")

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

        # scaling-verification 第0段: the crop recipe and the hash of the file
        # it must reproduce come as a pair, and the recipe needs an image to
        # be a recipe *for*.
        if self.crop is not None and self.final_sha256 is None:
            raise ValueError(
                "crop requires final_sha256 (the hash of the crop file the "
                "recipe must reproduce byte for byte) -- a box nobody can "
                "check against the committed bytes is not a recovered box"
            )
        if self.final_sha256 is not None and self.crop is None:
            raise ValueError(
                "final_sha256 requires crop (the recipe that reproduces those bytes)"
            )
        if self.crop is not None:
            if self.image_path is None:
                raise ValueError("crop requires image_path (the crop file it describes)")
            if self.crop.source_image_path == self.image_path:
                raise ValueError(
                    "crop.source_image_path must differ from image_path -- a crop "
                    "cut out of itself is not a recipe"
                )
        if self.final_sha256 is not None and (
            len(self.final_sha256) != _SHA256_HEX_LENGTH
            or not set(self.final_sha256) <= _HEX_DIGITS
        ):
            raise ValueError(
                "final_sha256 must be a lowercase hex sha256 digest "
                f"({_SHA256_HEX_LENGTH} chars), got {self.final_sha256!r}"
            )

    @property
    def is_licence_admitted(self) -> bool:
        """True iff license_id admits this figure to its subset (design
        §7.88.1). False for ND, unknown and missing licences, whatever subset
        the entry says: such an entry stays loadable but is neither scored
        nor distributed."""
        return licence_admitted_for(self.license_id, self.subset)

    @property
    def is_reproducible_crop(self) -> bool:
        """True iff this entry's scored image is a crop whose box has been
        recovered and whose bytes a re-cut of the source reproduces
        (scaling-verification 第0段's acceptance bar)."""
        return self.crop is not None and self.final_sha256 is not None

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
