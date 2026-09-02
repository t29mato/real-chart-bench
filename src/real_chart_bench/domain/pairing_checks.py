"""Benign/real classifier for registry-vs-printed-label axis range
disagreements, plus the general-purpose range checks it's built from.

## Background

`scripts/eval/generate_verified_pairs_visual_audit.py`'s `_derive_factor`
used to *fit* a unit-conversion factor between a verified_pairs entry's
`registry.json` `x_range`/`y_range` and the axis's printed tick-label
values, independently at the min and max end of the axis (`k_min`,
`k_max`), and flagged "needs attention" whenever the two disagreed -- 45 of
111 verified entries. The audit doc itself admits a flag there is "not
necessarily an error": the registry range legitimately, and often,
extends a bit past the outermost printed tick (an axis framed to leave
headroom for a data point beyond the last label).

Since commit 31bd7e9, `registry.json` and `ground_truth.json` are stored
in each paper's *display* units -- the same units the axis is printed in
-- with a documented exception: a minority of entries are still recorded
in raw SI, not yet migrated (README / design 7.47's "not yet converted to
display units" backlog).

## Why this is an AFFINE relationship, not a multiplicative ratio

An earlier version of this module tested "does a single multiplicative
factor `k = label / registry` agree at both the low and the high
endpoint" and treated agreement on a `k != 1` as a confident real bug. A
2026-09-02 review of all 208 axes across the 111 verified entries showed
that test is wrong on both ends:

- It fires on the *backlog* case (e.g. paper 46278's x-axis, exactly
  `1000.0` at both endpoints -- the axis prints "1000/T (1/K)" while the
  registry is plain `1/K`) -- registry, GT and printed axis are all
  self-consistent, just not yet migrated to display units. That is not an
  error.
- It is blind to *additive* unit differences entirely (Kelvin vs Celsius,
  e.g. paper 4965/figure 13164's x-axis: registry `[298.15, 353.15]` vs
  printed labels `(30, 80)` -- a pure offset, no ratio test can ever see
  it) and it silently mis-skips whenever an endpoint is exactly zero
  (paper 44283/figure 38965's y-axis: registry `[0.0, 2.5e-5]` vs labels
  `(0.5, 2.5)` -- the low end's ratio is undefined).

The general model that covers both the multiplicative and the additive
case, and degrades gracefully at a zero endpoint (subtraction, not
division), is an **affine** map `label = a * registry + b`. Two endpoint
pairs `(reg_lo, label_min)` and `(reg_hi, label_max)` determine `(a, b)`
*exactly* -- see `_affine_fit`.

**A structural limitation, stated plainly**: because two points always
determine an affine map exactly, `_affine_fit`'s own two input points can
never disagree with themselves -- there is no residual to test. The only
independent, already-available third constraint this module has is
whether the ground-truth curve's own extents still fall inside the
registry range (`registry_contains_gt`, rule (c) below) -- and for *any*
monotonic affine map, if `GT ⊆ [reg_lo, reg_hi]` then the map necessarily
sends `GT` inside `[label_min, label_max]` too. That containment is
mathematically guaranteed once rule (c) already passed, not new
information -- so this module reports the fitted `(a, b)` for a human's
own judgement of whether it looks like a "clean" ratio/offset, but it
cannot, from the registry endpoints and GT extents alone, distinguish a
genuine different-but-self-consistent unit space from a badly wrong (but
still GT-bounding) registry endpoint that isn't a unit issue at all --
unless that same wrong endpoint also happens to fail rule (c) (as it did,
by chance, for the worked real case below). Catching the general case
would need a third *independent* tick reading or a "does this factor look
like a physically plausible unit conversion" check -- out of scope for a
pure function operating on two endpoints and a curve's extents. See
`classify_range_disagreement`'s docstring and
`test_pairing_checks.py::test_large_margin_violation_with_contained_gt_is_still_unit_space_difference`
for this documented as a known gap, not papered over.

Concretely, from the same 208-axis review: paper 18759/figure 12217's
y-axis (registry `[25000, 135000]`, labels `(250, 1250)`) *is* correctly
caught -- not via the affine fit, but because one of its four GT curves
has y-extents around 662-1029, which falls *outside* `[25000, 135000]`
entirely -- rule (c) fails outright, independent of any unit-space
question.

## Scope

Pure domain: no I/O, no file paths, no plotting/imaging libraries, no
network. Callers (a future automated pairing pipeline, or the visual
audit script) are responsible for reading `registry.json`,
`axis_pixel_candidates.json`, `ground_truth.json` and image dimensions,
and passing plain numbers in here.

Priority is auditability over accuracy (CLAUDE.md): every public function
returns a `CheckResult`/`RangeDisagreementVerdict` carrying which sub-rule
fired and the numbers that decided it, not a bare bool -- and a check that
lacks the data to decide returns an explicit INDETERMINATE (`passed=None`)
rather than silently passing.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from real_chart_bench.domain.curve import ScaleType

# 0.25 * L is a deliberately round, retightenable margin tolerance for "how
# far past the outermost printed label can a registry endpoint legitimately
# sit" (normal axis-framing headroom). A 2026-09-02 measurement across the
# 404 endpoints on axes already confirmed to be in the SAME unit space (the
# BENIGN_MARGIN population `classify_range_disagreement` actually applies
# this to) found: p50 = 0.000, p75 = 0.000, p90 = 0.125, p95 = 0.446 -- i.e.
# most axes have NO margin at all, and 24/404 (5.9%) exceed 0.25L. That
# tail is still contaminated by unit-space-difference axes the (now
# superseded) ratio-only test failed to route away from the margin
# population -- so the true benign-margin distribution is likely tighter
# than 0.25L suggests. Kept at 0.25 (comfortably above the p90) rather than
# tightened to ~0.15L, pending a re-measurement once callers route
# unit-space-difference axes through `classify_range_disagreement` instead
# of into this population; retighten then.
_MARGIN_FRACTION = 0.25

# ~2x the observed inter-model pixel disagreement (see
# axis_pixel_candidates.json's `model_disagreement_px`, typically well under
# 1px, occasionally a few px on a hard read) plus line width, at 150dpi.
_PIXEL_TOLERANCE_PX = 3.0

# GT_span / L expected band. A 2026-09-02 measurement across 202
# same-unit-space axes found p5 = 0.376, p50 = 0.900, p95 = 1.093, with
# 13/202 (6.4%) outside [0.35, 1.15] -- this band already matches that
# distribution well (comfortably straddling p5..p95) and needs no change.
_MIN_COVERAGE_RATIO = 0.35
_MAX_COVERAGE_RATIO = 1.15

_LOG_MIN_DECADES = 1.0
_LINEAR_MAX_DECADES = 2.5

# Float-safety epsilon for boundary (<=) comparisons only -- not a margin.
_EPS = 1e-9


class Verdict(Enum):
    """Four-way outcome of `classify_range_disagreement`. Never a bare
    bool: INDETERMINATE means the available data genuinely can't decide,
    and must never be conflated with BENIGN_MARGIN. UNIT_SPACE_DIFFERENCE
    is distinct from both -- it is benign *for scoring* (the registry and
    GT are self-consistent with each other) but is NOT the same claim as
    BENIGN_MARGIN (the registry matches the printed axis); it flags the
    "still needs display-unit conversion" backlog (design 7.47) and must
    be reported separately, not folded into either other outcome.
    """

    BENIGN_MARGIN = "benign_margin"
    UNIT_SPACE_DIFFERENCE = "unit_space_difference"
    REAL_MISMATCH = "real_mismatch"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class CheckResult:
    """One named sub-rule's outcome plus the numbers that decided it.

    `passed`: True supports the non-error outcome, False supports
    REAL_MISMATCH, None means this sub-rule could not be evaluated
    (missing/degenerate input) -- an honest "don't know", never silently
    treated as a pass. Some checks (`affine_fit`) are purely informational
    and never report False -- see their own docstrings.
    """

    name: str
    passed: bool | None
    detail: str


@dataclass(frozen=True)
class RangeDisagreementVerdict:
    """The full audit trail for one axis's registry-vs-label disagreement
    classification: the overall verdict, every sub-check that ran, and a
    one-line reason summarizing which check(s) decided it."""

    verdict: Verdict
    checks: tuple[CheckResult, ...]
    reason: str


@dataclass(frozen=True)
class AxisPixelCalibration:
    """Printed-label pixel positions for one axis, plus the image's pixel
    extent along that axis -- enough to project any data-space value (e.g.
    a registry endpoint) to a pixel coordinate and check whether it lands
    inside the actual image (rule (d) of `classify_range_disagreement`).

    `label_lo_px`/`label_hi_px` are the pixel coordinates of the axis's
    `label_min`/`label_max` values (as passed to
    `classify_range_disagreement`) -- e.g.
    `axis_pixel_candidates.json`'s `pixel_bbox_mean.y_min_px`/`y_max_px`.
    `image_extent_px` is the image's width (x-axis) or height (y-axis) in
    pixels; valid pixel coordinates run `[0, image_extent_px]` (pixel-down
    vs pixel-up orientation doesn't matter here since both bounds are
    checked symmetrically).
    """

    label_lo_px: float
    label_hi_px: float
    image_extent_px: float


def _transform(value: float, scale: ScaleType) -> float | None:
    """Maps a raw axis value into "comparison space": identity for a
    linear axis, log10 for a log axis (undefined, returns None, for a
    non-positive value on a log axis)."""
    if scale is ScaleType.LOG:
        if value <= 0:
            return None
        return math.log10(value)
    return value


def _label_span(label_min: float, label_max: float, scale: ScaleType) -> float | None:
    """L = printed label span, in comparison space (log10-space for a log
    axis). None if not computable (non-positive label on a log axis)."""
    lo = _transform(label_min, scale)
    hi = _transform(label_max, scale)
    if lo is None or hi is None:
        return None
    return abs(hi - lo)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def containment(
    gt_extents: Sequence[float],
    label_min: float,
    label_max: float,
    *,
    scale: ScaleType = ScaleType.LINEAR,
    margin_fraction: float = _MARGIN_FRACTION,
) -> CheckResult:
    """Every GT extent lies within `[min_label - margin_fraction*L,
    max_label + margin_fraction*L]` (in log10-space on a log axis).

    General-purpose reusable check for a future automated pairing
    pipeline: unlike rule (c) of `classify_range_disagreement` (which
    checks exact containment against the *registry* range, no margin),
    this checks against the *printed labels* with the same 0.25L margin
    used elsewhere in this module.
    """
    name = "containment"
    if not gt_extents:
        return CheckResult(name, None, "no GT extents supplied -- cannot evaluate containment")

    L = _label_span(label_min, label_max, scale)
    if L is None or L == 0:
        return CheckResult(
            name, None, "degenerate or non-computable label span -- cannot evaluate containment"
        )

    lo_t = _transform(label_min, scale)
    hi_t = _transform(label_max, scale)
    margin = margin_fraction * L
    lower_bound, upper_bound = lo_t - margin, hi_t + margin

    violations = []
    for v in gt_extents:
        vt = _transform(v, scale)
        if vt is None:
            return CheckResult(
                name,
                None,
                f"GT extent {v:g} is non-positive on a log-scale axis -- "
                "cannot evaluate containment",
            )
        if not (lower_bound - _EPS <= vt <= upper_bound + _EPS):
            violations.append(v)

    detail = (
        f"bounds=[{lower_bound:.6g}, {upper_bound:.6g}] "
        f"(label +/- {margin_fraction:g}L, L={L:.6g})"
    )
    if violations:
        return CheckResult(name, False, detail + f", outside bounds: {violations}")
    return CheckResult(name, True, detail)


def coverage(
    gt_extents: Sequence[float],
    label_min: float,
    label_max: float,
    *,
    scale: ScaleType = ScaleType.LINEAR,
    min_ratio: float = _MIN_COVERAGE_RATIO,
    max_ratio: float = _MAX_COVERAGE_RATIO,
) -> CheckResult:
    """`GT_span / L` within `[min_ratio, max_ratio]` (default `[0.35,
    1.15]`, see the module-level constants' docstring for the measured
    distribution behind that band), in log10-space on a log axis. GT_span
    is the spread between the smallest and largest supplied GT extent."""
    name = "coverage"
    if not gt_extents:
        return CheckResult(name, None, "no GT extents supplied -- cannot evaluate coverage")

    L = _label_span(label_min, label_max, scale)
    if L is None or L == 0:
        return CheckResult(
            name, None, "degenerate or non-computable label span -- cannot evaluate coverage"
        )

    transformed = [_transform(v, scale) for v in gt_extents]
    if any(t is None for t in transformed):
        return CheckResult(
            name,
            None,
            "GT extent is non-positive on a log-scale axis -- cannot evaluate coverage",
        )

    gt_span = max(transformed) - min(transformed)
    ratio = gt_span / L
    detail = (
        f"GT_span={gt_span:.6g}, L={L:.6g}, ratio={ratio:.4f} "
        f"(expected [{min_ratio:g}, {max_ratio:g}])"
    )
    passed = min_ratio - _EPS <= ratio <= max_ratio + _EPS
    return CheckResult(name, passed, detail)


def scale_consistency(
    gt_extents: Sequence[float],
    scale: ScaleType,
    *,
    log_min_decades: float = _LOG_MIN_DECADES,
    linear_max_decades: float = _LINEAR_MAX_DECADES,
) -> CheckResult:
    """A log axis implies the GT curve spans at least `log_min_decades`
    decades (default 1); a linear axis implies it spans fewer than
    `linear_max_decades` decades (default 2.5) -- a sanity check that the
    recorded scale type is coherent with the data actually plotted on it,
    not a range/margin check. Decades are computed from the plain min/max
    of `gt_extents` regardless of the axis's own scale."""
    name = "scale_consistency"
    if not gt_extents:
        return CheckResult(
            name, None, "no GT extents supplied -- cannot evaluate scale consistency"
        )

    lo, hi = min(gt_extents), max(gt_extents)
    if lo <= 0:
        return CheckResult(name, None, "non-positive GT extent -- cannot compute decades spanned")

    decades = math.log10(hi / lo) if hi != lo else 0.0
    if scale is ScaleType.LOG:
        detail = f"GT spans {decades:.3f} decades on a log axis (expected >= {log_min_decades:g})"
        passed = decades >= log_min_decades - _EPS
    else:
        detail = (
            f"GT spans {decades:.3f} decades on a linear axis (expected < {linear_max_decades:g})"
        )
        passed = decades < linear_max_decades - _EPS
    return CheckResult(name, passed, detail)


def _affine_fit(
    label_min: float, label_max: float, reg_lo: float, reg_hi: float
) -> tuple[float, float] | None:
    """Exact affine fit `label = a * registry + b` through the two
    endpoint pairs `(reg_lo, label_min)` and `(reg_hi, label_max)`.
    Subtraction-based, not division-based, so a zero (or negative)
    endpoint on either side never raises -- unlike a plain ratio `k =
    label/registry`, which is undefined at registry=0 and misleading near
    it (see the module docstring's `44283/38965` example).

    None only when `reg_hi == reg_lo` (a degenerate, zero-width registry
    range -- no slope is determinable).

    This fit ALWAYS reproduces both input points exactly (two points
    determine a unique line) -- it has no residual against its own inputs
    and therefore cannot, by itself, be "wrong". See the module docstring
    for what that structurally means for how far this module can push the
    unit-space-difference vs. real-mismatch distinction.
    """
    reg_span = reg_hi - reg_lo
    if reg_span == 0:
        return None
    a = (label_max - label_min) / reg_span
    b = label_min - a * reg_lo
    return a, b


def _margin_check(
    label_min: float,
    label_max: float,
    reg_lo: float,
    reg_hi: float,
    scale: ScaleType,
    *,
    margin_fraction: float,
) -> CheckResult:
    """Rule (b): each registry endpoint within `margin_fraction * L` of its
    corresponding label (log10-space on a log axis). This is the gate
    between the two branches of `classify_range_disagreement`: passing
    means "plausibly the same unit space, any residual is a framing
    margin"; failing means "investigate as a possible different (but
    perhaps still self-consistent) unit space"."""
    name = "endpoint_margin"
    L = _label_span(label_min, label_max, scale)
    if L is None or L == 0:
        return CheckResult(
            name,
            None,
            "degenerate or non-computable label span (L=0 or non-positive log labels) -- "
            "cannot evaluate margin",
        )

    lab_lo_t, lab_hi_t = _transform(label_min, scale), _transform(label_max, scale)
    reg_lo_t, reg_hi_t = _transform(reg_lo, scale), _transform(reg_hi, scale)
    if reg_lo_t is None or reg_hi_t is None:
        return CheckResult(
            name,
            None,
            "registry endpoint is non-positive on a log-scale axis -- cannot evaluate margin",
        )

    gap_lo = abs(reg_lo_t - lab_lo_t)
    gap_hi = abs(reg_hi_t - lab_hi_t)
    tol = margin_fraction * L
    detail = (
        f"gap_lo={gap_lo:.6g} ({gap_lo / L:.3f}L), gap_hi={gap_hi:.6g} ({gap_hi / L:.3f}L), "
        f"tol={tol:.6g} ({margin_fraction:g}L)"
    )
    passed = gap_lo <= tol + _EPS and gap_hi <= tol + _EPS
    return CheckResult(name, passed, detail if passed else detail + " -- exceeds margin")


def _registry_containment_check(
    reg_lo: float, reg_hi: float, gt_extents: Sequence[float]
) -> CheckResult:
    """Rule (c): the registry range contains the ground-truth curve
    extents exactly (no margin -- the registry range is defined to be the
    axis frame that contains all plotted data). This is the ONLY check in
    this module that can independently produce REAL_MISMATCH in the
    unit-space-difference branch (see module docstring) -- e.g. paper
    18759/figure 12217, where one of four GT curves has y-extents
    (~662-1029) that fall entirely outside the registry's stated
    `[25000, 135000]`."""
    name = "registry_contains_gt"
    if not gt_extents:
        return CheckResult(name, None, "no GT extents supplied -- cannot evaluate containment")

    lo, hi = min(gt_extents), max(gt_extents)
    detail = f"registry=[{reg_lo:.6g}, {reg_hi:.6g}], GT=[{lo:.6g}, {hi:.6g}]"
    if reg_lo - _EPS <= lo and hi <= reg_hi + _EPS:
        return CheckResult(name, True, detail)
    return CheckResult(name, False, detail + " -- GT extent(s) fall outside the registry range")


def _project_to_pixel(
    value: float,
    label_min: float,
    label_max: float,
    scale: ScaleType,
    calibration: AxisPixelCalibration,
) -> float | None:
    """Projects a data-space `value` to a pixel coordinate via linear
    interpolation/extrapolation between the two printed-label pixel
    positions (log10-space interpolation on a log axis). None if not
    computable (non-positive value/labels on a log axis, or a degenerate
    zero-span calibration)."""
    v_t = _transform(value, scale)
    lo_t = _transform(label_min, scale)
    hi_t = _transform(label_max, scale)
    if v_t is None or lo_t is None or hi_t is None or hi_t == lo_t:
        return None
    frac = (v_t - lo_t) / (hi_t - lo_t)
    return calibration.label_lo_px + frac * (calibration.label_hi_px - calibration.label_lo_px)


def _pixel_bounds_check(
    label_min: float,
    label_max: float,
    reg_lo: float,
    reg_hi: float,
    scale: ScaleType,
    calibration: AxisPixelCalibration | None,
    *,
    tolerance_px: float,
) -> CheckResult:
    """Rule (d): both registry endpoints, projected through the
    label->pixel calibration, land inside the image bounds (+/-
    tolerance_px). NOT optional in the same-unit-space branch -- catches
    cases (paper 10939, figure 1528) where the gap passes rule (b) but the
    endpoint still falls outside the actual image. Computed (and reported)
    in the unit-space-difference branch too, but only informationally
    there -- it speaks to whether the *printed axis calibration* is
    accurate, which is orthogonal to whether the registry and GT are
    self-consistent in a different unit space."""
    name = "endpoint_pixel_bounds"
    if calibration is None:
        return CheckResult(
            name,
            None,
            "no pixel calibration available -- cannot verify endpoints project inside the image",
        )

    lo_px = _project_to_pixel(reg_lo, label_min, label_max, scale, calibration)
    hi_px = _project_to_pixel(reg_hi, label_min, label_max, scale, calibration)
    if lo_px is None or hi_px is None:
        return CheckResult(
            name,
            None,
            "endpoint could not be projected (degenerate calibration or non-positive "
            "log-axis value)",
        )

    bound_lo = -tolerance_px
    bound_hi = calibration.image_extent_px + tolerance_px
    lo_ok = bound_lo <= lo_px <= bound_hi
    hi_ok = bound_lo <= hi_px <= bound_hi
    detail = (
        f"reg_lo={reg_lo:g} -> {lo_px:.2f}px, reg_hi={reg_hi:g} -> {hi_px:.2f}px "
        f"(image bounds [0, {calibration.image_extent_px:g}] +/-{tolerance_px:g}px)"
    )
    if lo_ok and hi_ok:
        return CheckResult(name, True, detail)
    return CheckResult(name, False, detail + " -- endpoint falls outside the image")


def classify_range_disagreement(
    *,
    label_min: float,
    label_max: float,
    reg_lo: float,
    reg_hi: float,
    scale: ScaleType = ScaleType.LINEAR,
    gt_extents: Sequence[float] = (),
    calibration: AxisPixelCalibration | None = None,
    margin_fraction: float = _MARGIN_FRACTION,
    pixel_tolerance_px: float = _PIXEL_TOLERANCE_PX,
) -> RangeDisagreementVerdict:
    """Classifies one axis's registry-vs-printed-label range disagreement.

    Decision tree:

    1. Rule (b) (`endpoint_margin`): is each registry endpoint within
       `margin_fraction * L` of its corresponding printed label (L =
       printed label span, log10-space on a log axis)? This is the gate.

    2. If margin passes ("same unit space, at most a framing margin"):
       BENIGN_MARGIN iff rule (c) (`registry_contains_gt`: the registry
       range contains the GT curve extents, no margin) AND rule (d)
       (`endpoint_pixel_bounds`: both registry endpoints, projected
       through the label->pixel calibration, land inside the image, +/-
       `pixel_tolerance_px`) both pass. Either failing -> REAL_MISMATCH
       (rule (d) is NOT optional -- see paper 10939/figure 1528, design
       review 2026-09: gap passes rule (b) at only 0.125L but the
       registry endpoint still projects above the top of the image).
       Either being unevaluable (no GT extents / no calibration) with
       neither failing -> INDETERMINATE.

    3. If margin fails ("possibly a different, but perhaps still
       self-consistent, unit space" -- e.g. still-SI-not-yet-converted
       entries, or Kelvin vs Celsius): fit the exact 2-point affine map
       `label = a*registry + b` (`_affine_fit`, reported informationally
       -- see its docstring on why it can never itself disagree with its
       inputs) and re-check rule (c). GT extents genuinely outside the
       registry range -> REAL_MISMATCH (independent of the unit-space
       question). Otherwise, if GT extents were supplied ->
       UNIT_SPACE_DIFFERENCE (self-consistent, benign for scoring, but
       flagged as its own outcome -- this is the "still needs
       display-unit conversion" backlog, design 7.47, NOT the same claim
       as BENIGN_MARGIN). No GT extents supplied -> INDETERMINATE.

       **Known limitation** (see module docstring): step 3's rule (c)
       check cannot detect a registry endpoint that's simply wrong in a
       way that still happens to bound the GT curve -- there is no third
       independent constraint available to this pure function for that
       case. It reliably catches only registry ranges that fail to
       contain their own GT data (as the worked paper 18759/figure 12217
       example does).

    Any branch with a degenerate/non-computable `L` (equal labels, or
    non-positive labels on a log axis) short-circuits to INDETERMINATE
    before rule (b) is even evaluated.
    """
    L = _label_span(label_min, label_max, scale)
    if L is None or L == 0:
        degenerate = CheckResult(
            "endpoint_margin",
            None,
            "degenerate or non-computable label span (L=0 or non-positive log labels) -- "
            "cannot evaluate margin",
        )
        return RangeDisagreementVerdict(
            Verdict.INDETERMINATE, (degenerate,), f"INDETERMINATE: {degenerate.detail}"
        )

    margin_check = _margin_check(
        label_min, label_max, reg_lo, reg_hi, scale, margin_fraction=margin_fraction
    )
    containment_check = _registry_containment_check(reg_lo, reg_hi, gt_extents)
    pixel_check = _pixel_bounds_check(
        label_min, label_max, reg_lo, reg_hi, scale, calibration, tolerance_px=pixel_tolerance_px
    )

    if margin_check.passed is None:
        checks = (margin_check, containment_check, pixel_check)
        return RangeDisagreementVerdict(
            Verdict.INDETERMINATE, checks, f"INDETERMINATE: {margin_check.detail}"
        )

    if containment_check.passed is False:
        # Universal: a registry range that doesn't even bound its own GT
        # curve is a real problem regardless of which branch we're in.
        checks = (margin_check, containment_check, pixel_check)
        return RangeDisagreementVerdict(
            Verdict.REAL_MISMATCH,
            checks,
            f"REAL_MISMATCH: {containment_check.name} ({containment_check.detail})",
        )

    if margin_check.passed:
        checks = (margin_check, containment_check, pixel_check)
        if pixel_check.passed is False:
            return RangeDisagreementVerdict(
                Verdict.REAL_MISMATCH,
                checks,
                f"REAL_MISMATCH: {pixel_check.name} ({pixel_check.detail})",
            )
        if containment_check.passed is None or pixel_check.passed is None:
            unresolved = [c for c in (containment_check, pixel_check) if c.passed is None]
            reason = "INDETERMINATE: " + "; ".join(f"{c.name} ({c.detail})" for c in unresolved)
            return RangeDisagreementVerdict(Verdict.INDETERMINATE, checks, reason)
        return RangeDisagreementVerdict(
            Verdict.BENIGN_MARGIN, checks, f"BENIGN_MARGIN: {margin_check.detail}"
        )

    # Margin failed -- possible unit-space difference. Fit the affine map
    # (informational only, see _affine_fit's docstring) and fall back on
    # rule (c), already computed above (and already known not to be False
    # here).
    affine = _affine_fit(label_min, label_max, reg_lo, reg_hi)
    if affine is None:
        affine_check = CheckResult(
            "affine_fit", None, "degenerate registry span (reg_lo == reg_hi) -- cannot fit"
        )
        checks = (margin_check, containment_check, affine_check, pixel_check)
        return RangeDisagreementVerdict(
            Verdict.INDETERMINATE, checks, f"INDETERMINATE: {affine_check.detail}"
        )

    a, b = affine
    affine_check = CheckResult(
        "affine_fit",
        True,
        f"label = {a:.6g} * registry + {b:.6g} (exact fit through both endpoints -- "
        "informational; see module docstring on this check's limits)",
    )
    checks = (margin_check, containment_check, affine_check, pixel_check)

    if containment_check.passed is None:
        return RangeDisagreementVerdict(
            Verdict.INDETERMINATE, checks, f"INDETERMINATE: {containment_check.detail}"
        )
    # containment_check.passed is True here (False already returned above).
    return RangeDisagreementVerdict(
        Verdict.UNIT_SPACE_DIFFERENCE,
        checks,
        f"UNIT_SPACE_DIFFERENCE: {affine_check.detail}; {containment_check.detail}",
    )
