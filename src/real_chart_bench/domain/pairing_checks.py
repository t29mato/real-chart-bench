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

**A structural limitation, found and then closed**: because two points
always determine an affine map exactly, `_affine_fit`'s own two input
points can never disagree with themselves -- there is no residual to
test. GT-extent containment (`registry_contains_gt`, rule (c)) is *not*
an independent third constraint either: for *any* monotonic affine map,
if `GT ⊆ [reg_lo, reg_hi]` then the map necessarily sends `GT` inside
`[label_min, label_max]` too, so once rule (c) has passed, checking it
again in "label space" adds no information. An initial version of this
module used rule (c) alone as the different-unit-space branch's gate,
which meant it could not, from the registry endpoints and GT extents
alone, distinguish a genuine different-but-self-consistent unit space
from a badly wrong (but still GT-bounding) registry endpoint that isn't a
unit issue at all -- unless that same wrong endpoint also happened to
fail rule (c) (as it did, by chance, for paper 18759/figure 12217 below).

**The actual third constraint**: the ground-truth curve's own physical
unit (`ground_truth.json`'s `unit_x`/`unit_y`) and the printed axis's own
unit (`axis_pixel_candidates.json`'s `x_axis_unit`/`y_axis_unit`) imply an
*expected* registry->label conversion derived with **no reference
whatsoever to the endpoint values** -- via
`domain/unit_conversion.py::si_to_display_factor` (built for exactly this
purpose, design 7.46). That is genuinely independent information, and
`rule (e)` (`unit_dimensional_analysis`, see `classify_range_disagreement`)
uses it: when unit strings are supplied, they -- not GT containment --
decide UNIT_SPACE_DIFFERENCE vs. REAL_MISMATCH in the different-unit-space
branch. Without them, that branch now returns INDETERMINATE rather than
confidently guessing from an uninformative constraint (see
`test_pairing_checks.py`'s
`TestClassifyRangeDisagreementUnitSpaceDifference` for the paired
without-units-INDETERMINATE / with-units-resolves tests).

One more real pattern the unit-dimensional check surfaces on its own: a
chart axis is sometimes labeled with its *own* scale-factor annotation on
top of the physical unit -- e.g. an Arrhenius plot's x-axis printed as
"1000/T (1/K)" (paper 46278/figure 51437) rather than plain "1/T (1/K)",
or a y-axis titled "sigma (x10^4 S/cm)". There, `unit_x`/`x_axis_unit`
are `K^(-1)`/`1/K` -- *dimensionally identical*, dimensional analysis
predicts a factor of 1 -- yet the endpoints cleanly imply `a=1000`. That
is not a unit mismatch (nothing was misconverted) and not the SI-vs-
display backlog (nothing needs migrating) -- it is the axis-label's own
declared multiplier, real and common in this corpus. `Verdict.AXIS_SCALE_FACTOR`
is a dedicated fifth outcome for exactly this signature: dimensionally
compatible units, but the fitted slope disagrees with the dimensionally-
expected one by a *clean* power of ten (`_clean_power_of_ten`, tight
tolerance -- a deliberate labeling choice is numerically exact, not
framed with margin the way an axis range is).

A second gap found while wiring this in: `si_to_display_factor` does not
model additive unit offsets at all (see its own `_normalize`, which
folds `°C` to `K` for *scale* purposes only and explicitly documents that
the absolute-value offset is a separate concern it doesn't handle) --
so Kelvin<->Celsius's `-273.15` is never something dimensional analysis
alone can report. Rather than bolt general offset-aware unit modeling
into `unit_conversion.py` (a real design task of its own, out of scope
here), this module hardcodes the one offset pair this corpus actually
uses (`_expected_additive_offset`) and states plainly that it does not
generalize -- a different additive unit pair (there are none observed in
this corpus yet) would need real design work, not a second hardcoded
case bolted on ad hoc.

Concretely, from the same review: paper 18759/figure 12217's y-axis
(registry `[25000, 135000]`, labels `(250, 1250)`) *is* correctly caught
by GT containment alone (rule (c), independent of the unit question) --
one of its four GT curves has y-extents around 662-1029, which falls
*outside* `[25000, 135000]` entirely.

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
from real_chart_bench.domain.unit_conversion import (
    IncompatibleUnitsError,
    UnitParseError,
    si_to_display_factor,
)

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

# Tolerance for recognizing a fitted/expected ratio as a "clean" power of
# ten (see Verdict.AXIS_SCALE_FACTOR). Tight, unlike _MARGIN_FRACTION: a
# deliberate axis-label scale-factor annotation ("1000/T", "x10^4") is a
# specific numeric choice, not something framed with headroom the way an
# axis range is -- so it should reproduce almost exactly, and a "clean
# power of ten that's actually off by 8%" would be a suspicious coincidence
# worth flagging as a mismatch instead, not waving through with a loose
# tolerance the way rule (b)'s margin does.
_CLEAN_POWER_OF_TEN_TOL = 0.03

# Kelvin -> Celsius offset (K to degC: degC = K - 273.15). Hardcoded
# because `si_to_display_factor` does not model additive unit offsets at
# all (see module docstring) -- this is the one additive unit pair
# observed in this corpus, not a general mechanism; a different additive
# pair would need real design work in unit_conversion.py, not a second
# case bolted on here.
_KELVIN_TO_CELSIUS_OFFSET = -273.15

# Float-safety epsilon for boundary (<=) comparisons only -- not a margin.
_EPS = 1e-9


class Verdict(Enum):
    """Five-way outcome of `classify_range_disagreement`. Never a bare
    bool: INDETERMINATE means the available data genuinely can't decide,
    and must never be conflated with BENIGN_MARGIN. UNIT_SPACE_DIFFERENCE
    is distinct from both -- it is benign *for scoring* (the registry and
    GT are self-consistent with each other) but is NOT the same claim as
    BENIGN_MARGIN (the registry matches the printed axis); it flags the
    "still needs display-unit conversion" backlog (design 7.47) and must
    be reported separately, not folded into either other outcome.
    AXIS_SCALE_FACTOR is separate again: dimensionally identical units,
    but the axis itself is printed with its own scale-factor annotation
    (e.g. "1000/T", "sigma x10^4") -- nothing was mis-converted and
    nothing needs migrating, the axis title just declares a multiplier.
    """

    BENIGN_MARGIN = "benign_margin"
    UNIT_SPACE_DIFFERENCE = "unit_space_difference"
    AXIS_SCALE_FACTOR = "axis_scale_factor"
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


def _is_missing_or_ambiguous_unit(unit: str | None) -> bool:
    """True for None or "-" (this corpus's placeholder for "dimensionless /
    no printed unit captured", e.g. ZT). Per the 2026-09 correction: "do
    not guess" -- both cases must fall out to INDETERMINATE, not be
    treated as an implicit match."""
    return unit is None or unit.strip() == "-"


def _is_kelvin(unit: str) -> bool:
    return unit.strip() == "K"


def _is_celsius(unit: str) -> bool:
    return unit.strip() in ("°C", "degC")


def _expected_additive_offset(gt_unit: str, printed_unit: str) -> float:
    """The known-gap workaround described in the module docstring:
    `si_to_display_factor` never reports an additive offset, so the one
    additive unit pair this corpus uses (Kelvin GT vs. Celsius-printed
    axis, or vice versa) is recognized here by name. Returns 0.0 for every
    other unit pair -- including any *other* additive relationship, which
    this function does not attempt to generalize to (see module
    docstring)."""
    if _is_kelvin(gt_unit) and _is_celsius(printed_unit):
        return _KELVIN_TO_CELSIUS_OFFSET
    if _is_celsius(gt_unit) and _is_kelvin(printed_unit):
        return -_KELVIN_TO_CELSIUS_OFFSET
    return 0.0


def _clean_power_of_ten(ratio: float, tol: float) -> int | None:
    """None if `ratio` isn't within `tol` (relative) of `10**n` for some
    nonzero integer `n`; otherwise `n`. `n == 0` (ratio ~= 1, i.e. "no
    disagreement at all") is deliberately excluded -- that's the "matches"
    case, handled separately, not a scale-factor signature."""
    if ratio <= 0:
        return None
    n = round(math.log10(ratio))
    if n == 0:
        return None
    if abs(ratio / (10.0**n) - 1.0) <= tol:
        return n
    return None


def _unit_dimensional_check(
    gt_unit: str | None,
    printed_unit: str | None,
    a: float,
    reg_lo: float,
    reg_hi: float,
    label_min: float,
    label_max: float,
    L: float,
    *,
    margin_fraction: float,
    scale_factor_tol: float,
) -> tuple[CheckResult, Verdict | None]:
    """Rule (e): the genuinely independent third constraint (2026-09
    correction to this module -- see module docstring on why GT-extent
    containment alone is NOT independent). `gt_unit`/`printed_unit` imply
    an expected registry->label conversion with no reference to the
    endpoint values at all.

    Returns `(CheckResult, verdict_override)`. `verdict_override` is
    `None` when the generic rule-(c)-based UNIT_SPACE_DIFFERENCE logic in
    `classify_range_disagreement` should decide (unit strings absent/
    unparseable, or they confirm a plain match); otherwise it is the
    specific verdict this check determines directly: REAL_MISMATCH for
    incompatible dimensions or a confirmed disagreement, AXIS_SCALE_FACTOR
    for a clean power-of-ten disagreement between dimensionally-identical
    units.
    """
    name = "unit_dimensional_analysis"
    if _is_missing_or_ambiguous_unit(gt_unit) or _is_missing_or_ambiguous_unit(printed_unit):
        return (
            CheckResult(
                name,
                None,
                f"no usable unit strings supplied (gt_unit={gt_unit!r}, "
                f"printed_unit={printed_unit!r}) -- cannot independently verify the unit "
                "relationship",
            ),
            None,
        )

    try:
        expected_k = si_to_display_factor(gt_unit, printed_unit)
    except IncompatibleUnitsError as exc:
        return (
            CheckResult(
                name,
                False,
                f"GT unit {gt_unit!r} and printed unit {printed_unit!r} are dimensionally "
                f"incompatible ({exc}) -- likely a wrong figure pairing, not just a bad "
                "endpoint",
            ),
            Verdict.REAL_MISMATCH,
        )
    except UnitParseError as exc:
        return (
            CheckResult(
                name,
                None,
                f"could not parse unit string(s) ({exc}) -- cannot independently verify the "
                "unit relationship",
            ),
            None,
        )

    expected_offset = _expected_additive_offset(gt_unit, printed_unit)
    predicted_lo = expected_k * reg_lo + expected_offset
    predicted_hi = expected_k * reg_hi + expected_offset
    gap_lo = abs(predicted_lo - label_min)
    gap_hi = abs(predicted_hi - label_max)
    tol = margin_fraction * L
    detail_base = (
        f"expected factor={expected_k:.6g}"
        + (f", expected offset={expected_offset:.6g}" if expected_offset else "")
        + f" (from unit strings {gt_unit!r} -> {printed_unit!r}); fitted a={a:.6g}; "
        f"expected conversion predicts label=[{predicted_lo:.6g}, {predicted_hi:.6g}] vs "
        f"actual [{label_min:.6g}, {label_max:.6g}] (gap_lo={gap_lo:.6g}, gap_hi={gap_hi:.6g}, "
        f"tol={tol:.6g})"
    )
    if gap_lo <= tol + _EPS and gap_hi <= tol + _EPS:
        # Confirms a plain multiplicative/additive unit conversion -- let
        # the generic rule-(c)-based logic finalize UNIT_SPACE_DIFFERENCE
        # (still subject to GT containment, already checked above).
        return CheckResult(name, True, detail_base + " -- matches"), None

    # Not a margin-consistent match to the expected conversion. Check the
    # "printed axis carries its own scale-factor annotation" signature
    # (design 7.42-adjacent, e.g. "1000/T", "sigma x10^4"): identical/
    # compatible dimensions (expected_offset == 0 rules out the Kelvin/
    # Celsius case, where a bare power-of-ten ratio wouldn't mean this)
    # but the FITTED slope is a clean power of ten away from the
    # dimensionally-expected one. No margin tolerance here -- see
    # _CLEAN_POWER_OF_TEN_TOL's docstring.
    if expected_offset == 0.0 and expected_k != 0:
        n = _clean_power_of_ten(a / expected_k, scale_factor_tol)
        if n is not None:
            return (
                CheckResult(
                    name,
                    True,
                    detail_base + f" -- fitted/expected ratio is a clean 10^{n:+d}, "
                    "consistent with a printed axis-label scale factor (e.g. '1000/T', "
                    "'sigma x10^4'), not a unit or endpoint error",
                ),
                Verdict.AXIS_SCALE_FACTOR,
            )

    return (
        CheckResult(
            name,
            False,
            detail_base + " -- the fitted endpoints do not encode the unit relationship the "
            "unit strings themselves declare",
        ),
        Verdict.REAL_MISMATCH,
    )


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
    gt_unit: str | None = None,
    printed_unit: str | None = None,
    margin_fraction: float = _MARGIN_FRACTION,
    pixel_tolerance_px: float = _PIXEL_TOLERANCE_PX,
    scale_factor_tol: float = _CLEAN_POWER_OF_TEN_TOL,
) -> RangeDisagreementVerdict:
    """Classifies one axis's registry-vs-printed-label range disagreement.

    `gt_unit`/`printed_unit` (e.g. `ground_truth.json`'s `unit_x`/`unit_y`
    and `axis_pixel_candidates.json`'s `x_axis_unit`/`y_axis_unit`) are
    optional but strongly recommended -- see rule (e) below and the module
    docstring on why they, not GT-extent containment, are the actual
    independent third constraint for the different-unit-space branch.

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
       entries, or Kelvin vs Celsius): rule (c) still applies first and is
       still universal -- GT extents genuinely outside the registry range
       -> REAL_MISMATCH regardless of anything else (paper 18759/figure
       12217). Otherwise, fit the exact 2-point affine map `label =
       a*registry + b` (`_affine_fit`, reported informationally) and run
       rule (e) (`unit_dimensional_analysis`, see its own docstring):

       - `gt_unit`/`printed_unit` missing, `"-"`, or unparseable ->
         rule (e) is INDETERMINATE (honest "don't know" -- do not guess).
       - dimensionally incompatible -> REAL_MISMATCH (possibly a wrong
         figure pairing entirely).
       - compatible, and the expected conversion applied to the raw
         registry endpoints reproduces the actual labels within
         `margin_fraction * L` -> confirms UNIT_SPACE_DIFFERENCE (subject
         to rule (c), already known to have passed by this point).
       - compatible, expected match fails, but the fitted slope `a` is a
         *clean power of ten* away from the dimensionally-expected factor
         -> AXIS_SCALE_FACTOR (the axis prints its own scale-factor
         annotation, e.g. "1000/T" -- paper 46278/figure 51437).
       - compatible but neither of the above -> REAL_MISMATCH (the
         endpoints don't encode the relationship the units declare).

       If rule (e) came back INDETERMINATE (no usable unit strings) and
       rule (c) also could not run (no GT extents) or ran with GT
       contained, the branch's overall verdict is INDETERMINATE -- it no
       longer defaults to UNIT_SPACE_DIFFERENCE on GT containment alone
       (see module docstring: that was shown not to be independent
       information).

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

    # Margin failed -- possible unit-space difference, axis-label scale
    # factor, or real mismatch. Fit the affine map (informational, see
    # _affine_fit's docstring) and hand off to rule (e), the genuinely
    # independent third constraint (see module docstring) -- rule (c),
    # already computed above, is known not to be False here but is no
    # longer sufficient on its own to confirm UNIT_SPACE_DIFFERENCE.
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

    unit_check, verdict_override = _unit_dimensional_check(
        gt_unit,
        printed_unit,
        a,
        reg_lo,
        reg_hi,
        label_min,
        label_max,
        L,
        margin_fraction=margin_fraction,
        scale_factor_tol=scale_factor_tol,
    )
    checks = (margin_check, containment_check, affine_check, unit_check, pixel_check)

    if verdict_override is Verdict.REAL_MISMATCH:
        return RangeDisagreementVerdict(
            Verdict.REAL_MISMATCH, checks, f"REAL_MISMATCH: {unit_check.name} ({unit_check.detail})"
        )
    if verdict_override is Verdict.AXIS_SCALE_FACTOR:
        return RangeDisagreementVerdict(
            Verdict.AXIS_SCALE_FACTOR,
            checks,
            f"AXIS_SCALE_FACTOR: {unit_check.name} ({unit_check.detail})",
        )

    # No unit-based verdict was determined (unit_check.passed is None --
    # missing/unparseable unit strings -- or True, a confirmed match).
    # Either way, rule (c) must ALSO have actually run (not just "not
    # False") for a confident UNIT_SPACE_DIFFERENCE -- both are required,
    # same pattern as the same-unit branch above.
    if containment_check.passed is None or unit_check.passed is None:
        unresolved = [c for c in (containment_check, unit_check) if c.passed is None]
        reason = "INDETERMINATE: " + "; ".join(f"{c.name} ({c.detail})" for c in unresolved)
        return RangeDisagreementVerdict(Verdict.INDETERMINATE, checks, reason)

    return RangeDisagreementVerdict(
        Verdict.UNIT_SPACE_DIFFERENCE,
        checks,
        f"UNIT_SPACE_DIFFERENCE: {affine_check.detail}; {unit_check.detail}",
    )
