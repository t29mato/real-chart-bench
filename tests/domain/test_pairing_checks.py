"""pairing_checks: the benign/real classifier for registry-vs-printed-label
axis range disagreements (design finding following commit 31bd7e9 -- since
registry.json/ground_truth.json are stored in each paper's *display* units,
the a-priori SI->display factor is known to be 1, so any residual
disagreement needs a principled classification instead of `_derive_factor`'s
fitted-factor "needs attention" flag in
`generate_verified_pairs_visual_audit.py`).

A 2026-09-02 review of all 208 axes across the 111 verified entries found
the registry-vs-label relationship is AFFINE (`label = a*registry + b`),
not multiplicative -- a pure ratio test is blind to additive unit
differences (Kelvin vs Celsius) and undefined/misleading at a zero
endpoint.

A same-day follow-up correction found that GT-extent containment (used as
the different-unit-space branch's gate in the first version of this
module) is NOT an independent third constraint -- any monotonic affine
map preserves interval containment, so it can never disagree with the
endpoints it was fit from. The genuinely independent third constraint is
the GT curve's own physical unit vs. the printed axis's own unit
(`domain/unit_conversion.py::si_to_display_factor`), which implies an
expected conversion with no reference to the endpoint values at all.
`classify_range_disagreement` therefore has FIVE outcomes: BENIGN_MARGIN
(same unit space, residual is just axis-framing headroom),
UNIT_SPACE_DIFFERENCE (a different but self-consistent unit space --
benign for scoring, a distinct "still needs display-unit conversion"
backlog, NOT the same claim as BENIGN_MARGIN, now actually *evidenced* by
the unit strings rather than assumed from containment alone),
AXIS_SCALE_FACTOR (dimensionally identical units, but the endpoints
cleanly imply a power-of-ten multiplier -- the printed axis carries its
own scale-factor annotation, e.g. "1000/T"), REAL_MISMATCH, and
INDETERMINATE.

Where possible, tests use REAL numbers from `data/verified_pairs/` (cited
by paper_id/figure_id) rather than invented ones, since this module's own
docstring documents a genuine mathematical limitation and real data is
the only way to be honest about what it does and doesn't catch. One
citation from the correction's own instructions (paper 17038/figure
20816's x-axis GT unit) did not match the committed data when checked
(its `unit_x` is `'nm'`, not `'-'`) -- the dimensionless/missing-unit
tests below use synthetic unit strings instead, on top of otherwise-real
registry/label/GT numbers, and a real `'-'`-unit entry (paper 10939,
figure 1530's y-axis, ZT) is cited separately for context even though its
margin passes (same-unit branch, so it never reaches the unit-dimensional
check at all).

Priority is auditability over accuracy (CLAUDE.md) -- every test also
checks that the returned verdict carries *which* sub-check fired, not just
a bare bool, and that missing data produces an explicit INDETERMINATE
rather than a silent non-error outcome.
"""

from __future__ import annotations

import pytest

from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.pairing_checks import (
    AxisPixelCalibration,
    Verdict,
    classify_range_disagreement,
    containment,
    coverage,
    display_conversion,
    scale_consistency,
)


def _check(verdict, name):
    matches = [c for c in verdict.checks if c.name == name]
    assert matches, f"no check named {name!r} in {[c.name for c in verdict.checks]}"
    return matches[0]


class TestClassifyRangeDisagreementBenignMargin:
    def test_one_sided_benign_gap_is_benign_margin(self):
        # labels 300..800 (L=500), registry extends past the top label to
        # 850 only -- the "axis framed wider than the outermost tick"
        # pattern the audit doc calls out as not necessarily an error.
        calibration = AxisPixelCalibration(
            label_lo_px=50.0, label_hi_px=950.0, image_extent_px=1100.0
        )

        verdict = classify_range_disagreement(
            label_min=300.0,
            label_max=800.0,
            reg_lo=300.0,
            reg_hi=850.0,
            scale=ScaleType.LINEAR,
            gt_extents=(310.0, 780.0),
            calibration=calibration,
        )

        assert verdict.verdict is Verdict.BENIGN_MARGIN
        assert all(c.passed is not False for c in verdict.checks)

    def test_missing_calibration_is_indeterminate_not_silently_benign(self):
        # Same as the one-sided benign gap above, but rule (d) is not
        # optional: no calibration data means we cannot confirm the
        # registry endpoint actually lands inside the image, so this must
        # come back INDETERMINATE, never a silent BENIGN_MARGIN.
        verdict = classify_range_disagreement(
            label_min=300.0,
            label_max=800.0,
            reg_lo=300.0,
            reg_hi=850.0,
            scale=ScaleType.LINEAR,
            gt_extents=(310.0, 780.0),
            calibration=None,
        )

        assert verdict.verdict is Verdict.INDETERMINATE
        check = _check(verdict, "endpoint_pixel_bounds")
        assert check.passed is None

    def test_paper_10939_figure_1528_is_real_mismatch_via_pixel_bounds_only(self):
        # Real case from the design review: y-axis labeled -110..-30 uV/K
        # (L=80), registry claimed the top endpoint as -20 (pre-correction
        # value, see registry.json evidence for paper 10939/figure 1528).
        # gap_hi = |-20 - (-30)| = 10 = 0.125L, comfortably under the
        # 0.25L margin -- rule (b) passes, so this is the SAME-unit-space
        # branch. But projecting -20 through the actual tick-label pixel
        # calibration (axis_pixel_candidates.json: y_min_px=548.5 for
        # label -110, y_max_px=13.5 for label -30, image height 650px)
        # lands at pixel -53.4 -- above the top of the 650px-tall image --
        # so only rule (d) catches it.
        calibration = AxisPixelCalibration(
            label_lo_px=548.5, label_hi_px=13.5, image_extent_px=650.0
        )

        verdict = classify_range_disagreement(
            label_min=-110.0,
            label_max=-30.0,
            reg_lo=-110.0,
            reg_hi=-20.0,
            scale=ScaleType.LINEAR,
            gt_extents=(-100.1, -42.2),
            calibration=calibration,
        )

        assert verdict.verdict is Verdict.REAL_MISMATCH
        margin_check = _check(verdict, "endpoint_margin")
        assert margin_check.passed is True  # (b) passes
        containment_check = _check(verdict, "registry_contains_gt")
        assert containment_check.passed is True  # (c) passes
        pixel_check = _check(verdict, "endpoint_pixel_bounds")
        assert pixel_check.passed is False  # (d) is the only failure
        assert "reg_hi" in pixel_check.detail

    def test_log_axis_one_sided_benign_gap_uses_log10_space(self):
        # labels 10..10000 (3 decades, L=3 in log10 space). Registry hi
        # extends to 13000 -- a gap of log10(13000)-log10(10000)=0.114
        # decades, well under 0.25*3=0.75 decades.
        calibration = AxisPixelCalibration(
            label_lo_px=900.0, label_hi_px=50.0, image_extent_px=1000.0
        )

        verdict = classify_range_disagreement(
            label_min=10.0,
            label_max=10000.0,
            reg_lo=10.0,
            reg_hi=13000.0,
            scale=ScaleType.LOG,
            gt_extents=(50.0, 9000.0),
            calibration=calibration,
        )

        assert verdict.verdict is Verdict.BENIGN_MARGIN
        margin_check = _check(verdict, "endpoint_margin")
        assert "0.11" in margin_check.detail or "0.1" in margin_check.detail

    def test_gt_extent_exactly_on_registry_boundary_is_contained(self):
        # (c) uses exact registry containment, not a margin -- a GT value
        # exactly equal to the registry endpoint must still count as
        # contained (boundary inclusive).
        calibration = AxisPixelCalibration(
            label_lo_px=100.0, label_hi_px=900.0, image_extent_px=1000.0
        )

        verdict = classify_range_disagreement(
            label_min=0.0,
            label_max=100.0,
            reg_lo=0.0,
            reg_hi=100.0,
            scale=ScaleType.LINEAR,
            gt_extents=(0.0, 100.0),
            calibration=calibration,
        )

        containment_check = _check(verdict, "registry_contains_gt")
        assert containment_check.passed is True
        assert verdict.verdict is Verdict.BENIGN_MARGIN

    def test_missing_gt_extents_makes_registry_containment_indeterminate(self):
        verdict = classify_range_disagreement(
            label_min=300.0,
            label_max=800.0,
            reg_lo=300.0,
            reg_hi=850.0,
            scale=ScaleType.LINEAR,
            gt_extents=(),
            calibration=AxisPixelCalibration(
                label_lo_px=50.0, label_hi_px=950.0, image_extent_px=1100.0
            ),
        )

        assert verdict.verdict is Verdict.INDETERMINATE
        check = _check(verdict, "registry_contains_gt")
        assert check.passed is None

    def test_degenerate_label_span_is_indeterminate(self):
        verdict = classify_range_disagreement(
            label_min=5.0,
            label_max=5.0,
            reg_lo=4.0,
            reg_hi=6.0,
            scale=ScaleType.LINEAR,
            gt_extents=(4.5, 5.5),
            calibration=AxisPixelCalibration(
                label_lo_px=100.0, label_hi_px=100.0, image_extent_px=500.0
            ),
        )

        assert verdict.verdict is Verdict.INDETERMINATE
        assert len(verdict.checks) == 1
        margin_check = verdict.checks[0]
        assert margin_check.name == "endpoint_margin"
        assert margin_check.passed is None


class TestRegistryContainmentMargin:
    """Rule (c) (`registry_contains_gt`) originally required EXACT
    containment (no margin). A 2026-09-02 measurement across all 222 axes
    of the 111 verified entries found this zero-tolerance design was itself
    producing the same kind of "flags ordinary framing margin as an error"
    false positives the pre-design-7.49 fitted factor did -- 33/40 of the
    then-current REAL_MISMATCH axes were pure rule (c) firings on ordinary
    (median 0.9%, max 6.8% of registry span) GT-vs-registry overshoot, well
    within this project's own established 15% registry-vs-GT tolerance
    (design 7.44). `_CONTAINMENT_MARGIN_FRACTION` (default 0.02) fixes
    this -- see its own docstring for the full measured distribution and
    the reasoning for 0.02 specifically (not the 0.10 that would isolate
    only the two known defects, per the design's cost asymmetry: a false
    ACCEPT silently corrupts ground truth, a false REJECT only costs a
    human a few seconds of review).
    """

    def test_overshoot_just_under_the_margin_is_benign_margin(self):
        # registry span = 500 (labels == registry, so rule (b) passes
        # exactly with zero gap), margin = 0.02*500 = 10. GT dips 9 below
        # reg_lo -- 1.8% of the span, just under the containment margin.
        calibration = AxisPixelCalibration(
            label_lo_px=50.0, label_hi_px=950.0, image_extent_px=1100.0
        )
        verdict = classify_range_disagreement(
            label_min=300.0,
            label_max=800.0,
            reg_lo=300.0,
            reg_hi=800.0,
            scale=ScaleType.LINEAR,
            gt_extents=(291.0, 780.0),
            calibration=calibration,
        )
        assert verdict.verdict is Verdict.BENIGN_MARGIN
        containment_check = _check(verdict, "registry_contains_gt")
        assert containment_check.passed is True

    def test_overshoot_just_over_the_margin_is_real_mismatch(self):
        # Same setup, but GT dips 11 below reg_lo -- 2.2% of the span, just
        # over the containment margin.
        calibration = AxisPixelCalibration(
            label_lo_px=50.0, label_hi_px=950.0, image_extent_px=1100.0
        )
        verdict = classify_range_disagreement(
            label_min=300.0,
            label_max=800.0,
            reg_lo=300.0,
            reg_hi=800.0,
            scale=ScaleType.LINEAR,
            gt_extents=(289.0, 780.0),
            calibration=calibration,
        )
        assert verdict.verdict is Verdict.REAL_MISMATCH
        containment_check = _check(verdict, "registry_contains_gt")
        assert containment_check.passed is False

    def test_paper_17044_figure_20740_negative_resistivity_still_real_mismatch(self):
        # Real numbers, design 7.53's largest measured overshoot (68.4% of
        # the registry span): registry y_range [0.001, 0.02] ohm*m, LOG
        # scale, but one GT curve dips to -0.000451613 -- a physically
        # impossible negative resistivity. Far beyond even a loosened
        # margin, and (per _registry_containment_check's docstring) a
        # non-positive value on a log-scale axis deliberately does NOT
        # short-circuit to "can't evaluate" -- it falls through to a raw
        # linear-space comparison instead, so this must still fail.
        verdict = classify_range_disagreement(
            label_min=0.001,
            label_max=0.02,
            reg_lo=0.001,
            reg_hi=0.02,
            scale=ScaleType.LOG,
            gt_extents=(-0.0004516129, 0.01275806, -0.0001612903, 0.01508065),
        )
        assert verdict.verdict is Verdict.REAL_MISMATCH
        containment_check = _check(verdict, "registry_contains_gt")
        assert containment_check.passed is False

    def test_paper_18759_figure_12217_100x_gt_error_still_real_mismatch(self):
        # Real numbers, design 7.53's second-largest measured overshoot
        # (22.1% of the registry span) -- one of four GT curves is off by
        # 100x (a Starrydata unit-tagging error, S/m vs S/cm). Far beyond
        # the 0.02 containment margin.
        verdict = classify_range_disagreement(
            label_min=250.0,
            label_max=1250.0,
            reg_lo=25000.0,
            reg_hi=135000.0,
            gt_extents=(662.2915, 103454.0),
        )
        assert verdict.verdict is Verdict.REAL_MISMATCH
        containment_check = _check(verdict, "registry_contains_gt")
        assert containment_check.passed is False

    def test_containment_margin_fraction_is_a_parameter_not_only_a_constant(self):
        # A caller can tighten (or loosen) the containment margin without
        # editing the module -- e.g. shrink it to 0.0 to recover the
        # pre-2026-09-02 exact-containment behavior for a single call.
        calibration = AxisPixelCalibration(
            label_lo_px=50.0, label_hi_px=950.0, image_extent_px=1100.0
        )
        verdict = classify_range_disagreement(
            label_min=300.0,
            label_max=800.0,
            reg_lo=300.0,
            reg_hi=800.0,
            scale=ScaleType.LINEAR,
            gt_extents=(291.0, 780.0),  # 1.8% overshoot -- passes the 0.02 default
            calibration=calibration,
            containment_margin_fraction=0.0,
        )
        assert verdict.verdict is Verdict.REAL_MISMATCH
        containment_check = _check(verdict, "registry_contains_gt")
        assert containment_check.passed is False


class TestClassifyRangeDisagreementUnitSpaceDifference:
    """Real registry-vs-label-vs-GT-vs-unit numbers from
    data/verified_pairs/, covering: the zero-endpoint case, negative-valued
    axes, an additive (Kelvin/Celsius) offset, and a clean axis-label
    scale factor -- the exact scenarios the 2026-09-02 review found the
    old ratio-only rule (a) mishandled, now resolved via the genuinely
    independent unit-string constraint (rule (e)) instead of the
    non-independent GT-containment-only gate the first version of this
    module used.

    Each real case is shown twice: WITHOUT unit strings (INDETERMINATE --
    honest "can't confirm" now that GT containment alone is known not to
    be independent evidence) and WITH them (resolves to a definite
    verdict) -- so the value of the extra constraint is visible in the
    test names themselves, per the correction's instruction.
    """

    def test_axis_scale_factor_without_units_is_indeterminate(self):
        # paper 46278, figure 51437, x-axis: registry [0.0008, 0.0018]
        # (K^-1) vs printed labels (0.8, 1.8) ("1000/T (1/K)").
        verdict = classify_range_disagreement(
            label_min=0.8,
            label_max=1.8,
            reg_lo=0.0008,
            reg_hi=0.0018,
            scale=ScaleType.LINEAR,
            gt_extents=(0.0008906, 0.001745),
        )

        assert verdict.verdict is Verdict.INDETERMINATE
        unit_check = _check(verdict, "unit_dimensional_analysis")
        assert unit_check.passed is None

    def test_axis_scale_factor_with_units_resolves_to_axis_scale_factor(self):
        # Same as above, but with the real unit strings supplied: GT unit
        # 'K^(-1)', printed axis unit '1/K' -- DIMENSIONALLY IDENTICAL
        # (dimensional analysis predicts factor 1), yet the endpoints
        # cleanly imply a=1000. This is the "axis prints its own
        # 1000/T-style scale-factor annotation" signature, not a unit
        # mismatch and not the SI-not-yet-converted backlog (nothing here
        # actually needs unit conversion) -- hence its own
        # AXIS_SCALE_FACTOR outcome rather than UNIT_SPACE_DIFFERENCE.
        verdict = classify_range_disagreement(
            label_min=0.8,
            label_max=1.8,
            reg_lo=0.0008,
            reg_hi=0.0018,
            scale=ScaleType.LINEAR,
            gt_extents=(0.0008906, 0.001745),
            gt_unit="K^(-1)",
            printed_unit="1/K",
        )

        assert verdict.verdict is Verdict.AXIS_SCALE_FACTOR
        unit_check = _check(verdict, "unit_dimensional_analysis")
        assert unit_check.passed is True
        assert "10^+3" in unit_check.detail

    def test_kelvin_celsius_offset_without_units_is_indeterminate(self):
        verdict = classify_range_disagreement(
            label_min=30.0,
            label_max=80.0,
            reg_lo=298.15,
            reg_hi=353.15,
            scale=ScaleType.LINEAR,
            gt_extents=(298.458, 348.303),
        )

        assert verdict.verdict is Verdict.INDETERMINATE

    def test_kelvin_celsius_offset_with_units_resolves_to_unit_space_difference(self):
        # paper 4965, figure 13164, x-axis: registry [298.15, 353.15] K
        # vs printed labels (30, 80) degC -- a pure additive offset a
        # ratio test can never see (fitted a~=0.909, b~=-241, not exactly
        # 1/-273.15 because the registry's low end also carries a ~5K
        # framing margin -- see _expected_additive_offset's use of
        # margin_fraction*L, which absorbs exactly this kind of
        # contamination the same way rule (b) does).
        verdict = classify_range_disagreement(
            label_min=30.0,
            label_max=80.0,
            reg_lo=298.15,
            reg_hi=353.15,
            scale=ScaleType.LINEAR,
            gt_extents=(298.458, 348.303),
            gt_unit="K",
            printed_unit="°C",
        )

        assert verdict.verdict is Verdict.UNIT_SPACE_DIFFERENCE
        unit_check = _check(verdict, "unit_dimensional_analysis")
        assert unit_check.passed is True
        assert "-273.15" in unit_check.detail

    def test_zero_registry_endpoint_with_units_resolves_to_unit_space_difference(self):
        # paper 44283, figure 38965, y-axis: registry [0.0, 2.5e-05] ohm*m
        # vs printed labels (0.5, 2.5) mOhm*cm -- the low endpoint is
        # exactly 0, which is exactly what broke the old ratio-based rule
        # (division by zero / silently skipped). The affine fit AND the
        # expected-conversion check both handle it natively via
        # subtraction, not division.
        verdict = classify_range_disagreement(
            label_min=0.5,
            label_max=2.5,
            reg_lo=0.0,
            reg_hi=2.5e-05,
            scale=ScaleType.LINEAR,
            gt_extents=(2.53165e-06, 2.11392e-05),
            gt_unit="ohm*m",
            printed_unit="mΩ·cm",
        )

        assert verdict.verdict is Verdict.UNIT_SPACE_DIFFERENCE

    def test_negative_valued_axis_with_units_resolves_to_unit_space_difference(self):
        # paper 4173, figure 20121, y-axis: registry [-5.6e-05, -4.4e-05]
        # V/K vs printed labels (-60, -40) uV/K -- x1e6 on negative
        # values.
        verdict = classify_range_disagreement(
            label_min=-60.0,
            label_max=-40.0,
            reg_lo=-5.6e-05,
            reg_hi=-4.4e-05,
            scale=ScaleType.LINEAR,
            gt_extents=(-5.519232e-05, -4.628295e-05),
            gt_unit="V*K^(-1)",
            printed_unit="μV/K",
        )

        assert verdict.verdict is Verdict.UNIT_SPACE_DIFFERENCE

    def test_large_margin_violation_without_units_is_indeterminate(self):
        # This is the exact scenario that pinned the module's original
        # "known limitation" (GT-extent containment alone cannot tell a
        # genuine unit-space difference apart from a registry endpoint
        # that's simply wrong by a wide margin but still happens to
        # contain the GT curve). Now that containment alone is no longer
        # sufficient for a confident verdict, this correctly comes back
        # INDETERMINATE rather than the previous version's (wrong, per
        # the correction) confident UNIT_SPACE_DIFFERENCE.
        verdict = classify_range_disagreement(
            label_min=300.0,
            label_max=800.0,
            reg_lo=300.0,
            reg_hi=1000.0,  # 0.4L past the label max -- clearly exceeds rule (b)'s margin
            scale=ScaleType.LINEAR,
            gt_extents=(310.0, 780.0),
        )

        assert verdict.verdict is Verdict.INDETERMINATE
        margin_check = _check(verdict, "endpoint_margin")
        assert margin_check.passed is False
        unit_check = _check(verdict, "unit_dimensional_analysis")
        assert unit_check.passed is None

    def test_large_margin_violation_with_matching_units_resolves_to_real_mismatch(self):
        # Same numbers as above, but now declaring "no unit conversion is
        # expected at all" (gt_unit == printed_unit == 'K', an arbitrary
        # same-unit choice for this synthetic case): the expected
        # conversion (factor 1, no offset) applied to the raw registry
        # endpoints does NOT reproduce the printed labels within margin,
        # and 0.714286 is not a clean power of ten either -- so unlike the
        # without-units case, this now correctly resolves to a definite
        # REAL_MISMATCH instead of staying stuck at INDETERMINATE. This is
        # the demonstration that the unit-string constraint actually
        # closes the gap the without-units test above documents.
        verdict = classify_range_disagreement(
            label_min=300.0,
            label_max=800.0,
            reg_lo=300.0,
            reg_hi=1000.0,
            scale=ScaleType.LINEAR,
            gt_extents=(310.0, 780.0),
            gt_unit="K",
            printed_unit="K",
        )

        assert verdict.verdict is Verdict.REAL_MISMATCH
        unit_check = _check(verdict, "unit_dimensional_analysis")
        assert unit_check.passed is False

    def test_incompatible_unit_dimensions_is_real_mismatch(self):
        # Synthetic numbers (no real committed example of this specific
        # failure mode), but real, already-tested unit strings: a Seebeck
        # coefficient (V*K^-1) GT curve cannot legitimately be printed on
        # a conductivity (S/cm) axis -- strong evidence of a wrong figure
        # pairing, not just a bad endpoint.
        verdict = classify_range_disagreement(
            label_min=100.0,
            label_max=200.0,
            reg_lo=0.001,
            reg_hi=0.002,
            scale=ScaleType.LINEAR,
            gt_extents=(0.0012, 0.0018),
            gt_unit="V*K^(-1)",
            printed_unit="S/cm",
        )

        assert verdict.verdict is Verdict.REAL_MISMATCH
        unit_check = _check(verdict, "unit_dimensional_analysis")
        assert unit_check.passed is False
        assert "incompatible" in unit_check.detail

    def test_incompatible_dimensions_is_real_mismatch_even_with_no_gt_extents(self):
        # Unlike a plain unit-space-difference match, incompatible
        # dimensions is decided entirely from the unit strings -- it does
        # not need GT extents (rule (c)) to resolve.
        verdict = classify_range_disagreement(
            label_min=100.0,
            label_max=200.0,
            reg_lo=0.001,
            reg_hi=0.002,
            scale=ScaleType.LINEAR,
            gt_extents=(),
            gt_unit="V*K^(-1)",
            printed_unit="S/cm",
        )

        assert verdict.verdict is Verdict.REAL_MISMATCH

    def test_axis_scale_factor_resolves_even_with_no_gt_extents(self):
        # Like the incompatible-dimensions case above, AXIS_SCALE_FACTOR
        # is decided from the unit strings + fitted slope alone -- no GT
        # extents needed.
        verdict = classify_range_disagreement(
            label_min=0.8,
            label_max=1.8,
            reg_lo=0.0008,
            reg_hi=0.0018,
            scale=ScaleType.LINEAR,
            gt_extents=(),
            gt_unit="K^(-1)",
            printed_unit="1/K",
        )

        assert verdict.verdict is Verdict.AXIS_SCALE_FACTOR

    def test_dimensionless_ambiguous_unit_is_indeterminate(self):
        # '-' is this corpus's placeholder for "no printed unit captured /
        # genuinely dimensionless" (e.g. ZT -- see paper 10939/figure
        # 1530's real y-axis, though that entry's margin actually passes
        # so it never reaches this branch). Per the correction: "do not
        # guess" -- '-' must not be treated as an implicit match.
        verdict = classify_range_disagreement(
            label_min=300.0,
            label_max=800.0,
            reg_lo=300.0,
            reg_hi=1000.0,
            scale=ScaleType.LINEAR,
            gt_extents=(310.0, 780.0),
            gt_unit="-",
            printed_unit="-",
        )

        assert verdict.verdict is Verdict.INDETERMINATE
        unit_check = _check(verdict, "unit_dimensional_analysis")
        assert unit_check.passed is None

    def test_unparseable_unit_string_is_indeterminate(self):
        verdict = classify_range_disagreement(
            label_min=300.0,
            label_max=800.0,
            reg_lo=300.0,
            reg_hi=1000.0,
            scale=ScaleType.LINEAR,
            gt_extents=(310.0, 780.0),
            gt_unit="K",
            printed_unit="???garbled???",
        )

        assert verdict.verdict is Verdict.INDETERMINATE
        unit_check = _check(verdict, "unit_dimensional_analysis")
        assert unit_check.passed is None

    def test_gt_outside_registry_range_is_real_mismatch_regardless_of_units(self):
        # paper 18759, figure 12217, y-axis: registry [25000, 135000] vs
        # printed labels (250, 1250). One of the four GT curves has
        # y-extents around 662-1029 -- entirely outside the registry's
        # stated range. Rule (c) is universal and catches this before
        # rule (e) even runs -- no unit strings needed.
        verdict = classify_range_disagreement(
            label_min=250.0,
            label_max=1250.0,
            reg_lo=25000.0,
            reg_hi=135000.0,
            scale=ScaleType.LINEAR,
            gt_extents=(662.2915, 103454.0),
        )

        assert verdict.verdict is Verdict.REAL_MISMATCH
        containment_check = _check(verdict, "registry_contains_gt")
        assert containment_check.passed is False

    def test_missing_gt_extents_in_different_unit_branch_is_indeterminate(self):
        # The unit strings here confirm a plain (margin-consistent) match
        # (Kelvin/Celsius, not a scale-factor or incompatible-dimensions
        # signal) -- that alone still requires rule (c) to have actually
        # run before a confident UNIT_SPACE_DIFFERENCE, same as the
        # same-unit branch's pattern. AXIS_SCALE_FACTOR and the
        # incompatible-dimensions REAL_MISMATCH are different: both are
        # decided purely from the unit strings + fitted slope, independent
        # of GT, so they do NOT need GT extents to resolve (see
        # test_axis_scale_factor_with_units_resolves_to_axis_scale_factor,
        # which passes real GT extents anyway but doesn't strictly need
        # to).
        verdict = classify_range_disagreement(
            label_min=30.0,
            label_max=80.0,
            reg_lo=298.15,
            reg_hi=353.15,
            scale=ScaleType.LINEAR,
            gt_extents=(),
            gt_unit="K",
            printed_unit="°C",
        )

        assert verdict.verdict is Verdict.INDETERMINATE
        check = _check(verdict, "registry_contains_gt")
        assert check.passed is None

    def test_degenerate_registry_span_cannot_fit_affine_map(self):
        verdict = classify_range_disagreement(
            label_min=0.0,
            label_max=100.0,
            reg_lo=50.0,
            reg_hi=50.0,  # zero-width registry range, far from both labels
            scale=ScaleType.LINEAR,
            gt_extents=(),
        )

        assert verdict.verdict is Verdict.INDETERMINATE
        affine_check = _check(verdict, "affine_fit")
        assert affine_check.passed is None


class TestContainment:
    def test_within_bounds_passes(self):
        result = containment((310.0, 780.0), label_min=300.0, label_max=800.0)
        assert result.passed is True

    def test_outside_margin_fails(self):
        # 900 is 100 past the label max (800), margin is 0.25*500=125, so
        # this specific value alone still passes -- push further out.
        result = containment((950.0,), label_min=300.0, label_max=800.0)
        assert result.passed is False

    def test_exactly_on_margin_boundary_passes(self):
        # margin = 0.25 * 500 = 125 -> boundary at 800+125=925
        result = containment((925.0,), label_min=300.0, label_max=800.0)
        assert result.passed is True

    def test_empty_gt_extents_is_indeterminate(self):
        result = containment((), label_min=300.0, label_max=800.0)
        assert result.passed is None

    def test_log_scale_uses_log10_space(self):
        # label 10..10000 (log10 span 3, margin 0.75 decades -> upper bound
        # 10**(4+0.75) = 10**4.75 ~= 56234)
        result = containment((56000.0,), label_min=10.0, label_max=10000.0, scale=ScaleType.LOG)
        assert result.passed is True
        result = containment((60000.0,), label_min=10.0, label_max=10000.0, scale=ScaleType.LOG)
        assert result.passed is False

    def test_non_positive_extent_on_log_axis_is_indeterminate(self):
        result = containment((-5.0,), label_min=10.0, label_max=10000.0, scale=ScaleType.LOG)
        assert result.passed is None


class TestCoverage:
    def test_within_expected_ratio_passes(self):
        # GT span 400 over L=500 -> ratio 0.8, within [0.35, 1.15]
        result = coverage((350.0, 750.0), label_min=300.0, label_max=800.0)
        assert result.passed is True

    def test_below_minimum_ratio_fails(self):
        # GT span 100 over L=500 -> ratio 0.2 < 0.35
        result = coverage((400.0, 500.0), label_min=300.0, label_max=800.0)
        assert result.passed is False

    def test_above_maximum_ratio_fails(self):
        # GT span 700 over L=500 -> ratio 1.4 > 1.15
        result = coverage((250.0, 950.0), label_min=300.0, label_max=800.0)
        assert result.passed is False

    def test_empty_gt_extents_is_indeterminate(self):
        result = coverage((), label_min=300.0, label_max=800.0)
        assert result.passed is None

    def test_degenerate_label_span_is_indeterminate(self):
        result = coverage((300.0, 300.0), label_min=300.0, label_max=300.0)
        assert result.passed is None


class TestScaleConsistency:
    def test_log_axis_with_at_least_one_decade_passes(self):
        result = scale_consistency((5.0, 5000.0), scale=ScaleType.LOG)
        assert result.passed is True

    def test_log_axis_with_less_than_one_decade_fails(self):
        result = scale_consistency((100.0, 500.0), scale=ScaleType.LOG)
        assert result.passed is False

    def test_linear_axis_under_2_5_decades_passes(self):
        # ratio 100 -> 2 decades
        result = scale_consistency((1.0, 100.0), scale=ScaleType.LINEAR)
        assert result.passed is True

    def test_linear_axis_at_or_over_2_5_decades_fails(self):
        # ratio 1000 -> 3 decades
        result = scale_consistency((1.0, 1000.0), scale=ScaleType.LINEAR)
        assert result.passed is False

    def test_non_positive_extent_is_indeterminate(self):
        result = scale_consistency((-1.0, 100.0), scale=ScaleType.LINEAR)
        assert result.passed is None

    def test_empty_gt_extents_is_indeterminate(self):
        result = scale_consistency((), scale=ScaleType.LINEAR)
        assert result.passed is None


class TestVerdictAuditTrail:
    def test_real_mismatch_reason_names_the_failing_check(self):
        verdict = classify_range_disagreement(
            label_min=250.0,
            label_max=1250.0,
            reg_lo=25000.0,
            reg_hi=135000.0,
            gt_extents=(662.2915, 103454.0),
        )
        assert "registry_contains_gt" in verdict.reason

    def test_same_unit_branch_checks_tuple_has_the_three_named_sub_rules(self):
        verdict = classify_range_disagreement(
            label_min=300.0, label_max=800.0, reg_lo=300.0, reg_hi=850.0
        )
        names = {c.name for c in verdict.checks}
        assert names == {"endpoint_margin", "registry_contains_gt", "endpoint_pixel_bounds"}

    def test_different_unit_branch_checks_tuple_includes_affine_fit_and_unit_check(self):
        verdict = classify_range_disagreement(
            label_min=0.8,
            label_max=1.8,
            reg_lo=0.0008,
            reg_hi=0.0018,
            gt_extents=(0.001,),
            gt_unit="K^(-1)",
            printed_unit="1/K",
        )
        names = {c.name for c in verdict.checks}
        assert names == {
            "endpoint_margin",
            "registry_contains_gt",
            "affine_fit",
            "unit_dimensional_analysis",
            "endpoint_pixel_bounds",
        }

    def test_axis_scale_factor_reason_names_the_unit_check(self):
        verdict = classify_range_disagreement(
            label_min=0.8,
            label_max=1.8,
            reg_lo=0.0008,
            reg_hi=0.0018,
            gt_extents=(0.0008906, 0.001745),
            gt_unit="K^(-1)",
            printed_unit="1/K",
        )
        assert "unit_dimensional_analysis" in verdict.reason


class TestDisplayConversion:
    """`display_conversion` is the direct replacement for
    `generate_verified_pairs_visual_audit.py`'s old `_derive_factor`: given a
    verdict (from `classify_range_disagreement` on the same axis), which
    conversion is safe to apply to re-express a registry-space value (a GT
    curve point, or the registry range itself) in the printed axis's display
    units for the re-plot.
    """

    def test_benign_margin_is_identity(self):
        # Registry is already display-unit (design 7.47) -- no conversion.
        assert display_conversion(
            Verdict.BENIGN_MARGIN,
            label_min=300.0,
            label_max=800.0,
            reg_lo=300.0,
            reg_hi=850.0,
        ) == (1.0, 0.0)

    def test_unit_space_difference_returns_expected_multiplicative_factor(self):
        # paper 44283/38965, y-axis: ohm*m -> mOhm*cm is x1e5.
        factor, offset = display_conversion(
            Verdict.UNIT_SPACE_DIFFERENCE,
            label_min=0.5,
            label_max=2.5,
            reg_lo=0.0,
            reg_hi=2.5e-05,
            gt_unit="ohm*m",
            printed_unit="mΩ·cm",
        )
        assert offset == 0.0
        assert factor == pytest.approx(1e5, rel=1e-6)

    def test_unit_space_difference_returns_expected_additive_offset(self):
        # paper 4965/13164, x-axis: K -> degC.
        factor, offset = display_conversion(
            Verdict.UNIT_SPACE_DIFFERENCE,
            label_min=30.0,
            label_max=80.0,
            reg_lo=298.15,
            reg_hi=353.15,
            gt_unit="K",
            printed_unit="°C",
        )
        assert factor == pytest.approx(1.0)
        assert offset == pytest.approx(-273.15)

    def test_unit_space_difference_without_units_falls_back_to_identity(self):
        # Defensive: shouldn't happen from a real classify_range_disagreement
        # call (UNIT_SPACE_DIFFERENCE requires resolved unit strings), but
        # display_conversion doesn't trust its caller blindly either.
        assert display_conversion(
            Verdict.UNIT_SPACE_DIFFERENCE,
            label_min=30.0,
            label_max=80.0,
            reg_lo=298.15,
            reg_hi=353.15,
        ) == (1.0, 0.0)

    def test_axis_scale_factor_reproduces_the_exact_endpoint_fit(self):
        # paper 46278/51437, x-axis: registry K^-1 [0.0008, 0.0018] vs
        # printed 1000/T labels (0.8, 1.8) -- a=1000, b=0.
        factor, offset = display_conversion(
            Verdict.AXIS_SCALE_FACTOR,
            label_min=0.8,
            label_max=1.8,
            reg_lo=0.0008,
            reg_hi=0.0018,
        )
        assert factor == pytest.approx(1000.0)
        assert offset == pytest.approx(0.0, abs=1e-9)

    def test_axis_scale_factor_with_degenerate_registry_span_falls_back_to_identity(self):
        assert display_conversion(
            Verdict.AXIS_SCALE_FACTOR,
            label_min=0.0,
            label_max=100.0,
            reg_lo=50.0,
            reg_hi=50.0,
        ) == (1.0, 0.0)

    def test_real_mismatch_is_identity_raw_si_fallback(self):
        assert display_conversion(
            Verdict.REAL_MISMATCH,
            label_min=300.0,
            label_max=800.0,
            reg_lo=300.0,
            reg_hi=1000.0,
        ) == (1.0, 0.0)

    def test_indeterminate_is_identity_raw_si_fallback(self):
        assert display_conversion(
            Verdict.INDETERMINATE,
            label_min=300.0,
            label_max=800.0,
            reg_lo=300.0,
            reg_hi=1000.0,
        ) == (1.0, 0.0)

    def test_dispatches_correctly_from_a_real_classify_range_disagreement_call(self):
        # End-to-end: feed classify_range_disagreement's own verdict straight
        # into display_conversion, the way the audit script will.
        verdict = classify_range_disagreement(
            label_min=0.8,
            label_max=1.8,
            reg_lo=0.0008,
            reg_hi=0.0018,
            gt_extents=(0.0008906, 0.001745),
            gt_unit="K^(-1)",
            printed_unit="1/K",
        )
        assert verdict.verdict is Verdict.AXIS_SCALE_FACTOR
        factor, offset = display_conversion(
            verdict.verdict,
            label_min=0.8,
            label_max=1.8,
            reg_lo=0.0008,
            reg_hi=0.0018,
            gt_unit="K^(-1)",
            printed_unit="1/K",
        )
        assert factor == pytest.approx(1000.0)
        assert offset == pytest.approx(0.0, abs=1e-9)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
