"""Tests for domain/unit_conversion.py.

Two unit strings for the *same physical quantity* (Starrydata's fixed SI
notation on one side, a human-written display unit read off a chart's axis
on the other) should reduce to a single multiplicative factor: how many
display-units equal one SI-unit. This is used to independently predict the
SI->display conversion factor from the *unit names themselves* (dimensional
analysis), as a cross-check against the numeric factor already derivable
from axis_pixel_candidates.json's printed tick values (see
scripts/eval/generate_verified_pairs_visual_audit.py's `_derive_factor`) --
two independent methods that should agree if both Starrydata's raw value
and the registry's calibration are correct.

Starrydata's own unit strings (ground_truth.json's `unit_x`/`unit_y`) follow
a fixed machine format: base units joined with `*`, exponents as `^(N)`,
e.g. "ohm*m", "V*K^(-1)", "W*m^(-1)*K^(-2)", "m^(-3)", never with a metric
prefix (always base SI). Display units, read off a chart axis by a human or
vision model, are far less regular: "uV/K", "S/cm", "mOhm.cm", "10^4 S/m",
"cm^-3", "W/mK", "-" (dimensionless), etc.
"""

from __future__ import annotations

import math

import pytest

from real_chart_bench.domain.unit_conversion import (
    IncompatibleUnitsError,
    UnitParseError,
    si_to_display_factor,
)


def _approx(a: float, b: float, rel: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=rel)


class TestSimplePrefixedUnits:
    def test_volt_to_microvolt_is_1e6(self):
        assert _approx(si_to_display_factor("V*K^(-1)", "uV/K"), 1e6)

    def test_volt_to_millivolt_is_1e3(self):
        assert _approx(si_to_display_factor("V*K^(-1)", "mV/K"), 1e3)

    def test_same_unit_both_sides_is_1(self):
        assert _approx(si_to_display_factor("V*K^(-1)", "V/K"), 1.0)

    def test_mu_and_u_prefix_spellings_are_equivalent(self):
        assert _approx(si_to_display_factor("V*K^(-1)", "μV/K"), 1e6)


class TestLengthPrefixCombinations:
    def test_resistivity_ohm_m_to_micro_ohm_cm(self):
        # uOhm*cm = 1e-6 ohm * 1e-2 m = 1e-8 ohm*m -> factor = 1/1e-8 = 1e8
        assert _approx(si_to_display_factor("ohm*m", "uOhm*cm"), 1e8)

    def test_resistivity_ohm_m_to_milli_ohm_cm(self):
        # mOhm*cm = 1e-3 * 1e-2 = 1e-5 ohm*m -> factor = 1e5
        assert _approx(si_to_display_factor("ohm*m", "mOhm.cm"), 1e5)

    def test_resistivity_ohm_m_to_mega_ohm_cm(self):
        # MOhm*cm = 1e6 * 1e-2 = 1e4 ohm*m -> factor = 1/1e4 = 1e-4
        assert _approx(si_to_display_factor("ohm*m", "MOhm.cm"), 1e-4)

    def test_conductivity_siemens_per_meter_to_siemens_per_cm(self):
        # S/cm = S / 1e-2 m = 100 S/m -> factor = 1/100
        assert _approx(si_to_display_factor("ohm^(-1)*m^(-1)", "S/cm"), 0.01)

    def test_conductivity_via_ohm_inverse_matches_siemens_spelling(self):
        assert _approx(
            si_to_display_factor("ohm^(-1)*m^(-1)", "ohm^(-1)*cm^(-1)"),
            si_to_display_factor("ohm^(-1)*m^(-1)", "S/cm"),
        )


class TestExplicitDecadeMultiplier:
    def test_leading_10n_multiplier_s_per_m(self):
        # display reads directly in units of 10^4 S/m -> 1 (10^4 S/m) = 1e4 S/m
        # SI->display factor = 1 / 1e4
        assert _approx(si_to_display_factor("ohm^(-1)*m^(-1)", "10^4 S/m"), 1e-4)

    def test_negative_decade_exponent(self):
        # display reads in units of 10^-3 S/m -> 1 (10^-3 S/m) = 1e-3 S/m
        # SI->display factor = 1 / 1e-3 = 1e3
        assert _approx(si_to_display_factor("ohm^(-1)*m^(-1)", "10^-3 S/m"), 1e3)

    def test_power_factor_micro_watt_per_cm_per_k2(self):
        # uW/cm/K^2 = 1e-6 W / (1e-2 m) / K^2 = 1e-4 W/m/K^2 -> factor 1e4
        assert _approx(
            si_to_display_factor("W*m^(-1)*K^(-2)", "uW/cm/K^2"), 1e4
        )

    def test_power_factor_1e_minus_3_w_per_mk2(self):
        # display already in units of 1e-3 W/m/K^2 -> factor 1e3
        assert _approx(
            si_to_display_factor("W*m^(-1)*K^(-2)", "10^-3 W/mK^2"), 1e3
        )


class TestThermalConductivityNoConversion:
    def test_w_per_m_k_matches_si_directly(self):
        assert _approx(
            si_to_display_factor("W*m^(-1)*K^(-1)", "W/mK"), 1.0
        )

    def test_w_per_m_k_with_dot_notation(self):
        assert _approx(
            si_to_display_factor("W*m^(-1)*K^(-1)", "W/(m·K)"), 1.0
        )


class TestCarrierConcentrationAndMobility:
    def test_per_cubic_meter_to_per_cubic_cm(self):
        # cm^-3 = (1e-2 m)^-3 = 1e6 m^-3 -> factor = 1/1e6
        assert _approx(si_to_display_factor("m^(-3)", "cm^(-3)"), 1e-6)

    def test_mobility_m2_per_vs_to_cm2_per_vs(self):
        # cm^2/V/s = (1e-2 m)^2 / V / s = 1e-4 m^2/V/s -> factor = 1e4
        assert _approx(
            si_to_display_factor("m^2*V^(-1)*s^(-1)", "cm^2/V/s"), 1e4
        )


class TestUnicodeSuperscripts:
    def test_squared_superscript_digit(self):
        # cm² = (1e-2 m)^2 = 1e-4 m^2 -> factor 1e4
        assert _approx(
            si_to_display_factor("m^2*V^(-1)*s^(-1)", "cm²/V/s"), 1e4
        )

    def test_negative_superscript_exponent(self):
        assert _approx(si_to_display_factor("m^(-3)", "cm⁻³"), 1e-6)


class TestRedundantSeparatorIsIgnored:
    def test_double_asterisk_does_not_break_parsing(self):
        # a stray double separator (e.g. "V**K^-1") should still parse as
        # if it were a single "*" -- the empty piece between the two stars
        # is simply skipped.
        assert _approx(si_to_display_factor("V*K^(-1)", "V**K^-1"), 1.0)


class TestDimensionlessAndMissingUnit:
    def test_dimensionless_both_sides_is_1(self):
        assert _approx(si_to_display_factor("-", "-"), 1.0)

    def test_empty_string_treated_as_dimensionless(self):
        assert _approx(si_to_display_factor("", ""), 1.0)


class TestRealWorldNotationVariants:
    """Regression cases from cross-checking two independent vision-model
    reads of all 111 verified_pairs axis labels (2026-09-01): every
    disagreement between the two passes turned out to be a parser gap, not
    an actual reading disagreement -- these lock in the fixes."""

    def test_omega_symbol_for_ohm(self):
        assert _approx(si_to_display_factor("ohm*m", "MΩ·cm"), 1e-4)

    def test_omega_with_micro_prefix(self):
        assert _approx(si_to_display_factor("ohm*m", "μΩcm"), 1e8)

    def test_omega_inverse_space_separated(self):
        assert _approx(
            si_to_display_factor("ohm^(-1)*m^(-1)", "Ω^-1 cm^-1"), 0.01
        )

    def test_space_separated_implicit_multiplication(self):
        assert _approx(
            si_to_display_factor("W*m^(-1)*K^(-1)", "W m^-1 K^-1"), 1.0
        )

    def test_space_separated_micro_volt_kelvin(self):
        assert _approx(si_to_display_factor("V*K^(-1)", "μV K^-1"), 1e6)

    def test_mobility_space_separated(self):
        assert _approx(
            si_to_display_factor("m^2*V^(-1)*s^(-1)", "cm^2 V^-1 s^-1"), 1e4
        )

    def test_degree_celsius_is_dimensionally_kelvin(self):
        # For span/scale comparisons (not absolute-value conversion) a
        # degree Celsius is the same size as a Kelvin -- the offset between
        # the two scales is a separate, additive concern (see
        # generate_verified_pairs_visual_audit.py's `_derive_factor`, which
        # already handles degC-vs-K as an additive relationship). This
        # module only judges *dimension* + *multiplicative scale*.
        assert _approx(si_to_display_factor("K", "degC"), 1.0)
        assert _approx(si_to_display_factor("K", "°C"), 1.0)

    def test_times_sign_decade_multiplier(self):
        assert _approx(si_to_display_factor("ohm^(-1)*m^(-1)", "×10^4 S·m^-1"), 1e-4)

    def test_siemens_directly_adjacent_to_next_unit_no_separator(self):
        # "Scm^-1" (no space/operator between S and cm) -- S/cm
        assert _approx(si_to_display_factor("ohm^(-1)*m^(-1)", "Scm^-1"), 0.01)


class TestIncompatibleOrUnparseableUnits:
    def test_incompatible_dimensions_raises(self):
        with pytest.raises(IncompatibleUnitsError):
            si_to_display_factor("ohm*m", "V/K")

    def test_unparseable_display_unit_raises(self):
        with pytest.raises(UnitParseError):
            si_to_display_factor("ohm*m", "some nonsense text with no units")
