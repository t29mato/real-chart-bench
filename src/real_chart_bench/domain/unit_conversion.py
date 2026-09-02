"""Unit-string dimensional analysis for cross-checking chart axis units.

Two unit strings describing the *same physical quantity* -- Starrydata's
fixed SI notation (`ground_truth.json`'s `unit_x`/`unit_y`, e.g. "ohm*m",
"V*K^(-1)") and a display unit read off a chart's printed axis (e.g.
"uOhm*cm", "S/cm", "10^4 S/m") -- reduce to a single multiplicative factor:
how many display-units equal one SI-unit. Given both unit strings, this
predicts the SI -> display conversion factor from the *unit names
themselves*, independent of any numeric tick-label reading. Cross-checking
this dimensional-analysis-derived factor against the numeric factor already
derivable from `axis_pixel_candidates.json`'s printed tick values (see
`scripts/eval/generate_verified_pairs_visual_audit.py`'s `_derive_factor`)
gives two independent routes to the same number -- agreement is a strong
signal that both Starrydata's stored curve and the registry's transcribed
calibration are correct; disagreement flags one of the three (Starrydata's
raw value, the registry's numbers, or a misread axis unit) as suspect.

No I/O, pure functions -- adapter concerns (reading the unit string off an
image) live elsewhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class UnitParseError(ValueError):
    """A unit string could not be parsed into known base units."""


class IncompatibleUnitsError(ValueError):
    """Two unit strings don't describe the same physical dimension."""


# Metric prefixes, longest-symbol-first so "da" is tried before "d". A
# prefix must be followed by a recognized base unit to count -- "m" alone
# (no base unit following) is the base unit "meter", not a bare prefix.
_PREFIXES: list[tuple[str, float]] = [
    ("da", 1e1),
    ("n", 1e-9),
    ("u", 1e-6),
    ("μ", 1e-6),  # μ (Greek mu, U+03BC)
    ("µ", 1e-6),  # µ (micro sign, U+00B5)
    ("m", 1e-3),
    ("c", 1e-2),
    ("d", 1e-1),
    ("k", 1e3),
    ("M", 1e6),
    ("G", 1e9),
]

# Base unit symbols this domain actually uses. "ohm"/"Ohm"/"OHM" all match
# case-insensitively (charts spell it inconsistently); everything else is
# case-sensitive since case distinguishes real units here (k=kilo vs K=kelvin,
# m=milli/meter vs M=mega). "S" (siemens) folds into "ohm" with a negated
# exponent (S = ohm^-1) so conductivity/resistivity always compare on the
# same basis regardless of which spelling a chart uses. "cm" is not a
# separate base -- it is meter with a fixed extra 1e-2 scale, handled by the
# normal prefix mechanism (rather than a distinct dimension) so "cm" and
# "m" always compare as the same physical length.
_CASE_INSENSITIVE_BASES = ("ohm", "Ω")
_CASE_SENSITIVE_BASES = ("V", "W", "K", "s", "m")


def _match_base(token_lower: str, token: str) -> str | None:
    for base in _CASE_INSENSITIVE_BASES:
        if token_lower == base.lower():
            return "ohm"
    for base in _CASE_SENSITIVE_BASES:
        if token == base:
            return base
    return None


_SUPERSCRIPT_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPT_MINUS = "⁻"


@dataclass(frozen=True)
class _ParsedUnit:
    dims: dict[str, int]  # base unit -> exponent
    scale: float  # accumulated multiplier relative to pure SI base units


def _insert_carets_for_superscripts(text: str) -> str:
    """Turns e.g. 'cm²' into 'cm^2' and 'K⁻²' into 'K^-2' before the digits
    are translated to plain ASCII, so exponents are explicit for the tokenizer."""
    out = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in _SUPERSCRIPT_DIGITS or ch == _SUPERSCRIPT_MINUS:
            out.append("^")
            while i < len(text) and (
                text[i] in _SUPERSCRIPT_DIGITS or text[i] == _SUPERSCRIPT_MINUS
            ):
                if text[i] == _SUPERSCRIPT_MINUS:
                    out.append("-")
                else:
                    out.append(str(_SUPERSCRIPT_DIGITS.index(text[i])))
                i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _normalize(text: str) -> str:
    text = _insert_carets_for_superscripts(text)
    text = text.replace("·", "*").replace("∕", "/")
    # A "." between letters (e.g. "mOhm.cm") is multiplication; a "." used
    # as a decimal point (inside "10^-3", numbers) never appears in unit
    # text here, so treating every "." as "*" is safe for this domain.
    text = text.replace(".", "*")
    # Strip grouping parens that aren't part of an exponent: turn "^(-1)"
    # into "^-1" first, then remove any parens still remaining.
    text = re.sub(r"\^\((-?\d+)\)", r"^\1", text)
    text = text.replace("(", "").replace(")", "")
    # Degree Celsius/old-style degree-Kelvin are dimensionally Kelvin for
    # span/scale purposes (see TestRealWorldNotationVariants for why the
    # absolute-value offset is a separate, additive concern this module
    # doesn't need to handle).
    text = text.replace("°C", "K").replace("degC", "K").replace("°K", "K")
    return text.strip()


_PREFIXES_BY_LENGTH = sorted(_PREFIXES, key=lambda p: -len(p[0]))

# Kelvin is never spelled with a metric prefix in this domain (a milli-Kelvin
# axis would be absurd for materials measured at 2-1000K) -- "mK", "cK" etc.
# always mean the base units "m" and "K" written back-to-back with an
# implicit multiplication (e.g. "W/mK" = W/(m*K)), never "prefixed Kelvin".
# Excluding K from prefix-matching is what resolves that ambiguity.
_PREFIXABLE_BASES = ("V", "W", "s", "m", "ohm")


def _match_exponent(chunk: str, j: int) -> tuple[int, int]:
    """Returns (exponent, new_position) for an optional '^N' at position j."""
    if j < len(chunk) and chunk[j] == "^":
        m = re.match(r"\^(-?\d+)", chunk[j:])
        if m:
            return int(m.group(1)), j + len(m.group(0))
    return 1, j


def _lex_chunk(chunk: str) -> list[tuple[str, int]]:
    """Tokenizes one `*`/`/`-free chunk into (base_unit, local_exponent)
    pairs, handling implicit concatenation (e.g. "mK^2" = "m" * "K^2",
    "S" folded to "ohm^-1"). local_exponent does not yet include the
    chunk's overall +/- sign from its position relative to a "/".

    Prefix+base combinations (e.g. "m"+"V" = millivolt) are tried before a
    bare single-letter base match, so "mV" doesn't greedily consume just
    "m" as the base unit "meter" and leave a dangling "V". The reverse
    ambiguity ("mK" as milli-Kelvin vs. "m"*"K") is resolved by never
    letting Kelvin take a prefix at all (see `_PREFIXABLE_BASES`).
    """
    terms: list[tuple[str, int]] = []
    i = 0
    n = len(chunk)
    while i < n:
        # Bare capital "S" is always siemens in this domain -- no prefix
        # or other base symbol starts with "S", so there's no ambiguity to
        # guard against (unlike "m", which is also the meter symbol).
        if chunk[i] == "S":
            exp, j = _match_exponent(chunk, i + 1)
            terms.append(("ohm", -exp))  # S = ohm^-1
            i = j
            continue

        matched = False
        for psym, pscale in _PREFIXES_BY_LENGTH:
            if not chunk[i:].startswith(psym):
                continue
            rest_start = i + len(psym)
            for blen in range(min(4, n - rest_start), 0, -1):
                candidate = chunk[rest_start : rest_start + blen]
                base = _match_base(candidate.lower(), candidate)
                if base is None or base not in _PREFIXABLE_BASES:
                    continue
                j = rest_start + blen
                exp, j = _match_exponent(chunk, j)
                terms.append((base, exp))
                terms.append((f"__scale__{psym}", exp))
                i = j
                matched = True
                break
            if matched:
                break
        if matched:
            continue

        for length in range(min(4, n - i), 0, -1):
            candidate = chunk[i : i + length]
            base = _match_base(candidate.lower(), candidate)
            if base is None:
                continue
            j = i + length
            exp, j = _match_exponent(chunk, j)
            terms.append((base, exp))
            i = j
            matched = True
            break
        if matched:
            continue

        raise UnitParseError(f"could not parse unit text at {chunk[i:]!r} in {chunk!r}")
    return terms


def _split_terms(text: str) -> list[tuple[str, int]]:
    text = _normalize(text)
    if text in ("", "-"):
        return []

    decade_terms: list[tuple[str, int]] = []
    decade_match = re.match(r"^[×x]?10\^?(-?\d+)\s+(.*)$", text)
    if decade_match:
        exponent = int(decade_match.group(1))
        decade_terms = [(f"__decade__{exponent}", 1)]
        text = decade_match.group(2)

    all_terms: list[tuple[str, int]] = list(decade_terms)
    sign = 1
    for piece in re.split(r"([*/])", text):
        piece = piece.strip()
        if piece == "*":
            continue
        if piece == "/":
            sign = -1
            continue
        if not piece:
            continue
        if re.fullmatch(r"1(\.0+)?", piece):
            # A bare "1" numerator (e.g. "1/K", the reciprocal-unit
            # notation common on Arrhenius-plot axes, design 7.42) is
            # dimensionless and contributes no term -- distinct from "-"
            # (handled above, meaning "no unit at all"), this is "no unit
            # *in this position*", e.g. the "1" in "1/K" vs the "K" that
            # follows it.
            continue
        # Whitespace between unit factors is implicit multiplication (e.g.
        # "W m^-1 K^-1"), same as an explicit "*" would be.
        for sub_piece in piece.split():
            for base, exp in _lex_chunk(sub_piece):
                all_terms.append((base, exp * sign))
    return all_terms


def _parse_unit(text: str) -> _ParsedUnit:
    terms = _split_terms(text)
    dims: dict[str, int] = {}
    scale = 1.0
    for token, exp in terms:
        if token.startswith("__decade__"):
            decade_exp = int(token[len("__decade__") :])
            scale *= (10.0**decade_exp) ** exp
        elif token.startswith("__scale__"):
            psym = token[len("__scale__") :]
            pscale = next(p for s, p in _PREFIXES if s == psym)
            scale *= pscale**exp
        else:
            dims[token] = dims.get(token, 0) + exp
    dims = {k: v for k, v in dims.items() if v != 0}
    return _ParsedUnit(dims=dims, scale=scale)


def si_to_display_factor(si_unit: str, display_unit: str) -> float:
    """How many `display_unit`s equal one `si_unit`.

    Raises `UnitParseError` if either string can't be parsed, and
    `IncompatibleUnitsError` if they parse to different physical
    dimensions (e.g. comparing a resistivity unit against a Seebeck-
    coefficient unit).
    """
    si = _parse_unit(si_unit)
    display = _parse_unit(display_unit)
    if si.dims != display.dims:
        raise IncompatibleUnitsError(
            f"{si_unit!r} ({si.dims}) and {display_unit!r} ({display.dims}) "
            "are not the same physical dimension"
        )
    return si.scale / display.scale
