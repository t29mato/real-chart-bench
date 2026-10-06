"""Ground truth in each figure's printed space (design 7.82, owner decision
2026-10-06).

Thirteen y axes print log10 of the quantity, and one x axis prints degC while
Starrydata stores kelvin. Until 2026-10-06 the ground truth kept the
quantity (10^tick, K) and the noaxis prompt told the model to convert. Now
the ground truth holds what the axis prints, so one "report as printed" rule
serves every axis; the answers given under the old rules are mapped back
here, at scoring time. Both maps are exact inverses of the old rules.
"""

from __future__ import annotations

import math

# figure_id -> {axis: op}; op maps an old-rule value to the printed value
PRINTED_SPACE_MIGRATION: dict[str, dict[str, str]] = {
    **{f: {"y": "log10"} for f in ("51437", "51438", "51439", "51440", "51441", "51442")},
    "40067": {"y": "log10"},
    "45356": {"y": "log10"},
    "45360": {"y": "log10"},
    "40587": {"y": "log10"},
    "45818": {"y": "log10"},
    "45323": {"y": "log10"},
    "48871": {"y": "log10"},
    "20121": {"x": "k_to_degc"},
}


def to_printed(value: float, op: str) -> float | None:
    """None when the old-rule value has no printed counterpart (log10 of a
    non-positive number)."""
    if op == "log10":
        return math.log10(value) if value > 0 else None
    if op == "k_to_degc":
        return value - 273.15
    raise ValueError(f"unknown op {op!r}")


def _is_number(v) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def answer_to_printed(answer: list, ops: dict[str, str]) -> list:
    """Convert the named axes of every series. A value with no printed
    counterpart becomes NaN: the point stays a predicted point that matches
    nothing, exactly as the old scoring treated a non-positive value on a log
    axis. Anything that is not a number (or a series that is not a dict) is
    left for the curve parser to drop."""

    def conv(vs, op):
        out = []
        for v in vs:
            if _is_number(v):
                p = to_printed(v, op)
                out.append(math.nan if p is None else p)
            else:
                out.append(v)
        return out

    out = []
    for s in answer:
        if not isinstance(s, dict):
            out.append(s)
            continue
        s = dict(s)
        for axis, op in ops.items():
            s[axis] = conv(list(s.get(axis) or []), op)
        out.append(s)
    return out


def _inside_share(values, lo: float, hi: float) -> float:
    span = (hi - lo) or 1.0
    nums = [v for v in values if _is_number(v) and not math.isnan(v)]
    if not nums:
        return 0.0
    return sum(lo - 0.05 * span <= v <= hi + 0.05 * span for v in nums) / len(nums)


def answer_to_printed_if_old(
    answer: list, ops: dict[str, str], printed_ranges: dict[str, tuple[float, float]]
) -> tuple[list, bool]:
    """For runs whose answers may be in either space (an agent told the old
    rule that answered through a tool reading printed ticks): convert only
    when the answer's own values sit in the old space -- decided per figure
    by which space puts more of the values inside the printed axis range
    (widened by 5% of its span), never by the score."""
    converted = answer_to_printed(answer, ops)
    as_is = conv = 0.0
    for axis in ops:
        lo, hi = printed_ranges[axis]
        as_is += _inside_share(
            [v for s in answer if isinstance(s, dict) for v in s.get(axis) or []], lo, hi
        )
        conv += _inside_share(
            [v for s in converted if isinstance(s, dict) for v in s.get(axis) or []], lo, hi
        )
    return (converted, True) if conv > as_is else (answer, False)
