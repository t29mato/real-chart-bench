"""Curve: the domain value object for a single (x, y) data series.

Pure value object — no I/O, no external-layer imports (see domain/__init__.py).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum


class ScaleType(Enum):
    """Axis scale. Only x_scale is modeled (see design §3.2/§3.3 and §4.2 class
    diagram): distance is computed by interpolating in log(x) space when the
    chart's x-axis is logarithmic, since linear interpolation across a log
    axis misrepresents the underlying sampling."""

    LINEAR = "linear"
    LOG = "log"


def _as_float_tuple(values, field_name: str) -> tuple[float, ...]:
    try:
        return tuple(float(v) for v in values)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must contain only numeric values") from exc


@dataclass(frozen=True)
class Curve:
    """A single data series: parallel x/y value tuples plus metadata.

    Points are sorted by x on construction so downstream interpolation
    (numpy.interp requires non-decreasing x) is deterministic regardless of
    input order.

    Invariants:
    - at least one point (empty curves are rejected; "zero predicted series"
      is represented at the *list* level as an empty ``list[Curve]``, not as
      a Curve with zero points — see HungarianCurveMatcher)
    - x_values and y_values have equal length
    - x_values/y_values must be numeric (categorical axes must be
      ordinal-encoded upstream — see docs/design §3.3 and
      tests/domain/test_categorical_x_axis_scope.py)
    """

    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    series_label: str = ""
    x_scale: ScaleType = ScaleType.LINEAR
    x_min: float = field(init=False, repr=False)
    x_max: float = field(init=False, repr=False)
    y_min: float = field(init=False, repr=False)
    y_max: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        x = _as_float_tuple(self.x_values, "x_values")
        y = _as_float_tuple(self.y_values, "y_values")

        if len(x) == 0:
            raise ValueError("Curve must have at least one point")
        if len(x) != len(y):
            raise ValueError("x_values and y_values must have the same length")

        order = sorted(range(len(x)), key=lambda i: x[i])
        x_sorted = tuple(x[i] for i in order)
        y_sorted = tuple(y[i] for i in order)

        # frozen dataclass: use object.__setattr__ to set derived/normalized fields
        object.__setattr__(self, "x_values", x_sorted)
        object.__setattr__(self, "y_values", y_sorted)
        object.__setattr__(self, "x_min", x_sorted[0])
        object.__setattr__(self, "x_max", x_sorted[-1])
        object.__setattr__(self, "y_min", min(y_sorted))
        object.__setattr__(self, "y_max", max(y_sorted))

    def __len__(self) -> int:
        return len(self.x_values)


def curves_for_curve_scoring(curves: Sequence[Curve]) -> list[Curve]:
    """The predicted curves as the curve-distance metric sees them (owner
    decision 2026-10-05): a point whose x or y is non-finite (inf / nan) is
    ignored, and a series left with fewer than two finite points leaves curve
    scoring altogether.

    Only series that held a non-finite value are touched: an all-finite curve
    is passed through as the same object, whatever its length. Finite values
    that cannot be placed on a log axis (<= 0) are *not* removed here -- the
    metric still rejects them, as before.

    The point metric does not use this: there a non-finite point stays a
    predicted point that never matches (domain/point_metrics.py).
    """
    out: list[Curve] = []
    for curve in curves:
        keep = [
            i
            for i, (x, y) in enumerate(zip(curve.x_values, curve.y_values, strict=True))
            if math.isfinite(x) and math.isfinite(y)
        ]
        if len(keep) == len(curve):
            out.append(curve)
        elif len(keep) >= 2:
            out.append(
                Curve(
                    x_values=tuple(curve.x_values[i] for i in keep),
                    y_values=tuple(curve.y_values[i] for i in keep),
                    series_label=curve.series_label,
                    x_scale=curve.x_scale,
                )
            )
    return out
