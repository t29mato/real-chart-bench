"""Port for any chart-data-extraction model (design §4.2 ModelRunnerPort,
§7.15 evaluation harness). LLMs, dedicated models (LineFormer), and classic
CV tools all implement this same interface.

v0 scope (§7.15): the task supplies axis calibration (x_range/y_range/
x_scale) alongside the image — see domain/pixel_calibration.py for why.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from real_chart_bench.domain.curve import Curve, ScaleType


@dataclass(frozen=True)
class ExtractionTask:
    image_bytes: bytes
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    x_scale: ScaleType = ScaleType.LINEAR
    y_scale: ScaleType = ScaleType.LINEAR  # design §7.25
    # design §7.84: the second y axis, printed on the right, when the figure
    # has one. It shares the x axis. None -- one y axis -- is every figure of
    # the dataset today.
    y2_range: tuple[float, float] | None = None
    y2_scale: ScaleType = ScaleType.LINEAR

    def __post_init__(self) -> None:
        if self.y2_range is None and self.y2_scale is not ScaleType.LINEAR:
            raise ValueError("y2_scale requires y2_range (the second axis's extent)")


class ModelRunnerPort(Protocol):
    def extract(self, task: ExtractionTask) -> list[Curve]: ...
