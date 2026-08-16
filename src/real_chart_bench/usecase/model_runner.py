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


class ModelRunnerPort(Protocol):
    def extract(self, task: ExtractionTask) -> list[Curve]: ...
