"""LineFormer (ICDAR 2023, pretrained) behind ModelRunnerPort -- by replay.

LineFormer needs torch 1.13 / mmcv-full 1.7 / its vendored mmdetection under
Python 3.10 (design §7.16, §7.35-§7.37), none of which this package can
import. Inference therefore runs out of process in that env
(scripts/eval/lineformer/worker.py), loading the model once and writing one
JSONL record per image of raw *pixel-space* series. This adapter replays those
records: it looks the task's image up by content hash and maps pixels to data
space with the task's calibration.

Keeping the raw pixel output (not just scores) means the pixel->data mapping
can be changed and rescored without another GPU run.

Pixel->data mapping: the plot area is taken to be the full image frame, the
same choice notebooks/lineformer_colab.ipynb made for the n42 run, so this run
measures the same method on the current figures. LineFormer itself detects
lines, not axes; any better plot-area estimate is a separate method.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.pixel_calibration import PixelCalibration
from real_chart_bench.usecase.model_runner import ExtractionTask

Series = tuple[tuple[float, float], ...]


def image_key(image_bytes: bytes) -> str:
    return hashlib.sha256(image_bytes).hexdigest()


@dataclass(frozen=True)
class LineFormerPrediction:
    figure_id: str
    image_key: str
    width: int
    height: int
    series: tuple[Series, ...]
    error: str | None = None

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> LineFormerPrediction:
        return cls(
            figure_id=record["figure_id"],
            image_key=record["image_key"],
            width=int(record["width"]),
            height=int(record["height"]),
            series=tuple(
                tuple((float(x), float(y)) for x, y in points)
                for points in record.get("series") or ()
            ),
            error=record.get("error"),
        )


class PrecomputedLineFormerModelRunner:
    def __init__(self, predictions: Iterable[LineFormerPrediction]):
        self._by_key = {p.image_key: p for p in predictions}

    def extract(self, task: ExtractionTask) -> list[Curve]:
        prediction = self._by_key[image_key(task.image_bytes)]
        if prediction.error is not None:
            raise RuntimeError(f"LineFormer worker failed: {prediction.error}")

        calibration = PixelCalibration(
            pixel_bbox=(0.0, 0.0, float(prediction.width), float(prediction.height)),
            x_range=task.x_range,
            y_range=task.y_range,
            x_scale=task.x_scale,
            y_scale=task.y_scale,
        )
        curves = []
        for i, points in enumerate(prediction.series):
            if not points:
                continue
            data = [calibration.to_data(px, py) for px, py in points]
            curves.append(
                Curve(
                    x_values=tuple(x for x, _ in data),
                    y_values=tuple(y for _, y in data),
                    series_label=f"series_{i}",
                    x_scale=task.x_scale,
                )
            )
        return curves
