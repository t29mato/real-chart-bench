"""LLM-backed implementation of ModelRunnerPort (design §7.16).

**Scaffold only — not wired to a real vendor SDK.** LlmClientPort has no
default/real implementation in this repo; extract() cannot make a network
call unless the caller explicitly injects a real client, which requires
adding vendor SDK code that doesn't exist yet. This mirrors the guard
pattern already used for the HF Hub upload step
(scripts/publish/prepare_hf_dataset.py): a structural block against
accidental real execution, not just a comment saying "don't run this".

Real execution (Claude/GPT/Gemini) is gated on オーナー approval — see
design §7.16 cost estimate.
"""

from __future__ import annotations

import json
import re

from real_chart_bench.domain.curve import Curve
from real_chart_bench.usecase.llm_client import LlmClientPort
from real_chart_bench.usecase.model_runner import ExtractionTask

_PROMPT_TEMPLATE = """\
You are extracting data series from a scientific chart image.

The chart's plot area spans:
  x-axis: {x_lo} to {x_hi} ({x_scale} scale)
  y-axis: {y_lo} to {y_hi}

Return ONLY a JSON object of this exact shape, no other text:
{{"series": [{{"label": "<series name>", "x": [<numbers>], "y": [<numbers>]}}]}}

List every distinct data series (by color/marker/legend) you can see, with
its data points in the axis units given above.
"""

_CODE_FENCE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL)


def _build_prompt(task: ExtractionTask) -> str:
    return _PROMPT_TEMPLATE.format(
        x_lo=task.x_range[0],
        x_hi=task.x_range[1],
        x_scale=task.x_scale.value,
        y_lo=task.y_range[0],
        y_hi=task.y_range[1],
    )


def _strip_code_fence(text: str) -> str:
    match = _CODE_FENCE.match(text.strip())
    return match.group(1) if match else text


class LlmModelRunner:
    def __init__(self, *, client: LlmClientPort, model_name: str) -> None:
        self._client = client
        self._model_name = model_name

    def extract(self, task: ExtractionTask) -> list[Curve]:
        prompt = _build_prompt(task)
        raw_response = self._client.complete(
            model=self._model_name, image_bytes=task.image_bytes, prompt=prompt
        )
        return self._parse_response(raw_response, task)

    @staticmethod
    def _parse_response(raw_response: str, task: ExtractionTask) -> list[Curve]:
        try:
            payload = json.loads(_strip_code_fence(raw_response))
        except json.JSONDecodeError:
            return []

        series = payload.get("series")
        if not isinstance(series, list):
            return []

        curves = []
        for entry in series:
            try:
                x_values = tuple(float(v) for v in entry["x"])
                y_values = tuple(float(v) for v in entry["y"])
                if len(x_values) != len(y_values) or not x_values:
                    continue
                curves.append(
                    Curve(
                        x_values=x_values,
                        y_values=y_values,
                        x_scale=task.x_scale,
                        series_label=str(entry.get("label", "")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue
        return curves
