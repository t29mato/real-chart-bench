import json

import pytest

from real_chart_bench.adapter.llm_model_runner import LlmModelRunner
from real_chart_bench.usecase.model_runner import ExtractionTask


class _FakeLlmClient:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.last_call = None

    def complete(self, *, model, image_bytes, prompt):
        self.last_call = {"model": model, "image_bytes": image_bytes, "prompt": prompt}
        return self._response_text


def _task():
    return ExtractionTask(image_bytes=b"fake-png-bytes", x_range=(0, 10), y_range=(0, 10))


def test_parses_well_formed_json_response_into_curves():
    response = json.dumps(
        {
            "series": [
                {"label": "a", "x": [1, 2, 3], "y": [4, 5, 6]},
                {"label": "b", "x": [1, 2], "y": [7, 8]},
            ]
        }
    )
    client = _FakeLlmClient(response)
    runner = LlmModelRunner(client=client, model_name="claude-test")

    curves = runner.extract(_task())

    assert len(curves) == 2
    assert curves[0].series_label == "a"
    assert curves[0].x_values == (1.0, 2.0, 3.0)


def test_prompt_includes_the_axis_calibration():
    client = _FakeLlmClient(json.dumps({"series": []}))
    runner = LlmModelRunner(client=client, model_name="claude-test")

    runner.extract(_task())

    assert "0" in client.last_call["prompt"]
    assert "10" in client.last_call["prompt"]
    assert client.last_call["model"] == "claude-test"
    assert client.last_call["image_bytes"] == b"fake-png-bytes"


def test_malformed_json_response_yields_no_curves_not_a_crash():
    client = _FakeLlmClient("not json at all")
    runner = LlmModelRunner(client=client, model_name="claude-test")

    assert runner.extract(_task()) == []


def test_response_missing_series_key_yields_no_curves():
    client = _FakeLlmClient(json.dumps({"unexpected": "shape"}))
    runner = LlmModelRunner(client=client, model_name="claude-test")

    assert runner.extract(_task()) == []


def test_series_with_mismatched_x_y_length_is_skipped_not_fatal():
    response = json.dumps(
        {
            "series": [
                {"label": "bad", "x": [1, 2, 3], "y": [4, 5]},
                {"label": "good", "x": [1, 2], "y": [4, 5]},
            ]
        }
    )
    client = _FakeLlmClient(response)
    runner = LlmModelRunner(client=client, model_name="claude-test")

    curves = runner.extract(_task())

    assert len(curves) == 1
    assert curves[0].series_label == "good"


@pytest.mark.parametrize("fence", ["```json\n{body}\n```", "```\n{body}\n```", "{body}"])
def test_code_fence_stripping_variants(fence):
    body = json.dumps({"series": [{"label": "a", "x": [1, 2], "y": [2, 3]}]})
    client = _FakeLlmClient(fence.format(body=body))
    runner = LlmModelRunner(client=client, model_name="claude-test")

    curves = runner.extract(_task())

    assert len(curves) == 1
