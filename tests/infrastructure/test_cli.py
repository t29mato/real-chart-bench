"""Tests for the CLI entry point.

LLMO requirement (docs/design/benchmark-architecture.md §4.4): every CLI command
must support a JSON output mode so that other agents/tools can consume results
without scraping human-readable text.
"""

import json

import pytest

from real_chart_bench.infrastructure.cli import main


def test_capabilities_json_output_is_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["capabilities", "--format", "json"])

    assert exit_code == 0
    out = capsys.readouterr().out
    payload = json.loads(out)
    assert payload["tool"] == "real-chart-bench"
    assert "status" in payload
    assert "commands" in payload


def test_capabilities_text_output_is_human_readable(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["capabilities", "--format", "text"])

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "real-chart-bench" in out
    # text mode must not just be dumped JSON
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


def test_default_format_is_json_for_machine_consumption(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["capabilities"])

    assert exit_code == 0
    out = capsys.readouterr().out
    json.loads(out)  # must not raise


def test_version_command_supports_json(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["version", "--format", "json"])

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert "version" in payload


def test_unknown_command_exits_nonzero() -> None:
    with pytest.raises(SystemExit):
        main(["not-a-real-command"])
