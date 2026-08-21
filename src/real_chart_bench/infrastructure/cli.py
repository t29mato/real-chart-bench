"""CLI entry point.

LLMO design rule (docs/design/benchmark-architecture.md §4.4): every command must
support ``--format json`` and defaults to JSON output, since the primary consumers
of this tool include other agents/scripts, not only humans. ``--format text``
opts into a human-readable rendering.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from real_chart_bench import __version__

COMMANDS = ("capabilities", "version")


def _capabilities_payload() -> dict:
    return {
        "tool": "real-chart-bench",
        "version": __version__,
        "purpose": (
            "Benchmark how accurately models (LLMs, dedicated chart models, "
            "classic tools) recover XY data from real research-paper figures."
        ),
        "status": "pre-alpha: v0 dataset + evaluation harness + leaderboard are live",
        "commands": list(COMMANDS),
        "design_doc": "docs/design/benchmark-architecture.md",
        "readme": "README.md",
        "leaderboard_url": "https://t29mato.github.io/real-chart-bench/",
        "ground_truth_manifest": "data/manifest/v0/",
        "verified_pairs_registry": "data/verified_pairs/registry.json",
    }


def _version_payload() -> dict:
    return {"tool": "real-chart-bench", "version": __version__}


def _render(payload: dict, fmt: str) -> str:
    if fmt == "json":
        return json.dumps(payload, ensure_ascii=False)
    # text mode: simple "key: value" lines, not a JSON dump
    lines = []
    for key, value in payload.items():
        if isinstance(value, list):
            value = ", ".join(value)
        lines.append(f"{key}: {value}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="real-chart-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in COMMANDS:
        sub = subparsers.add_parser(name)
        sub.add_argument(
            "--format",
            choices=("json", "text"),
            default="json",
            help="Output format (default: json, for machine consumption).",
        )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    payload = _capabilities_payload() if args.command == "capabilities" else _version_payload()
    print(_render(payload, args.format))
    return 0


def run() -> None:
    """Console-script entry point (see [project.scripts] in pyproject.toml)."""
    sys.exit(main())


if __name__ == "__main__":
    run()
