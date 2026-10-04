"""Reads a local VLM run's raw output (data/local_vlm_run_v2/<model>/<condition>.jsonl,
design §7.69 / §7.73 (3)) into the answers the LLM scorer replays.

Each line is one (condition, figure) inference written by
scripts/eval/local_vlm/worker*.py: the raw text the model generated, the
series list parsed from it (``parsed``, None when parsing failed), and why it
failed (``parse_error`` / ``error``), plus timing and memory.

Only ``parsed`` becomes an answer. A failed figure is left out, so the scorer
counts it as a total miss -- the same rule as an LLM that did not answer
(scripts/eval/score_llm_predictions.py). The failures are returned by kind so
the results file can say how many there were.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LocalVlmRun:
    answers: dict[str, list]
    n_records: int
    # runtime exception during generation or parsing (no usable output)
    errors: tuple[str, ...]
    # generated text that did not parse into a series list
    parse_failures: tuple[str, ...]
    # generation stopped at max_tokens (usually also a parse failure)
    truncated: tuple[str, ...]
    # parsed, but the parser noted something (e.g. accepted a mismatched key)
    accepted_with_warning: tuple[str, ...]
    # largest per-figure peak (mlx, this process only); None if never recorded
    peak_memory_gb_max: float | None = None


def load_local_vlm_run(path: Path) -> LocalVlmRun:
    answers: dict[str, list] = {}
    seen: set[str] = set()
    errors, parse_failures, truncated, warned = [], [], [], []
    peaks: list[float] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        fig = rec["fig"]
        if fig in seen:
            raise ValueError(f"{path}: {fig} is recorded more than once")
        seen.add(fig)
        if rec.get("peak_memory_gb") is not None:
            peaks.append(float(rec["peak_memory_gb"]))
        if rec.get("truncated"):
            truncated.append(fig)
        if rec.get("error"):
            errors.append(fig)
        elif rec.get("parsed") is None:
            parse_failures.append(fig)
        else:
            answers[fig] = rec["parsed"]
            if rec.get("parse_error"):
                warned.append(fig)
    return LocalVlmRun(
        answers=answers,
        n_records=len(seen),
        errors=tuple(errors),
        parse_failures=tuple(parse_failures),
        truncated=tuple(truncated),
        accepted_with_warning=tuple(warned),
        peak_memory_gb_max=max(peaks) if peaks else None,
    )
