"""Reads an archived agent run's answers (data/llm_run_v*/<condition>/<model>/,
design §7.66 / §7.73 (2)) into the answers the LLM scorer replays.

Each batch an agent answered is one ``<part>.predictions.json`` (task id ->
series list). The parts to read are named explicitly rather than globbed: a
model dir can hold other prediction files that are not part of the official
run -- the v3 Sonnet dirs keep ``part{1,2}_nopillow`` (first attempts made
without image tools, rerun since) -- and a ``part*`` glob would load them and,
sorted after ``part2``, overwrite the official answers.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

# The two batches every v2/v3 model was launched with
OFFICIAL_PARTS: tuple[str, ...] = ("part1", "part2")


def load_agent_run_parts(model_dir: Path, parts: Sequence[str]) -> dict[str, list]:
    """Merge exactly the named parts. A missing part, or a figure answered in
    two parts, is an error: either means the wrong files are being scored."""
    if not parts:
        raise ValueError("no parts named")
    answers: dict[str, list] = {}
    for part in parts:
        path = model_dir / f"{part}.predictions.json"
        if not path.exists():
            raise FileNotFoundError(f"{path}: {part} is missing")
        for fig, series in json.loads(path.read_text()).items():
            if fig in answers:
                raise ValueError(f"{model_dir}: {fig} is answered in more than one part")
            answers[fig] = series
    return answers
