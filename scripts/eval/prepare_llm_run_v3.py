"""Prepare the 2026-10-04 LLM run v3 (design 7.73 (2)).

v3 is v2 with one change to the prompt (scripts/eval/llm_run_v3_prompt.md):
extract measured data points (markers) only -- no fit, trend, guide or theory
lines. Figures, conditions, models, batch split and contamination guards are
v2's, so this reuses prepare_llm_run_v2.prepare(). Only the seed differs, so
v3's fig_NNN.png names do not map onto v2's: an agent that has seen v2's
ordering cannot carry it over.

Writes <work>/<condition>/<model>/part<k>/{images/,tasks.json} and
<work>/_key.json, and copies the key and task lists into data/llm_run_v3/.

Usage: python scripts/eval/prepare_llm_run_v3.py <work_dir>
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from prepare_llm_run_v2 import REPO, prepare  # noqa: E402

SEED = 20261004
ARCHIVE = REPO / "data/llm_run_v3"


def main() -> None:
    prepare(pathlib.Path(sys.argv[1]), seed=SEED, archive=ARCHIVE)


if __name__ == "__main__":
    main()
