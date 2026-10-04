"""Copy the v3 run's answers into data/llm_run_v3/ as soon as they exist.

Same rules as archive_llm_run_v2.py (safe to re-run mid-run; never replaces
an archived file with one holding fewer figures; prints on-disk counts).

Usage: python scripts/eval/archive_llm_run_v3.py <work_dir>
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from archive_llm_run_v2 import REPO, archive  # noqa: E402

DEST = REPO / "data/llm_run_v3"


def main() -> None:
    archive(pathlib.Path(sys.argv[1]), DEST)


if __name__ == "__main__":
    main()
