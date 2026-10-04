"""Copy the v2 run's answers into the repository as soon as they exist.

Safe to run repeatedly mid-run (a lesson from 2026-09: a /tmp wipe lost one
run, and a re-archive once copied a regressed file over a finished one).
Per (condition, model, batch) it archives predictions.json, plus
axis_notes.json for noaxis, into data/llm_run_v2/<condition>/<model>/, and
refuses to replace an archived file with one holding fewer figures.

It also prints what is on disk -- figures, series, points -- because an
agent's own summary of its file has been wrong before (RUN_NOTES of the
2026-09-12 run). The archived file is the record, not the agent's report.

Usage: python scripts/eval/archive_llm_run_v2.py <work_dir>
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DEST = REPO / "data/llm_run_v2"


def _count(path: pathlib.Path) -> tuple[int, int, int] | None:
    try:
        data = json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(data, dict):
        return None
    series = [s for v in data.values() if isinstance(v, list) for s in v if isinstance(s, dict)]
    points = sum(len(s.get("x") or []) for s in series)
    return len(data), len(series), points


def archive(work: pathlib.Path, dest: pathlib.Path) -> None:
    """Archive every part dir under `work` into `dest`. Shared with v3."""
    for part_dir in sorted(work.glob("*/*/part*")):
        cond, model, part = part_dir.parts[-3:]
        n_tasks = len(json.loads((part_dir / "tasks.json").read_text()))
        for name in ("predictions.json", "axis_notes.json"):
            src = part_dir / name
            if not src.exists():
                continue
            counts = _count(src)
            label = f"{cond:<10} {model:<18} {part} {name:<17}"
            if counts is None:
                print(f"{label} unreadable (mid-write?) -- skipped")
                continue
            target = dest / cond / model / f"{part}.{name}"
            old = _count(target) if target.exists() else None
            if old and old[0] > counts[0]:
                print(f"{label} {counts[0]}/{n_tasks} < archived {old[0]} -- kept archive")
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, target)
            print(f"{label} {counts[0]:>3}/{n_tasks} figs {counts[1]:>4} series {counts[2]:>6} pts")


def main() -> None:
    archive(pathlib.Path(sys.argv[1]), DEST)


if __name__ == "__main__":
    main()
