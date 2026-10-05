"""Copy Codex batches' answers and logs into data/llm_run_codex/ (design 7.81):
<condition>/<model>/<batch>.{predictions.json,axis_notes.json,events.jsonl,
last_message.txt}. Safe to re-run; never replaces an archived answer with one
covering fewer figures; refuses to archive a log that names the account.

Usage: python scripts/eval/archive_llm_run_codex.py <work_dir>
"""

from __future__ import annotations

import json
import pathlib
import re
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DEST = REPO / "data/llm_run_codex"
EMAIL = re.compile(rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")


def _n(path: pathlib.Path) -> int | None:
    try:
        return len(json.loads(path.read_text()))
    except (OSError, ValueError):
        return None


def main() -> None:
    work = pathlib.Path(sys.argv[1])
    for d in sorted(p for p in work.glob("*/*/*") if p.is_dir()):
        cond, model, batch = d.parts[-3:]
        target = DEST / cond / model
        n_tasks = len(json.loads((d / "tasks.json").read_text()))
        files = {
            f"{batch}.predictions.json": d / "predictions.json",
            f"{batch}.axis_notes.json": d / "axis_notes.json",
            f"{batch}.events.jsonl": d.parent / f"{batch}.events.jsonl",
            f"{batch}.last_message.txt": d.parent / f"{batch}.last_message.txt",
        }
        # logs of a failed first attempt or of a resumed session
        for extra in d.parent.glob(f"{batch}.*.events.jsonl"):
            files[extra.name] = extra
        for name, src in files.items():
            if not src.exists():
                continue
            if EMAIL.search(src.read_bytes()):
                print(f"{cond}/{model}/{name}: contains an e-mail address -- not archived")
                continue
            dst = target / name
            if name.endswith(".json"):
                new, old = _n(src), _n(dst) if dst.exists() else None
                if new is None:
                    print(f"{cond}/{model}/{name}: unreadable (mid-write?) -- skipped")
                    continue
                if old and old > new:
                    print(f"{cond}/{model}/{name}: {new} < archived {old} -- kept archive")
                    continue
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)
            if name.endswith("predictions.json"):
                print(f"{cond:<10} {model:<12} {batch:<8} {_n(dst)}/{n_tasks} figs")


if __name__ == "__main__":
    main()
