"""Copy the answers of 方式D's Claude-orchestrator runs into the repo:
<work>/<condition>/<model>/<part>/predictions.json ->
data/llm_run_orchestrator/<noaxis|pixpts_px>/<model>/<part>.predictions.json,
plus the agent's saved tool results' count per figure as a usage record
(<part>.tool_calls.json: task id -> number of saved results and overlays).
Never replaces an archived answer with one covering fewer figures.

  .venv/bin/python scripts/eval/archive_orchestrator_claude.py <work_dir>

Score: score_llm_predictions.py orch-claude-noaxis | orch-claude-pixcal
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DEST = REPO / "data/llm_run_orchestrator"
V2 = {"noaxis": "noaxis", "pixcal": "pixpts_px"}


def main() -> None:
    work = pathlib.Path(sys.argv[1])
    for d in sorted(p for p in work.glob("*/*/part*") if p.is_dir()):
        cond, model, part = d.parts[-3:]
        src = d / "predictions.json"
        if cond not in V2 or not src.exists():
            print(f"{d}: no predictions -- skipped")
            continue
        target = DEST / V2[cond] / model
        dst = target / f"{part}.predictions.json"
        try:
            new = len(json.loads(src.read_text()))
        except ValueError:
            print(f"{d}: unreadable predictions (mid-write?) -- skipped")
            continue
        if dst.exists() and len(json.loads(dst.read_text())) > new:
            print(f"{dst}: archived copy covers more figures -- kept")
            continue
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        usage = {f.name: {"results": len(list(f.glob("*.json"))),
                          "overlays": len(list(f.glob("*.png")))}
                 for f in sorted((d / "work").glob("*")) if f.is_dir()}
        (target / f"{part}.tool_calls.json").write_text(json.dumps(usage, indent=1) + "\n")
        print(f"{dst}: {new} figures")


if __name__ == "__main__":
    main()
