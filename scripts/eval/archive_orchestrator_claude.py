"""Copy the answers of 方式D's Claude-orchestrator runs into the repo:
<work>/<condition>/<model>/<part>/predictions.json ->
data/llm_run_orchestrator/<noaxis|pixpts_px>/<model>/<part>.predictions.json,
plus the agent's saved tool results' count per figure as a usage record
(<part>.tool_calls.json: task id -> number of saved results and overlays).
Never replaces an archived answer with one covering fewer figures.

  .venv/bin/python scripts/eval/archive_orchestrator_claude.py <work_dir> [--v2 | --v3]

Score: score_llm_predictions.py orch-claude-noaxis | orch-claude-pixcal
(--v2: data/llm_run_orchestrator_v2/, scored as orch2-claude-noaxis | -pixcal;
the usage record also counts verify calls and redos from _verify_log.json;
--v3: data/llm_run_orchestrator_v3/, scored as orch3-claude-noaxis | -pixcal)
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
DEST = REPO / "data/llm_run_orchestrator"
DEST_V2 = REPO / "data/llm_run_orchestrator_v2"  # --v2: the 検証とやり直し run
DEST_V3 = REPO / "data/llm_run_orchestrator_v3"  # --v3: verify v3 + blob_extract
V2 = {"noaxis": "noaxis", "pixcal": "pixpts_px"}


def main() -> None:
    work = pathlib.Path(sys.argv[1])
    dest = (DEST_V3 if "--v3" in sys.argv[2:] else DEST_V2 if "--v2" in sys.argv[2:]
            else DEST)
    for d in sorted(p for p in work.glob("*/*/part*") if p.is_dir()):
        cond, model, part = d.parts[-3:]
        src = d / "predictions.json"
        if cond not in V2 or not src.exists():
            print(f"{d}: no predictions -- skipped")
            continue
        target = dest / V2[cond] / model
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
        usage = {}
        for f in sorted((d / "work").glob("*")):
            if not f.is_dir():
                continue
            usage[f.name] = {"results": len(list(f.glob("*.json"))),
                             "overlays": len(list(f.glob("*.png")))}
            log = f / "_verify_log.json"
            if log.exists():
                vlog = json.loads(log.read_text())
                usage[f.name] |= {"verify": len(vlog),
                                  "verify_accepted": sum(v["accept"] for v in vlog),
                                  "verify_log": vlog}
        (target / f"{part}.tool_calls.json").write_text(json.dumps(usage, indent=1) + "\n")
        print(f"{dst}: {new} figures")


if __name__ == "__main__":
    main()
