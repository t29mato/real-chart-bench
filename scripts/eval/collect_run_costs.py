"""Collect run time and token counts (design 7.82 (3)) into data/run_costs/.

Sources outside the repository are read once and their facts saved, so the
numbers stay reproducible after the logs are gone:

  claude_agents.json -- each Claude Code subagent's completion notice in this
      project's Claude Code transcripts (~/.claude/projects/<project>/*.jsonl):
      subagent_tokens, tool_uses, duration_ms, as Claude Code reports them.
      Only runs whose agents were started from this machine have one; the v3
      run (the owner's Mac) and the 2026-09 runs are "not recorded".
  codex.json -- each Codex session (data/llm_run_codex/ event logs name it):
      the final cumulative token usage and the active time summed over its
      turns (task_started -> task_complete), from ~/.codex/sessions (and the
      local-model runs' separate CODEX_HOME). A turn
      that died on a server error still counts its tokens and time.

Local VLMs and chart-to-table models are not collected here: their per-figure
seconds and tokens are in their raw logs in the repository and are read when
results are written (attach_run_costs.py).

Usage: python scripts/eval/collect_run_costs.py
"""

from __future__ import annotations

import json
import pathlib
import re
from datetime import datetime

REPO = pathlib.Path(__file__).resolve().parents[2]
OUT = REPO / "data/run_costs"
TRANSCRIPTS = pathlib.Path.home() / ".claude/projects" / (
    "-" + str(REPO).strip("/").replace("/", "-")
)
# the user's Codex home (ChatGPT runs) and the separate one the local-model
# runs use (run_codex_local_batch.sh)
CODEX_SESSIONS = (
    pathlib.Path.home() / ".codex/sessions",
    pathlib.Path.home() / ".cache/real-chart-bench/codex-local/home/sessions",
)
FULL = {
    "opus": "claude-opus-5-5",
    "sonnet": "claude-sonnet-5-5",
    "fable": "claude-fable-5-1",
    "haiku": "claude-haiku-4-5",
}
# agent description -> the result row it answered for
RUNS = {
    "calibrated": "{m}-v0-r2",
    "noaxis": "{m}-v0-r2-noaxis",
    "pixcal": "{m}-v0-pixcal",
    "synth": "{m}-plotqa-dot-line-noaxis",
}
NOTICE = re.compile(
    r'<summary>Agent \\?"(?P<desc>[^"\\]+)\\?" finished</summary>.*?'
    r"<subagent_tokens>(?P<tok>\d+)</subagent_tokens>.*?"
    r"<tool_uses>(?P<tools>\d+)</tool_uses>.*?<duration_ms>(?P<ms>\d+)</duration_ms>",
    re.S,
)
DESC = re.compile(r"^(opus|sonnet|fable|haiku) (calibrated|noaxis|pixcal|synth) (part\d)$")


def claude_agents() -> list[dict]:
    seen, rows = set(), []
    for f in sorted(TRANSCRIPTS.glob("*.jsonl")):
        for line in f.open():
            if "subagent_tokens" not in line:
                continue
            for m in NOTICE.finditer(line):
                d = DESC.match(m["desc"])
                if not d:
                    continue
                rec = (m["desc"], int(m["tok"]), int(m["tools"]), int(m["ms"]))
                if rec in seen:  # the same notice is repeated in later turns
                    continue
                seen.add(rec)
                model, run, batch = d.groups()
                rows.append(
                    {
                        "result": RUNS[run].format(m=FULL[model]),
                        "batch": batch,
                        "subagent_tokens": rec[1],
                        "tool_uses": rec[2],
                        "seconds": round(rec[3] / 1000, 1),
                        "transcript": f.name,
                    }
                )
    return rows


def _ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _session(thread_id: str) -> dict | None:
    """Active time (task_started -> task_complete, summed over turns) and the
    session's final cumulative token usage, from Codex's own session log."""
    files = [f for root in CODEX_SESSIONS for f in root.rglob(f"*{thread_id}*.jsonl")]
    if not files:
        return None
    total, start, usage = 0.0, None, {}
    for line in files[0].open():
        d = json.loads(line)
        p = d.get("payload", {})
        kind = p.get("type")
        if kind == "task_started":
            start = _ts(d["timestamp"])
        elif kind in ("task_complete", "turn_aborted") and start:
            total += (_ts(d["timestamp"]) - start).total_seconds()
            start = None
        elif kind == "token_count" and (p.get("info") or {}).get("total_token_usage"):
            usage = p["info"]["total_token_usage"]
    return {"seconds": round(total, 1), "usage": usage}


def codex() -> list[dict]:
    """One row per Codex session (a resumed batch keeps its session, so its
    logs share one row)."""
    by_thread: dict[str, dict] = {}
    for ev in sorted((REPO / "data/llm_run_codex").glob("*/*/*.events.jsonl")):
        cond, model = ev.parts[-3:-1]
        events = [json.loads(line) for line in ev.open() if line.strip()]
        thread = next((e["thread_id"] for e in events if e.get("type") == "thread.started"), None)
        row = by_thread.setdefault(
            thread,
            {
                "condition": cond,
                "model": model,
                "batch": ev.name.split(".")[0],
                "session": thread,
                "logs": [],
                "completed": False,
            },
        )
        row["logs"].append(ev.name)
        row["completed"] |= any(e.get("type") == "turn.completed" for e in events)
    for thread, row in by_thread.items():
        row.update(_session(thread) or {"seconds": None, "usage": {}})
    return list(by_thread.values())


def main() -> None:
    OUT.mkdir(exist_ok=True)
    for name, rows in (("claude_agents", claude_agents()), ("codex", codex())):
        (OUT / f"{name}.json").write_text(json.dumps(rows, indent=2) + "\n")
        print(name, len(rows))


if __name__ == "__main__":
    main()
