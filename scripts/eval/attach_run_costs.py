"""Attach run time and token counts to every result file (design 7.82 (3)):
results/*.json gain `run_cost`. "As of this run" numbers -- local ones
depend on the machine, cloud ones on the provider's load that day.

  Claude agents  data/run_costs/claude_agents.json (collect_run_costs.py).
                 A resumed agent reports again, cumulatively; its last report
                 is kept. Batches ran in parallel, so `seconds_total` is agent
                 time summed over batches, not wall-clock time.
  Codex          data/run_costs/codex.json, summed over the row's sessions.
  local VLMs,    the raw logs the result names (local_run.raw_output /
  chart models   chart2table.raw_predictions): per-figure seconds and, where
                 logged, prompt and generated tokens, over the scored figures.

A row with no record gets `run_cost: {"recorded": false}` -- never a guess.

Usage: python scripts/eval/attach_run_costs.py
"""

from __future__ import annotations

import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
COSTS = REPO / "data/run_costs"
CODEX_RESULTS = {
    ("calibrated", ("pilot10",)): "-v0-codex-pilot10",
    ("calibrated", ("pilot10", "part1", "part2")): "-v0-codex",
    ("noaxis", ("part1", "part2")): "-v0-codex-noaxis",
    ("pixcal", ("part1", "part2")): "-v0-codex-pixcal",
}


def _claude() -> dict[str, dict]:
    last: dict[tuple[str, str], dict] = {}
    for r in json.loads((COSTS / "claude_agents.json").read_text()):
        last[(r["result"], r["batch"])] = r  # file order: a later report wins
    out: dict[str, dict] = {}
    for (result, _), r in sorted(last.items()):
        c = out.setdefault(
            result,
            {
                "basis": "Claude Code subagent reports",
                "n_batches": 0,
                "seconds_total": 0.0,
                "subagent_tokens": 0,
                "tool_uses": 0,
            },
        )
        c["n_batches"] += 1
        c["seconds_total"] += r["seconds"]
        c["subagent_tokens"] += r["subagent_tokens"]
        c["tool_uses"] += r["tool_uses"]
    return out


def _codex() -> dict[str, dict]:
    rows = json.loads((COSTS / "codex.json").read_text())
    out = {}
    for (cond, batches), suffix in CODEX_RESULTS.items():
        for model in {r["model"] for r in rows}:
            sel = [
                r
                for r in rows
                if r["condition"] == cond and r["model"] == model and r["batch"] in batches
            ]
            if {r["batch"] for r in sel} != set(batches):
                continue
            usage: dict[str, int] = {}
            for r in sel:
                for k, v in r["usage"].items():
                    usage[k] = usage.get(k, 0) + v
            out[model + suffix] = {
                "basis": "Codex session logs (all sessions, incl. failed attempts)",
                "n_batches": len(batches),
                "n_sessions": len(sel),
                "seconds_total": round(sum(r["seconds"] or 0 for r in sel), 1),
                "tokens": usage,
            }
    return out


def _from_log(path: pathlib.Path, figures: set[str], fig_field: str) -> dict | None:
    if not path.exists():
        return None
    recs = [json.loads(line) for line in path.open() if line.strip()]
    recs = [r for r in recs if r.get(fig_field) in figures and r.get("seconds") is not None]
    if not recs:
        return None
    out = {
        "basis": f"per-figure log {path.relative_to(REPO)}",
        "n_figures": len(recs),
        "seconds_total": round(sum(r["seconds"] for r in recs), 1),
    }
    for k in ("prompt_tokens", "generation_tokens"):
        if all(isinstance(r.get(k), int) for r in recs):
            out[k] = sum(r[k] for r in recs)
    return out


def main() -> None:
    claude, codex = _claude(), _codex()
    n = 0
    for f in sorted((REPO / "results").glob("*.json")):
        r = json.loads(f.read_text())
        if "per_figure" not in r:
            continue
        figures = {p["figure_id"] for p in r["per_figure"]}
        cost = claude.get(f.stem) or codex.get(f.stem)
        if cost is None and r.get("local_run", {}).get("raw_output"):
            cost = _from_log(REPO / r["local_run"]["raw_output"], figures, "paper_figure")
            if cost and r["local_run"].get("batched_chunk"):
                cost["note"] = (
                    f"vLLM batches of {r['local_run']['batched_chunk']}: seconds are chunk "
                    "time / chunk size, comparable between rows on this machine only"
                )
            elif cost and r["local_run"].get("concurrent_with_other_models"):
                cost["note"] = "ran concurrently with other models: seconds are not comparable"
        if cost is None and r.get("chart2table", {}).get("raw_predictions"):
            cost = _from_log(REPO / r["chart2table"]["raw_predictions"], figures, "figure_id")
        if cost is None:
            cost = {"recorded": False}
        else:
            cost = {"recorded": True, **cost}
            if cost.get("seconds_total") is not None:
                cost["seconds_per_figure"] = round(cost["seconds_total"] / len(figures), 2)
        r["run_cost"] = cost
        f.write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n")
        n += cost["recorded"]
    print(f"run_cost recorded for {n} result files")


if __name__ == "__main__":
    main()
