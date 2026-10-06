#!/usr/bin/env python3
"""The tool command of a sealed 方式D directory (copied there as `tool` by
scripts/eval/prepare_orchestrator_claude.py). The orchestrating agent runs
only this; it never writes image-processing code of its own.

  ./tool <tool> <task id> '<params JSON>' [--save NAME]
  ./tool answer <task id> '{"series": [{"from": "r2", "index": -1, "label": ""}],
                            "calibration": "r1"}'
  ./tool verify <task id> '{"series": [...as for answer...], "calibration": "r1"}'
  ./tool check

verify checks the series on the image and gives a verdict; every verify is
logged (work/<task id>/_verify_log.json) and `answer` only takes series that
verify accepted -- or, once the retries are used up (domain/orchestration.
MAX_RETRIES), the best-scoring verified attempt.

Results are saved as work/<task id>/<NAME>.json; a parameter naming an
earlier result uses that NAME (e.g. "r2"). render_overlay writes
work/<task id>/<NAME or overlay_k>.png -- look at it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DIR / "tools"))

from real_chart_bench.adapter.orchestrator_tools import TOOLS, ToolBox, ToolError  # noqa: E402
from real_chart_bench.domain.digitizer_tools import (  # noqa: E402
    calibration_from_person,
    series_to_values,
)
from real_chart_bench.domain.orchestration import (  # noqa: E402
    answer_key,
    assemble_final,
    may_answer,
    parse_action,
    retry_decision,
)

CONFIG = json.loads((DIR / "tools/config.json").read_text())
TASKS = {t["id"]: t for t in json.loads((DIR / "tasks.json").read_text())}
PIXELS = CONFIG["condition"] == "pixcal"


def fail(msg: str) -> None:
    print(json.dumps({"error": msg}))
    sys.exit(2)


def _next_step(dec: dict, vlog: list) -> str:
    if dec["action"] == "accept":
        return "accepted: answer exactly these series (./tool answer)"
    if dec["action"] == "redo":
        return ("redo: fix what the reasons say with a different tool, setting or mask, then "
                f"verify again ({dec['retries_left'] + 1} retries left)")
    best = vlog[dec["pick"]]
    return (f"retries used up: answer the best-scoring attempt, {best['name']} "
            f"(score {best['score']})")


def _brief_verify(out: dict) -> dict:
    """verify's printout: the verdict first, lists shortened."""
    o = dict(out)
    for k in ("unexplained", "lookalikes"):
        if isinstance(o.get(k), dict):
            o[k] = {kk: (vv[:10] if isinstance(vv, list) else vv) for kk, vv in o[k].items()}
    if isinstance(o.get("detector"), dict):
        o["detector"] = {**o["detector"], "uncovered": o["detector"]["uncovered"][:10]}
    return o


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return
    if args[0] == "check":
        preds = json.loads((DIR / "predictions.json").read_text()) \
            if (DIR / "predictions.json").exists() else {}
        missing = sorted(set(TASKS) - set(preds))
        print(json.dumps({"tasks": len(TASKS), "answered": len(preds),
                          "series": sum(len(v) for v in preds.values()),
                          "points": sum(len(s["x"]) for v in preds.values() for s in v),
                          "missing": missing}))
        return
    save = None
    if "--save" in args:
        i = args.index("--save")
        save = args[i + 1]
        args = args[:i] + args[i + 2:]
    if len(args) not in (2, 3):
        fail("usage: ./tool <tool> <task id> '<params JSON>' [--save NAME]")
    tool, fig = args[0], args[1]
    if fig not in TASKS:
        fail(f"unknown task id {fig}")
    try:
        params = json.loads(args[2]) if len(args) == 3 else {}
    except json.JSONDecodeError as exc:
        fail(f"params are not JSON: {exc}")
    work = DIR / "work" / fig
    work.mkdir(parents=True, exist_ok=True)

    def resolve(ref: str) -> dict:
        p = work / f"{ref}.json"
        if not p.exists():
            raise ToolError(f"no saved result {ref!r} for {fig} (save results with --save)")
        return json.loads(p.read_text())

    log_path = work / "_verify_log.json"
    vlog = json.loads(log_path.read_text()) if log_path.exists() else []
    try:
        if tool == "answer":
            act = parse_action(json.dumps({"action": "final", **params}), list(TOOLS))
            key = answer_key(act.series, "" if PIXELS else act.calibration)
            ok, why = may_answer(vlog, key)
            if not ok:
                fail(f"not answered: {why}")
            results = {s["from"]: resolve(s["from"]) for s in act.series}
            if act.calibration and not PIXELS:
                results[act.calibration] = resolve(act.calibration)
            series, cal = assemble_final(act, results, need_calibration=not PIXELS)
            if cal is not None:
                series = series_to_values(series, cal)
            pred_path = DIR / "predictions.json"
            preds = json.loads(pred_path.read_text()) if pred_path.exists() else {}
            preds[fig] = series
            pred_path.write_text(json.dumps(preds) + "\n")
            out = {"answered": fig, "n_series": len(series),
                   "n_points": sum(len(s["x"]) for s in series),
                   "figures_answered": len(preds), "of": len(TASKS)}
        elif tool in TOOLS:
            task = TASKS[fig]
            given = calibration_from_person(task) if PIXELS else None
            if tool == "render_overlay" and "out" not in params:
                n = len(list(work.glob("*.png"))) + 1
                params["out"] = str(work / f"{save or f'overlay_{n}'}.png")
            elif tool == "render_overlay":
                params["out"] = str(work / Path(params["out"]).name)
            tb = ToolBox(DIR / "images" / fig, dets_dir=DIR / "tools/dets",
                         given_calibration=given, allow_auto_calibration=not PIXELS)
            out = tb.run(tool, params, resolve)
            if out.get("kind") == "overlay":
                out["path"] = str(Path(out["path"]).relative_to(DIR))
            if tool == "verify":
                key = answer_key(params.get("series") or [],
                                 "" if PIXELS else params.get("calibration", ""))
                vlog.append({"key": key, "accept": out["verdict"]["accept"],
                             "score": out["verdict"]["score"],
                             "name": save or f"verify_{len(vlog) + 1}"})
                log_path.write_text(json.dumps(vlog) + "\n")
                save = save or f"verify_{len(vlog)}"
                dec = retry_decision(vlog)
                out = {"verdict": out["verdict"], "next": _next_step(dec, vlog),
                       **{k: v for k, v in out.items() if k != "verdict"}}
        else:
            fail(f"unknown tool {tool!r}; tools: {', '.join(TOOLS)}, answer, check")
    except (ToolError, ValueError) as exc:
        fail(str(exc))
    if save:
        (work / f"{save}.json").write_text(json.dumps(out) + "\n")
        out = {"saved_as": save, **out}
    if tool == "verify":
        out = _brief_verify(out)
    text = json.dumps(out)
    if len(text) > 6000 and out.get("kind") in ("points", "values"):
        # long point lists: the counts and the start, the full result is saved
        brief = {k: v for k, v in out.items() if k != "series"}
        brief["series"] = [{"label": s.get("label"), "n": len(s["x"]),
                            "x": s["x"][:8], "y": s["y"][:8]} for s in out["series"]]
        brief["note"] = "point lists shortened in this printout" + (
            "; the full result is saved" if save else "; use --save to keep the full result")
        text = json.dumps(brief)
    print(text)


if __name__ == "__main__":
    main()
