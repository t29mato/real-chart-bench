"""Archive the no-axis run's raw answers as soon as each model finishes.

The first attempt at this run was lost: three of the four models had finished
and written predictions.json, then the session restarted and the /tmp scratch
directory went with it. The calibrated run survived only because it had been
archived; this one had not.

So this is written to be run repeatedly, mid-run, archiving whatever exists so
far rather than waiting for all four. Losing a completed model's answers costs
another full agent run, and there is no reason to hold them in a temp directory
a moment longer than necessary.

Also copies axis_notes.json, which the calibrated condition has no equivalent
of: it records what each model read the axis to be, so an extraction error can
be attributed to the calibration or to the marker reading rather than guessed at.
"""

from __future__ import annotations

import json
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parents[2]
WORK = pathlib.Path(
    "/tmp/claude-1000/-home-mato-repos-real-chart-bench/"
    "628beb76-383b-42e9-bf21-8d4188daf8dc/scratchpad/llm_eval_noaxis"
)
DEST = REPO / "data/llm_subset_n10_noaxis"
MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-haiku-4-5"]


def main() -> None:
    (DEST / "predictions").mkdir(parents=True, exist_ok=True)
    (DEST / "axis_notes").mkdir(parents=True, exist_ok=True)

    src_tasks = WORK / MODELS[0] / "tasks.json"
    if src_tasks.exists():
        shutil.copy(src_tasks, DEST / "tasks.json")

    for m in MODELS:
        for kind, sub in (("predictions", "predictions"), ("axis_notes", "axis_notes")):
            src = WORK / m / f"{kind}.json"
            if not src.exists():
                continue
            try:
                raw = json.loads(src.read_text())
            except json.JSONDecodeError as exc:
                print(f"  {m}/{kind}: JSONとして読めない ({exc}) — スキップ")
                continue
            (DEST / sub / f"{m}.json").write_text(
                json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
            if kind == "predictions":
                ns = sum(len(v) for v in raw.values())
                np_ = sum(min(len(s.get("x") or []), len(s.get("y") or []))
                          for v in raw.values() for s in v)
                print(f"  {m}: 図{len(raw)} 系列{ns} 点{np_}")

    have = sorted(p.stem for p in (DEST / "predictions").glob("*.json"))
    print(f"\n退避済み {len(have)}/{len(MODELS)} モデル: {', '.join(have) or '(なし)'}")
    print(f"→ {DEST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
