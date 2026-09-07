"""Copy the LLM-subset run into the repository so it survives this machine.

`prepare_llm_subset.py` writes to a session scratch directory under /tmp, which
is local to one machine and one session. The owner is continuing on a different
machine tomorrow, and a benchmark run that cannot be reproduced or re-scored
somewhere else is not a result, so the inputs and the raw model answers are
archived here alongside the scores.

What is archived and why each piece is needed:

  tasks.json        the exact prompt inputs -- axis ranges and scales, the same
                    information ExtractionTask gives every other baseline
  _key.json         the fig_NN -> figure_id mapping, needed to re-score against
                    ground truth and deliberately withheld from the models
  predictions/      each model's raw answers, so the scores can be recomputed
                    without paying for the models again, and so a later change
                    to the metric can be applied to the same answers

The images are not copied: they already live in data/verified_pairs/ and
_key.json names each one, so duplicating them would add megabytes for nothing.
"""

from __future__ import annotations

import json
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parents[2]
WORK = pathlib.Path(
    "/tmp/claude-1000/-home-mato-repos-real-chart-bench/"
    "628beb76-383b-42e9-bf21-8d4188daf8dc/scratchpad/llm_eval"
)
DEST = REPO / "data/llm_subset_n10"

MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-haiku-4-5"]


def main() -> None:
    (DEST / "predictions").mkdir(parents=True, exist_ok=True)
    for name in ("tasks.json", "_key.json"):
        shutil.copy(WORK / name, DEST / name)
        print(f"  {name}")

    for m in MODELS:
        src = WORK / m / "predictions.json"
        if not src.exists():
            print(f"  {m}: 予測なし（スキップ）")
            continue
        raw = json.loads(src.read_text())
        (DEST / "predictions" / f"{m}.json").write_text(
            json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
        n_series = sum(len(v) for v in raw.values())
        n_pts = sum(min(len(s.get("x") or []), len(s.get("y") or []))
                    for v in raw.values() for s in v)
        print(f"  predictions/{m}.json  図{len(raw)} 系列{n_series} 点{n_pts}")

    print(f"\n→ {DEST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
