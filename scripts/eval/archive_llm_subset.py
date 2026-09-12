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

import argparse
import json
import pathlib
import shutil

REPO = pathlib.Path(__file__).resolve().parents[2]
SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-mato-repos-real-chart-bench/"
    "628beb76-383b-42e9-bf21-8d4188daf8dc/scratchpad"
)

# Named runs, so the same archiving discipline covers the n=10 subset and the
# remaining-101 run. Run this while models are still working, not only at the
# end: the first noaxis run was lost entirely to a /tmp wipe, and a partially
# archived run is worth far more than a complete one that no longer exists.
RUNS = {
    "n10": {"work": SCRATCH / "llm_eval", "dest": REPO / "data/llm_subset_n10",
            "copy_meta": True},
    "rest": {"work": SCRATCH / "llm_eval_rest", "dest": REPO / "data/llm_subset_rest",
             # tasks.json/_key.json for this run were written straight into the
             # repository by prepare_llm_subset_rest.py, so there is nothing to
             # copy back and copying would risk overwriting them with scratch.
             "copy_meta": False},
}

MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-haiku-4-5"]


def main() -> None:
    ap = argparse.ArgumentParser(description="Archive an LLM-subset run into the repository.")
    ap.add_argument("run", nargs="?", default="n10", choices=sorted(RUNS))
    ap.add_argument("--force", action="store_true",
                    help="shorter runs may overwrite longer archived ones")
    args = ap.parse_args()
    run = RUNS[args.run]
    WORK, DEST = run["work"], run["dest"]

    (DEST / "predictions").mkdir(parents=True, exist_ok=True)
    if run["copy_meta"]:
        for name in ("tasks.json", "_key.json"):
            shutil.copy(WORK / name, DEST / name)
            print(f"  {name}")

    for m in MODELS:
        src = WORK / m / "predictions.json"
        if not src.exists():
            print(f"  {m}: 予測なし（スキップ）")
            continue
        raw = json.loads(src.read_text())
        dest_path = DEST / "predictions" / f"{m}.json"

        # Never let a shorter run overwrite a longer archived one. This is not
        # hypothetical: Haiku 4.5 reported 101/101 figures, was archived, and
        # then its scratch predictions.json reappeared holding only the first
        # 10 -- the agent had started over after its completion notification.
        # Without this guard the re-archive silently destroyed the finished
        # run, and only the git copy saved it.
        if dest_path.exists() and not args.force:
            have = json.loads(dest_path.read_text())
            if len(have) > len(raw):
                print(
                    f"  {m}: 退避済み{len(have)}図 > 今回{len(raw)}図 → "
                    f"上書きせずスキップ（--force で強制）"
                )
                continue

        dest_path.write_text(json.dumps(raw, ensure_ascii=False, indent=2) + "\n")
        n_series = sum(len(v) for v in raw.values())
        n_pts = sum(min(len(s.get("x") or []), len(s.get("y") or []))
                    for v in raw.values() for s in v)
        print(f"  predictions/{m}.json  図{len(raw)} 系列{n_series} 点{n_pts}")

    print(f"\n→ {DEST.relative_to(REPO)}")


if __name__ == "__main__":
    main()
