"""Re-scores every result from saved answers -- nothing is re-run.

Run after anything that changes what is scored: the registry (a figure
excluded), the ground truth, or the metric. Order matters: the CV baselines
write naive-cv-v0.json, whose dataset_version the later scorers read, and the
tick-calibrated subsets are cut from every other main-table row.

  1. CV baselines (run_baselines.py)
  2. LineFormer from its saved pixel output (run_lineformer.py --score-only)
  3. every LLM / local-VLM condition in score_llm_predictions.CONDITIONS
     whose answers exist
  4. chart-to-table models on the real figures (score_chart2table.py)
  5. tick-calibrated subsets (build_tickcal_subset.py)
  6. the leaderboard page

Stale result files are not deleted here; a row whose dataset_version no
longer matches the current one shows as its own (history) section.

Usage: python scripts/eval/rescore_all.py
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
EVAL = REPO / "scripts/eval"
PY = sys.executable


def _run(*args: str) -> None:
    print("+", " ".join(args), flush=True)
    result = subprocess.run([PY, *args], cwd=REPO, capture_output=True, text=True)
    tail = (result.stdout + result.stderr).strip().splitlines()[-3:]
    for line in tail:
        print("   ", line)
    if result.returncode != 0:
        raise SystemExit(f"failed: {' '.join(args)}")


def main() -> None:
    sys.path.insert(0, str(EVAL))
    sys.path.insert(0, str(REPO / "src"))
    import score_llm_predictions as llm

    _run(str(EVAL / "run_baselines.py"))
    _run(
        str(EVAL / "run_lineformer.py"),
        "--score-only",
        "--predictions",
        "data/lineformer_predictions/pretrained-n106.jsonl",
    )
    # run_lineformer.py names its file after the figure count; when the set
    # shrinks, the previous count's file is superseded (its raw predictions
    # stay in data/lineformer_predictions/)
    lf = sorted(
        (REPO / "results").glob("lineformer-pretrained-n*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    lf = [p for p in lf if p.stem[len("lineformer-pretrained-n") :].isdigit()]
    for stale in lf[:-1]:
        print("   removing superseded", stale.name)
        stale.unlink()
    for name in llm.CONDITIONS:
        _run(str(EVAL / "score_llm_predictions.py"), name)
    _run(str(EVAL / "score_chart2table.py"))
    _run(str(EVAL / "build_tickcal_subset.py"))
    _run(str(REPO / "scripts/leaderboard/generate.py"))


if __name__ == "__main__":
    main()
