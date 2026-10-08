"""Totals of the CHART-Infographics scatter measurement over every batch.

Reads data/chartinfo_runs/batchNN/<model>.score.txt (written by
score_chartinfo_pilot.py through archive_score_chartinfo.sh) and prints a
Markdown summary: per-model figure-weighted means of the official metric6b
scores (combined / name / data) and of this project's point F1, plus the
per-batch combined score. Each batch score is already a mean over its figures,
so weighting by the figure count gives the mean over all figures.

Usage: python scripts/eval/aggregate_chartinfo.py > data/chartinfo_runs/SUMMARY.md
"""

from __future__ import annotations

import collections
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / "data/chartinfo_runs"
LINE = re.compile(
    r"全(\d+)図\s+総合 ([\d.]+)\s+名前 ([\d.]+)\s+データ ([\d.]+)\s+\|\s+"
    r"点F1 ([\d.]+)\s+再現 ([\d.]+)\s+適合 ([\d.]+)"
)
FIELDS = ("combined", "name", "data", "point_f1", "recall", "precision")
NAMES = {
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-fable-5-1": "Claude Fable 5.1",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "gpt-6.1-sol": "GPT-6.1-Sol (Codex CLI)",
}


def read_scores() -> dict[str, dict[str, tuple[int, list[float]]]]:
    out: dict[str, dict[str, tuple[int, list[float]]]] = collections.defaultdict(dict)
    for f in sorted(RUNS.glob("batch*/*.score.txt")):
        m = LINE.search(f.read_text())
        if m:
            out[f.name.removesuffix(".score.txt")][f.parent.name] = (
                int(m[1]),
                [float(v) for v in m.groups()[1:]],
            )
    return out


def main() -> None:
    scores = read_scores()
    totals = {}
    for model, batches in scores.items():
        n = sum(k for k, _ in batches.values())
        totals[model] = (
            n,
            len(batches),
            [sum(k * v[i] for k, v in batches.values()) / n for i in range(len(FIELDS))],
        )
    order = sorted(totals, key=lambda m: -totals[m][2][0])
    print("# CHART-Infographics scatter measurement — totals\n")
    print(
        "Figure-weighted means over every scored batch (`aggregate_chartinfo.py`). "
        "combined / name / data: the official metric6b (unmodified). "
        "point F1 / recall / precision: this project's point F1 (tau 2%, "
        "axis ranges recovered from task2/task4). Fully automatic condition, prompt v2.\n"
    )
    print("| model | figures | batches | " + " | ".join(FIELDS) + " |")
    print("|---|---|---|" + "---|" * len(FIELDS))
    for m in order:
        n, k, v = totals[m]
        print(f"| {NAMES.get(m, m)} | {n} | {k} | " + " | ".join(f"{x:.3f}" for x in v) + " |")
    print(
        "| ICPR 2020 best (task 6b, upstream ground truth given; all chart types) "
        "| — | — | 0.710 | — | — | — | — | — |\n"
    )
    batches = sorted({b for s in scores.values() for b in s})
    print("## combined score per batch\n")
    print("| batch | figures | " + " | ".join(NAMES.get(m, m) for m in order) + " |")
    print("|---|---|" + "---|" * len(order))
    for b in batches:
        n = next(s[b][0] for s in scores.values() if b in s)
        cells = [f"{scores[m][b][1][0]:.3f}" if b in scores[m] else "—" for m in order]
        print(f"| {b} | {n} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
