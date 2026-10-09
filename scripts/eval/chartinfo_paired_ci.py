"""Paired bootstrap intervals for the differences between models on the
CHART-Infographics scatter measurement (paper 4.8).

Reads the per-figure rows of data/chartinfo_runs/batchNN/<model>.score.txt
(written by score_chartinfo_pilot.py), pairs the figures every model answered,
and resamples figures with replacement. Figures are the resampling unit; the
source papers are NOT used as clusters, so the intervals may be too narrow.

Usage: python scripts/eval/chartinfo_paired_ci.py [--n-boot 2000] [--seed 0]
"""

from __future__ import annotations

import argparse
import pathlib
import random

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / "data/chartinfo_runs"
MODELS = {
    "claude-opus-5-5": "Opus 5.5",
    "claude-fable-5-1": "Fable 5.1",
    "claude-sonnet-5-5": "Sonnet 5.5",
    "gpt-6.1-sol": "GPT-6.1-Sol",
}
PAIRS = [
    ("claude-opus-5-5", "claude-fable-5-1"),
    ("claude-opus-5-5", "gpt-6.1-sol"),
    ("claude-opus-5-5", "claude-sonnet-5-5"),
    ("claude-fable-5-1", "gpt-6.1-sol"),
    ("claude-fable-5-1", "claude-sonnet-5-5"),
    ("gpt-6.1-sol", "claude-sonnet-5-5"),
]
# columns after: fig, n_gt, n_pred
FIELDS = ("combined", "name", "data", "point_f1", "recall", "precision")


def parse_score_text(text: str) -> dict[str, dict[str, float]]:
    """Per-figure rows of a score.txt.

    Returns {fig: {combined, name, data, point_f1, recall, precision}}.
    """
    rows = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 3 + len(FIELDS) or not parts[0].startswith("fig_"):
            continue
        values = [float(v) for v in parts[3 : 3 + len(FIELDS)]]
        rows[parts[0]] = dict(zip(FIELDS, values, strict=True))
    return rows


def paired_bootstrap(
    a: list[float], b: list[float], n_boot: int = 2000, seed: int = 0, level: float = 0.95
) -> tuple[float, float, float]:
    """Mean of a - b and its percentile interval, resampling pairs."""
    if len(a) != len(b) or not a:
        raise ValueError("paired_bootstrap needs two non-empty lists of the same length")
    d = [x - y for x, y in zip(a, b, strict=True)]
    n = len(d)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(d) for _ in range(n)) / n for _ in range(n_boot))
    tail = (1 - level) / 2
    lo = means[int(tail * n_boot)]
    hi = means[min(n_boot - 1, int((1 - tail) * n_boot) - 1)]
    mean = sum(d) / n
    # guard against float noise when every difference is the same
    if all(abs(x - d[0]) < 1e-12 for x in d):
        lo = hi = mean
    return round(mean, 12), round(lo, 12), round(hi, 12)


def load_scores() -> dict[str, dict[tuple[str, str], dict[str, float]]]:
    out: dict[str, dict[tuple[str, str], dict[str, float]]] = {m: {} for m in MODELS}
    for model in MODELS:
        for f in sorted(RUNS.glob(f"batch*/{model}.score.txt")):
            for fig, row in parse_score_text(f.read_text()).items():
                out[model][(f.parent.name, fig)] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    scores = load_scores()
    keys = sorted(set.intersection(*(set(v) for v in scores.values())))
    print(f"figures answered by every model: {len(keys)}; n_boot {args.n_boot}, seed {args.seed}")
    print()
    print("| difference | combined | point F1 |")
    print("|---|---|---|")
    for a, b in PAIRS:
        cells = []
        for field in ("combined", "point_f1"):
            mean, lo, hi = paired_bootstrap(
                [scores[a][k][field] for k in keys],
                [scores[b][k][field] for k in keys],
                n_boot=args.n_boot,
                seed=args.seed,
            )
            cells.append(f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        print(f"| {MODELS[a]} − {MODELS[b]} | {cells[0]} | {cells[1]} |")


if __name__ == "__main__":
    main()
