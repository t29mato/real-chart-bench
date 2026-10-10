"""Bootstrap 95% intervals for the headline point-F1 numbers (paper 4.1 / 4.2).

Reads the per-figure scores of the committed result files (results/*.json),
keeps the non-dense figures (the ones in point_metrics), and reports

- per model and condition: macro point F1 at tau 0.02 with a bootstrap interval,
- per model: the paired difference pixcal - noaxis (figures both conditions scored).

Two resampling units are reported side by side:

- figure: figures are resampled independently (too narrow when figures of one
  paper are correlated),
- paper: whole papers are resampled (a figure's paper is the part of its
  figure_id before the first "-"); the statistic stays the mean over figures.

Usage: python scripts/eval/headline_ci.py [--n-boot 10000] [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
TAU = "0.02"

# label -> (noaxis stem, pixcal stem); same files as check_paper_numbers.py
MODELS: dict[str, tuple[str, str]] = {
    "Claude Opus 5.5": ("claude-opus-5-5-v0-r3-noaxis", "claude-opus-5-5-v0-pixcal"),
    "Claude Fable 5.1": ("claude-fable-5-1-v0-r3-noaxis", "claude-fable-5-1-v0-pixcal"),
    "Claude Sonnet 5.5": ("claude-sonnet-5-5-v0-r3-noaxis", "claude-sonnet-5-5-v0-pixcal"),
    "GPT-6.1-Sol": ("gpt-6.1-sol-v0-codex-noaxis", "gpt-6.1-sol-v0-codex-pixcal"),
    "Qwen3.8-27B Q4 (Codex CLI)": ("", "qwen3.8-27b-v0-codex-local-pixcal"),
    "Claude Haiku 4.5": ("claude-haiku-4-5-v0-r3-noaxis", "claude-haiku-4-5-v0-pixcal"),
}


def paper_of(figure_id: str) -> str:
    return figure_id.split("-", 1)[0]


def load_point_f1(data: dict, tau: str = TAU) -> dict[str, float]:
    """{figure_id: point_f1} over the figures that count in the macro (non-dense)."""
    out = {}
    for row in data["per_figure"]:
        if (row.get("marker_density") or {}).get("dense"):
            continue
        point = row.get("point")
        if not point:
            continue
        out[row["figure_id"]] = point["by_tau"][tau]["point_f1"]
    return out


def bootstrap_mean(
    values: list[float],
    groups: list[str] | None = None,
    n_boot: int = 10000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Mean of values and its percentile interval.

    groups None: resample values one by one. Otherwise resample the distinct
    groups with replacement and take the mean over all values of the chosen
    groups (a figure of a twice-drawn paper counts twice).
    """
    if not values:
        raise ValueError("bootstrap_mean needs a non-empty list")
    if groups is not None and len(groups) != len(values):
        raise ValueError("groups and values must have the same length")
    mean = sum(values) / len(values)
    rng = random.Random(seed)
    means = []
    if groups is None:
        n = len(values)
        for _ in range(n_boot):
            means.append(sum(rng.choice(values) for _ in range(n)) / n)
    else:
        by_group: dict[str, list[float]] = {}
        for g, v in zip(groups, values, strict=True):
            by_group.setdefault(g, []).append(v)
        sums = [(sum(v), len(v)) for v in by_group.values()]
        k = len(sums)
        for _ in range(n_boot):
            picks = [sums[rng.randrange(k)] for _ in range(k)]
            means.append(sum(s for s, _ in picks) / sum(c for _, c in picks))
    means.sort()
    tail = (1 - level) / 2
    lo = means[int(tail * n_boot)]
    hi = means[min(n_boot - 1, int((1 - tail) * n_boot) - 1)]
    if all(abs(v - values[0]) < 1e-12 for v in values):
        lo = hi = mean
    return mean, lo, hi


def paired_difference(a: dict[str, float], b: dict[str, float]) -> tuple[list[str], list[float]]:
    """Figure ids both score sets contain, and a - b on them."""
    keys = sorted(set(a) & set(b))
    return keys, [a[k] - b[k] for k in keys]


def load_stem(stem: str) -> dict[str, float]:
    return load_point_f1(json.loads((RESULTS / f"{stem}.json").read_text()))


def fmt(t: tuple[float, float, float], signed: bool = False) -> str:
    spec = "+.3f" if signed else ".3f"
    return f"{t[0]:{spec}} [{t[1]:{spec}}, {t[2]:{spec}}]"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    kw = {"n_boot": args.n_boot, "seed": args.seed}

    scores = {
        label: tuple(load_stem(s) if s else {} for s in stems) for label, stems in MODELS.items()
    }
    papers = {paper_of(k) for pair in scores.values() for s in pair for k in s}
    print(f"n_boot {args.n_boot}, seed {args.seed}, tau {TAU}; papers {len(papers)}")
    print()
    print("| model | condition | n figs | n papers | point F1 (figure) | point F1 (paper) |")
    print("|---|---|---|---|---|---|")
    for label, pair in scores.items():
        for cond, s in zip(("noaxis", "pixcal"), pair, strict=True):
            if not s:
                continue
            keys = sorted(s)
            vals = [s[k] for k in keys]
            grp = [paper_of(k) for k in keys]
            print(
                f"| {label} | {cond} | {len(keys)} | {len(set(grp))} | "
                f"{fmt(bootstrap_mean(vals, None, **kw))} | "
                f"{fmt(bootstrap_mean(vals, grp, **kw))} |"
            )
    print()
    print("| model | n figs | n papers | pixcal - noaxis (figure) | pixcal - noaxis (paper) |")
    print("|---|---|---|---|---|")
    for label, (noaxis, pixcal) in scores.items():
        if not noaxis or not pixcal:
            continue
        keys, diffs = paired_difference(pixcal, noaxis)
        grp = [paper_of(k) for k in keys]
        print(
            f"| {label} | {len(keys)} | {len(set(grp))} | "
            f"{fmt(bootstrap_mean(diffs, None, **kw), True)} | "
            f"{fmt(bootstrap_mean(diffs, grp, **kw), True)} |"
        )


if __name__ == "__main__":
    main()
