"""Picks the figure subset to independently re-digitize for the human-ceiling
measurement (戦略メモ「柱B: GTの信頼性を定量化する」, deliverable 3).

Candidates are the VERIFIED entries in data/verified_pairs/registry.json
(111 as of this writing) that have at least one ground-truth curve in
data/verified_pairs/ground_truth.json. The memo's own point about baseline
evaluation subsets applies here too: a convenience subset (all the same
paper, all the easiest figures) would over-credit whatever the re-digitizers
happen to be good at, not measure GT reliability broadly -- so the selection
explicitly covers five axes rather than sampling arbitrarily:

  - axis_type      linear-linear / log-x / log-y / log-both
  - series_bucket  single vs. multiple series
  - points_bucket  total digitized points per figure (population tertile)
  - density_bucket average points per series (population tertile,
                   a proxy for on-chart marker density -- the registry has
                   no direct marker-density field, see usecase module
                   docstring)
  - y_quantity     the plotted property (prop_y), e.g. "Seebeck coefficient"

See usecase/select_human_ceiling_subset.py for the actual (pure, unit
tested) algorithm this script is a thin I/O wrapper around.

Usage:
    python scripts/eval/select_human_ceiling_subset.py [--target-size N]

Prints the selected figure_ids and a full-population-vs-subset distribution
table for each of the five dimensions. Does not write any file under
data/human_ceiling/ -- selecting a subset is not itself annotation data, and
this repo deliberately ships no example/sample files there (see
data/human_ceiling/FORMAT.md).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.usecase.select_human_ceiling_subset import (  # noqa: E402
    build_figure_profiles,
    coverage_report,
    select_coverage_subset,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data/verified_pairs/registry.json"
GROUND_TRUTH_PATH = REPO_ROOT / "data/verified_pairs/ground_truth.json"

_DEFAULT_TARGET_SIZE = 25  # midpoint of the memo's 20-30 figure target


def _load_verified_registry_rows() -> list[dict]:
    registry = json.loads(REGISTRY_PATH.read_text())
    return [row for row in registry if row.get("status") == "verified"]


def _print_distribution_table(report) -> None:
    by_dim: dict[str, list] = {}
    for row in report:
        by_dim.setdefault(row.dimension, []).append(row)

    for dim, rows in by_dim.items():
        print(f"\n{dim}")
        print(f"  {'category':<30} {'full n':>7} {'full %':>8} {'subset n':>9} {'subset %':>9}")
        for row in sorted(rows, key=lambda r: -r.full_count):
            print(
                f"  {row.category:<30} {row.full_count:>7} {row.full_fraction * 100:>7.1f}% "
                f"{row.subset_count:>9} {row.subset_fraction * 100:>8.1f}%"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-size",
        type=int,
        default=_DEFAULT_TARGET_SIZE,
        help=f"figures to select (memo target: 20-30, default {_DEFAULT_TARGET_SIZE})",
    )
    args = parser.parse_args()

    registry_rows = _load_verified_registry_rows()
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    profiles = build_figure_profiles(registry_rows, ground_truth)

    print(
        f"{len(registry_rows)} VERIFIED registry entries, "
        f"{len(profiles)} with ground-truth curves"
    )

    selected = select_coverage_subset(profiles, target_size=args.target_size)

    print(f"\nSelected {len(selected)} figures for independent re-digitization:")
    for figure_id in selected:
        print(f"  {figure_id}")

    report = coverage_report(profiles, selected)
    print(
        f"\n=== Distribution: full population (n={len(profiles)}) "
        f"vs. selected subset (n={len(selected)}) ==="
    )
    _print_distribution_table(report)


if __name__ == "__main__":
    main()
