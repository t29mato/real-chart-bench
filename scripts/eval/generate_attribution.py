"""Generates data/verified_pairs/ATTRIBUTION.md from registry.json + papers.json
(design §7.33, HQ priority-interrupt 2026-08-23).

Why this exists: committing figure images/crops under data/verified_pairs/
(images/, crops/) is active redistribution of copyrighted figure content.
CC BY 4.0 requires attribution (credit the source, indicate the license,
indicate if changes were made) for each redistributed work. Generating this
from the registry rather than hand-writing it means it can never go stale as
the registry grows -- the same "never hardcode what you can derive" lesson
as dataset_version (design §7.28) and the leaderboard version banner.

The ``nc`` subset (design §7.88.1) gets its own file,
data/verified_pairs_nc/ATTRIBUTION.md, with a non-commercial notice and the
share-alike condition of NC-SA figures. It is only written once at least one
nc figure is committed; core's file and wording are unchanged.

Usage:
    python scripts/eval/generate_attribution.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.domain.dataset_subset import (  # noqa: E402
    DatasetSubset,
    distribution_dir_name,
)
from real_chart_bench.usecase.build_attribution import (  # noqa: E402
    attribution_rows,
    render_attribution,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data/verified_pairs/registry.json"
PAPERS_PATH = REPO_ROOT / "data/manifest/v0/papers.json"


def out_path(subset: DatasetSubset) -> pathlib.Path:
    return REPO_ROOT / "data" / distribution_dir_name(subset) / "ATTRIBUTION.md"


def main() -> None:
    registry = load_registry(REGISTRY_PATH)
    papers_by_id = {p["paper_id"]: p for p in json.loads(PAPERS_PATH.read_text())}

    for subset in DatasetSubset:
        rows = attribution_rows(
            registry,
            papers_by_id,
            subset=subset,
            file_exists=lambda path: (REPO_ROOT / path).exists(),
        )
        path = out_path(subset)
        if subset is not DatasetSubset.CORE and not rows and not path.exists():
            continue  # no nc figure committed yet: no nc directory either
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_attribution(subset, rows))
        print(f"wrote {path} ({len(rows)} attributed file(s), subset {subset.value})")


if __name__ == "__main__":
    main()
