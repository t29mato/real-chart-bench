"""Computes the human-ceiling agreement score and writes
results/human-ceiling.json in the same shape scripts/leaderboard/generate.py
already consumes (戦略メモ「柱B: GTの信頼性を定量化する」, deliverable 4).

Compares, per figure, the original Starrydata ground truth
(data/verified_pairs/ground_truth.json -- annotation_source=human by
construction, since that is what it is) against the independent
re-digitization(s) recorded under data/human_ceiling/annotations/ (see
data/human_ceiling/FORMAT.md). The comparison itself is
domain.human_ceiling.compare_annotations(), the exact same
NormalizedYDistanceMetric + HungarianCurveMatcher the leaderboard scores
models with.

Degrades loudly, not silently: if data/human_ceiling/annotations/ has no
files yet (the expected state until re-digitization actually happens), this
writes a pending_external_run row with an explanatory note -- never a
fabricated or placeholder score. If the annotations that do exist are not
all annotation_source=human, the result is still computed and written, but
under an honestly different model_id/model_name
(human-ceiling-mixed-sources / human-ceiling-machine-agreement) --
see usecase/build_human_ceiling_result.py; the "human-ceiling" identity is
only ever granted through domain.human_ceiling.require_human_ceiling() not
raising.

Usage:
    python scripts/eval/compute_human_ceiling.py
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import UTC, datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.adapter.human_ceiling_annotations import (  # noqa: E402
    load_annotation_files,
)
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.domain.curve import Curve, ScaleType  # noqa: E402
from real_chart_bench.domain.human_ceiling import AnnotationSource  # noqa: E402
from real_chart_bench.domain.matching import HungarianCurveMatcher  # noqa: E402
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric  # noqa: E402
from real_chart_bench.usecase.build_human_ceiling_result import (  # noqa: E402
    build_human_ceiling_result,
)
from real_chart_bench.usecase.compute_human_ceiling import (  # noqa: E402
    FigureAnnotation,
    compute_human_ceiling,
)
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data/verified_pairs/registry.json"
GROUND_TRUTH_PATH = REPO_ROOT / "data/verified_pairs/ground_truth.json"
ANNOTATIONS_DIR = REPO_ROOT / "data/human_ceiling/annotations"
RESULTS_PATH = REPO_ROOT / "results/human-ceiling.json"
DATASET_VERSION = "v0-eval-pilot-n111"

_ORIGINAL_ANNOTATOR_ID = "starrydata-original"


def _original_annotations() -> list[FigureAnnotation]:
    """The existing Starrydata digitization for every VERIFIED registry
    entry, wrapped as a FigureAnnotation(source=HUMAN) so it can be scored
    by the same compute_human_ceiling() pairing/grouping logic as any other
    annotation. Mirrors scripts/eval/run_baselines.py's own ground-truth
    loading (kept separate rather than importing that script, since scripts/
    are one-off entry points, not a shared library -- see AGENTS.md)."""
    registry = load_registry(REGISTRY_PATH)
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())

    annotations = []
    for pairing in select_verified_pairings(registry):
        raw_curves = ground_truth.get(pairing.figure_id, [])
        curves = tuple(
            Curve(
                x_values=tuple(row["x"]),
                y_values=tuple(row["y"]),
                series_label=row.get("prop_y", ""),
                x_scale=pairing.x_scale,
            )
            for row in raw_curves
            if row["x"]
        )
        if not curves:
            continue
        annotations.append(
            FigureAnnotation(
                figure_id=f"{pairing.paper_id}-{pairing.figure_id}",
                source=AnnotationSource.HUMAN,
                annotator_id=_ORIGINAL_ANNOTATOR_ID,
                annotated_at=pairing.verified_at,
                curves=curves,
            )
        )
    return annotations


def _x_scale_by_figure_id(registry) -> dict[str, ScaleType]:
    return {f"{p.paper_id}-{p.figure_id}": p.x_scale for p in registry}


def main() -> None:
    registry = load_registry(REGISTRY_PATH)
    original = _original_annotations()
    second = load_annotation_files(
        ANNOTATIONS_DIR, x_scale_by_figure_id=_x_scale_by_figure_id(registry)
    )

    matcher = HungarianCurveMatcher(metric=NormalizedYDistanceMetric())
    computation = compute_human_ceiling(annotations=original + second, matcher=matcher)

    if not second:
        print(
            f"No annotations found under {ANNOTATIONS_DIR} -- writing a pending row, "
            "not a fabricated score. Run scripts/eval/select_human_ceiling_subset.py "
            "to choose figures, then add independent digitizations there."
        )

    payload = build_human_ceiling_result(
        computation,
        dataset_version=DATASET_VERSION,
        run_at=datetime.now(UTC).isoformat(),
    )

    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {RESULTS_PATH} (model_id={payload['model_id']!r})")

    if computation.skipped:
        print(f"\n{len(computation.skipped)} figure(s) skipped (not scored):")
        for s in computation.skipped:
            print(f"  {s.figure_id}: {s.reason}")


if __name__ == "__main__":
    main()
