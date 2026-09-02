"""Renders a HumanCeilingComputation into the same results/*.json shape
scripts/leaderboard/generate.py already consumes (model_id, model_name,
dataset_version, run_at, n_figures, mean_summary_score, per_figure -- see
results/naive-cv-v0.json; or the pending_external_run shape build_leaderboard.py
already knows how to render as a "pending" row -- see build_leaderboard.py's
_PENDING_ROW_TEMPLATE).

The "human-ceiling" model_id/model_name -- the identity that makes a row read
as *the* human ceiling on the leaderboard -- may only ever be assigned
through ``domain.human_ceiling.require_human_ceiling()`` not raising. This is
the single call site that is allowed to grant that identity; every other
outcome (no annotations yet, mixed sources, machine-only sources) is labeled
honestly instead, per the project rule that an LLM judgment must never be
presented as a human one.
"""

from __future__ import annotations

from real_chart_bench.domain.human_ceiling import (
    AnnotationSource,
    CeilingLabel,
    NotAHumanCeilingError,
    classify_ceiling_label,
    require_human_ceiling,
)
from real_chart_bench.usecase.compute_human_ceiling import (
    CeilingComputationStatus,
    HumanCeilingComputation,
)

_PENDING_NOTE = (
    "no annotations yet under data/human_ceiling/annotations/ -- run "
    "scripts/eval/select_human_ceiling_subset.py to choose the figures to "
    "re-digitize, add independent digitizations there (see "
    "data/human_ceiling/FORMAT.md), then re-run scripts/eval/compute_human_ceiling.py."
)


def _leaderboard_identity(all_sources: tuple[AnnotationSource, ...]) -> tuple[str, str, str]:
    """Returns (model_id, model_name, ceiling_label). The "human-ceiling"
    identity is granted only through require_human_ceiling() not raising --
    every other branch is reached only via its NotAHumanCeilingError."""
    try:
        require_human_ceiling(all_sources)
    except NotAHumanCeilingError:
        label = classify_ceiling_label(all_sources)
        if label is CeilingLabel.MACHINE_AGREEMENT:
            return (
                "human-ceiling-machine-agreement",
                "Annotator agreement (MACHINE-ONLY sources -- not a human ceiling)",
                label.value,
            )
        return (
            "human-ceiling-mixed-sources",
            "Annotator agreement (MIXED sources -- not a human ceiling)",
            label.value,
        )
    return (
        "human-ceiling",
        "Human ceiling (independent re-digitization agreement)",
        CeilingLabel.HUMAN_CEILING.value,
    )


def build_human_ceiling_result(
    computation: HumanCeilingComputation,
    *,
    dataset_version: str,
    run_at: str,
) -> dict:
    if computation.status is CeilingComputationStatus.PENDING_NO_ANNOTATIONS:
        return {
            "model_id": "human-ceiling",
            "model_name": "Human ceiling (independent re-digitization agreement)",
            "status": "pending_external_run",
            "note": _PENDING_NOTE,
        }

    model_id, model_name, ceiling_label = _leaderboard_identity(computation.all_sources)

    return {
        "model_id": model_id,
        "model_name": model_name,
        "dataset_version": dataset_version,
        "run_at": run_at,
        "n_figures": computation.n_figures,
        "mean_summary_score": computation.mean_summary_score,
        "ceiling_label": ceiling_label,
        "annotation_sources": sorted({s.value for s in computation.all_sources}),
        "per_figure": [
            {
                "figure_id": row.figure_id,
                "summary_score": row.summary_score,
                "match_rate_a_as_predicted": row.match_rate_a_as_predicted,
                "match_rate_b_as_predicted": row.match_rate_b_as_predicted,
                "mean_coverage_ratio_a_as_predicted": row.mean_coverage_ratio_a_as_predicted,
                "mean_coverage_ratio_b_as_predicted": row.mean_coverage_ratio_b_as_predicted,
                "annotator_a": row.annotator_a,
                "annotator_b": row.annotator_b,
                "error": None,
            }
            for row in computation.per_figure
        ],
        "skipped_figures": [
            {"figure_id": s.figure_id, "reason": s.reason} for s in computation.skipped
        ],
    }
