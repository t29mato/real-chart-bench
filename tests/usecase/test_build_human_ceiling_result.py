"""build_human_ceiling_result(): turns a HumanCeilingComputation into the
same results/*.json shape scripts/leaderboard/generate.py already consumes
(see results/naive-cv-v0.json for the shape being matched), so the ceiling
can appear as a leaderboard row.

The "human-ceiling" model_id/model_name identity may only ever be assigned
through require_human_ceiling() not raising -- this is the structural
enforcement of the project rule that an LLM judgment must never be presented
as a human one.
"""

from __future__ import annotations

from real_chart_bench.domain.human_ceiling import AnnotationSource
from real_chart_bench.usecase.build_human_ceiling_result import build_human_ceiling_result
from real_chart_bench.usecase.compute_human_ceiling import (
    CeilingComputationStatus,
    FigureCeilingRow,
    HumanCeilingComputation,
    SkippedFigure,
)

_DATASET_VERSION = "v0-eval-pilot-n111"
_RUN_AT = "2026-09-10T00:00:00+00:00"


def _row(figure_id="4173-20120", score=0.9):
    return FigureCeilingRow(
        figure_id=figure_id,
        summary_score=score,
        match_rate_a_as_predicted=1.0,
        match_rate_b_as_predicted=1.0,
        mean_coverage_ratio_a_as_predicted=1.0,
        mean_coverage_ratio_b_as_predicted=1.0,
        annotator_a="alice",
        annotator_b="bob",
    )


def test_pending_computation_emits_a_pending_row_not_a_fabricated_score():
    computation = HumanCeilingComputation(
        status=CeilingComputationStatus.PENDING_NO_ANNOTATIONS,
        n_figures=0,
        mean_summary_score=None,
        per_figure=(),
        skipped=(),
        all_sources=(),
    )

    payload = build_human_ceiling_result(
        computation, dataset_version=_DATASET_VERSION, run_at=_RUN_AT
    )

    assert payload["status"] == "pending_external_run"
    assert "mean_summary_score" not in payload
    assert "annotation" in payload["note"].lower()
    assert "data/human_ceiling" in payload["note"]


def test_all_human_sources_get_the_official_human_ceiling_identity():
    computation = HumanCeilingComputation(
        status=CeilingComputationStatus.SCORED,
        n_figures=1,
        mean_summary_score=0.9,
        per_figure=(_row(),),
        skipped=(),
        all_sources=(AnnotationSource.HUMAN, AnnotationSource.HUMAN),
    )

    payload = build_human_ceiling_result(
        computation, dataset_version=_DATASET_VERSION, run_at=_RUN_AT
    )

    assert payload["model_id"] == "human-ceiling"
    assert "human ceiling" in payload["model_name"].lower()
    assert payload["mean_summary_score"] == 0.9
    assert payload["n_figures"] == 1
    assert payload["dataset_version"] == _DATASET_VERSION
    assert payload["run_at"] == _RUN_AT
    assert payload["ceiling_label"] == "human_ceiling"
    assert payload["annotation_sources"] == ["human"]


def test_any_llm_source_refuses_the_human_ceiling_identity():
    computation = HumanCeilingComputation(
        status=CeilingComputationStatus.SCORED,
        n_figures=1,
        mean_summary_score=0.9,
        per_figure=(_row(),),
        skipped=(),
        all_sources=(AnnotationSource.HUMAN, AnnotationSource.LLM),
    )

    payload = build_human_ceiling_result(
        computation, dataset_version=_DATASET_VERSION, run_at=_RUN_AT
    )

    assert payload["model_id"] != "human-ceiling"
    assert "not a human ceiling" in payload["model_name"].lower()
    assert payload["ceiling_label"] == "mixed_source_agreement"
    # the number is still reported -- refusal is about labeling, not hiding data
    assert payload["mean_summary_score"] == 0.9


def test_all_llm_sources_are_labeled_machine_agreement_not_mixed():
    computation = HumanCeilingComputation(
        status=CeilingComputationStatus.SCORED,
        n_figures=1,
        mean_summary_score=0.5,
        per_figure=(_row(),),
        skipped=(),
        all_sources=(AnnotationSource.LLM, AnnotationSource.LLM),
    )

    payload = build_human_ceiling_result(
        computation, dataset_version=_DATASET_VERSION, run_at=_RUN_AT
    )

    assert payload["model_id"] != "human-ceiling"
    assert payload["ceiling_label"] == "machine_agreement"
    assert "machine" in payload["model_name"].lower()


def test_per_figure_rows_are_present_for_the_breakdown_join():
    computation = HumanCeilingComputation(
        status=CeilingComputationStatus.SCORED,
        n_figures=2,
        mean_summary_score=0.8,
        per_figure=(_row("f1", 0.9), _row("f2", 0.7)),
        skipped=(SkippedFigure(figure_id="f3", reason="expected exactly 2, found 1"),),
        all_sources=(AnnotationSource.HUMAN, AnnotationSource.HUMAN),
    )

    payload = build_human_ceiling_result(
        computation, dataset_version=_DATASET_VERSION, run_at=_RUN_AT
    )

    figure_ids = {pf["figure_id"] for pf in payload["per_figure"]}
    assert figure_ids == {"f1", "f2"}
    assert all("summary_score" in pf for pf in payload["per_figure"])
    assert payload["skipped_figures"] == [
        {"figure_id": "f3", "reason": "expected exactly 2, found 1"}
    ]
