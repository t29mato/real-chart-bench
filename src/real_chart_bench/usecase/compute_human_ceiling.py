"""Groups independent-digitization annotations by figure and scores
inter-annotator agreement (戦略メモ「柱B: GTの信頼性を定量化する」).

Orchestration only: the actual agreement score comes unmodified from
``domain.human_ceiling.compare_annotations`` (which itself reuses the
project's normal figure-scoring machinery, see that module's docstring).
This module's job is grouping annotation records by figure_id, deciding
which figures have enough annotations to score, and reporting the figures it
could *not* score explicitly rather than silently dropping them -- same
philosophy as ``domain/matching.py``'s unmatched-series reporting.

v0 deliberately requires *exactly two* annotations per figure to score it: a
figure with only one annotation has no pair to compare (skipped, not scored
as a fabricated ceiling of some kind), and a figure with more than two is
also skipped rather than the code silently guessing which two to pair --
see ``_group_by_figure`` and its tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.human_ceiling import (
    AnnotationSource,
    PairwiseAgreementResult,
    compare_annotations,
)
from real_chart_bench.domain.matching import CurveMatcher


@dataclass(frozen=True)
class FigureAnnotation:
    """One independent digitization of one figure. Mirrors the mandatory
    fields of the data/human_ceiling/ JSON record format (see
    data/human_ceiling/FORMAT.md) -- ``source`` is never optional or
    defaulted anywhere in this pipeline, by design."""

    figure_id: str
    source: AnnotationSource
    annotator_id: str
    annotated_at: str
    curves: tuple[Curve, ...]


@dataclass(frozen=True)
class SkippedFigure:
    figure_id: str
    reason: str


@dataclass(frozen=True)
class FigureCeilingRow:
    figure_id: str
    summary_score: float
    match_rate_a_as_predicted: float
    match_rate_b_as_predicted: float
    mean_coverage_ratio_a_as_predicted: float
    mean_coverage_ratio_b_as_predicted: float
    annotator_a: str
    annotator_b: str


class CeilingComputationStatus(Enum):
    SCORED = "scored"
    PENDING_NO_ANNOTATIONS = "pending_no_annotations"


@dataclass(frozen=True)
class HumanCeilingComputation:
    status: CeilingComputationStatus
    n_figures: int
    mean_summary_score: float | None
    per_figure: tuple[FigureCeilingRow, ...]
    skipped: tuple[SkippedFigure, ...]
    all_sources: tuple[AnnotationSource, ...]


def _group_by_figure(
    annotations: Sequence[FigureAnnotation],
) -> dict[str, list[FigureAnnotation]]:
    grouped: dict[str, list[FigureAnnotation]] = {}
    for annotation in annotations:
        grouped.setdefault(annotation.figure_id, []).append(annotation)
    return grouped


def _row_from_agreement(
    agreement: PairwiseAgreementResult, annotator_a: str, annotator_b: str
) -> FigureCeilingRow:
    return FigureCeilingRow(
        figure_id=agreement.figure_id,
        summary_score=agreement.mean_summary_score,
        match_rate_a_as_predicted=agreement.a_as_predicted.match_rate,
        match_rate_b_as_predicted=agreement.b_as_predicted.match_rate,
        mean_coverage_ratio_a_as_predicted=agreement.a_as_predicted.mean_coverage_ratio,
        mean_coverage_ratio_b_as_predicted=agreement.b_as_predicted.mean_coverage_ratio,
        annotator_a=annotator_a,
        annotator_b=annotator_b,
    )


def compute_human_ceiling(
    *,
    annotations: Sequence[FigureAnnotation],
    matcher: CurveMatcher,
) -> HumanCeilingComputation:
    grouped = _group_by_figure(annotations)

    per_figure: list[FigureCeilingRow] = []
    skipped: list[SkippedFigure] = []
    all_sources: list[AnnotationSource] = []

    for figure_id, figure_annotations in grouped.items():
        n = len(figure_annotations)
        if n != 2:
            skipped.append(
                SkippedFigure(
                    figure_id=figure_id,
                    reason=(
                        f"expected exactly 2 annotations to compute agreement, found {n} "
                        f"(annotator_id(s): {[a.annotator_id for a in figure_annotations]})"
                    ),
                )
            )
            continue

        a, b = figure_annotations
        agreement = compare_annotations(
            figure_id=figure_id, annotation_a=a.curves, annotation_b=b.curves, matcher=matcher
        )
        per_figure.append(_row_from_agreement(agreement, a.annotator_id, b.annotator_id))
        all_sources.extend([a.source, b.source])

    if not per_figure:
        return HumanCeilingComputation(
            status=CeilingComputationStatus.PENDING_NO_ANNOTATIONS,
            n_figures=0,
            mean_summary_score=None,
            per_figure=(),
            skipped=tuple(skipped),
            all_sources=(),
        )

    mean_summary_score = sum(row.summary_score for row in per_figure) / len(per_figure)
    return HumanCeilingComputation(
        status=CeilingComputationStatus.SCORED,
        n_figures=len(per_figure),
        mean_summary_score=mean_summary_score,
        per_figure=tuple(per_figure),
        skipped=tuple(skipped),
        all_sources=tuple(all_sources),
    )
