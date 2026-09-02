"""compute_human_ceiling(): groups FigureAnnotation records by figure_id and
scores inter-annotator agreement for figures with exactly two annotations.

Boundary cases (per task spec, 戦略メモ「柱B」): a single annotation with no
pair to compare, more than two annotations for one figure, no annotations at
all, and mixed annotation_source values.
"""

from __future__ import annotations

import pytest

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.human_ceiling import AnnotationSource
from real_chart_bench.domain.matching import HungarianCurveMatcher
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric
from real_chart_bench.usecase.compute_human_ceiling import (
    CeilingComputationStatus,
    FigureAnnotation,
    compute_human_ceiling,
)


@pytest.fixture
def matcher():
    return HungarianCurveMatcher(metric=NormalizedYDistanceMetric())


def _curve(y_values):
    x = tuple(float(i) for i in range(len(y_values)))
    return Curve(x_values=x, y_values=tuple(float(v) for v in y_values))


def _annotation(figure_id, source, annotator_id, curves=None):
    return FigureAnnotation(
        figure_id=figure_id,
        source=source,
        annotator_id=annotator_id,
        annotated_at="2026-09-10",
        curves=tuple(curves if curves is not None else [_curve([1, 2, 3])]),
    )


def test_no_annotations_at_all_is_pending_not_a_fabricated_score(matcher):
    result = compute_human_ceiling(annotations=[], matcher=matcher)

    assert result.status is CeilingComputationStatus.PENDING_NO_ANNOTATIONS
    assert result.n_figures == 0
    assert result.mean_summary_score is None


def test_a_figure_with_only_one_annotation_is_skipped_not_scored(matcher):
    annotations = [_annotation("f1", AnnotationSource.HUMAN, "alice")]

    result = compute_human_ceiling(annotations=annotations, matcher=matcher)

    assert result.status is CeilingComputationStatus.PENDING_NO_ANNOTATIONS
    assert result.n_figures == 0
    assert len(result.skipped) == 1
    assert result.skipped[0].figure_id == "f1"
    assert "1" in result.skipped[0].reason


def test_a_figure_with_more_than_two_annotations_is_skipped_explicitly(matcher):
    annotations = [
        _annotation("f1", AnnotationSource.HUMAN, "alice"),
        _annotation("f1", AnnotationSource.HUMAN, "bob"),
        _annotation("f1", AnnotationSource.HUMAN, "carol"),
    ]

    result = compute_human_ceiling(annotations=annotations, matcher=matcher)

    assert result.n_figures == 0
    assert len(result.skipped) == 1
    assert "3" in result.skipped[0].reason


def test_two_human_annotations_for_one_figure_score_it(matcher):
    annotations = [
        _annotation("f1", AnnotationSource.HUMAN, "alice", [_curve([1, 2, 3])]),
        _annotation("f1", AnnotationSource.HUMAN, "bob", [_curve([1, 2, 3])]),
    ]

    result = compute_human_ceiling(annotations=annotations, matcher=matcher)

    assert result.status is CeilingComputationStatus.SCORED
    assert result.n_figures == 1
    assert result.mean_summary_score == pytest.approx(1.0)
    assert result.per_figure[0].figure_id == "f1"
    assert result.per_figure[0].summary_score == pytest.approx(1.0)


def test_overall_mean_is_the_mean_of_per_figure_scores(matcher):
    perfect = [_curve([1, 2, 3])]
    imperfect_a = [_curve([0.0, 0.0, 0.0])]
    imperfect_b = [_curve([0.0, 0.0, 1.0])]
    annotations = [
        _annotation("f1", AnnotationSource.HUMAN, "alice", perfect),
        _annotation("f1", AnnotationSource.HUMAN, "bob", perfect),
        _annotation("f2", AnnotationSource.HUMAN, "alice", imperfect_a),
        _annotation("f2", AnnotationSource.HUMAN, "bob", imperfect_b),
    ]

    result = compute_human_ceiling(annotations=annotations, matcher=matcher)

    assert result.n_figures == 2
    scores = [row.summary_score for row in result.per_figure]
    assert result.mean_summary_score == pytest.approx(sum(scores) / len(scores))


def test_all_sources_collects_every_source_used_in_scored_figures(matcher):
    annotations = [
        _annotation("f1", AnnotationSource.HUMAN, "alice"),
        _annotation("f1", AnnotationSource.LLM, "gpt-digitizer"),
    ]

    result = compute_human_ceiling(annotations=annotations, matcher=matcher)

    assert set(result.all_sources) == {AnnotationSource.HUMAN, AnnotationSource.LLM}


def test_skipped_figures_do_not_contribute_to_all_sources(matcher):
    annotations = [
        _annotation("f1", AnnotationSource.HUMAN, "alice"),  # lone annotation, skipped
        _annotation("f2", AnnotationSource.LLM, "gpt-digitizer"),
        _annotation("f2", AnnotationSource.LLM, "claude-digitizer"),
    ]

    result = compute_human_ceiling(annotations=annotations, matcher=matcher)

    assert result.n_figures == 1
    assert set(result.all_sources) == {AnnotationSource.LLM}
    assert len(result.skipped) == 1


def test_mismatched_curve_counts_between_the_two_annotations_still_scores(matcher):
    annotations = [
        _annotation("f1", AnnotationSource.HUMAN, "alice", [_curve([1, 2, 3]), _curve([4, 5, 6])]),
        _annotation("f1", AnnotationSource.HUMAN, "bob", [_curve([1, 2, 3])]),
    ]

    result = compute_human_ceiling(annotations=annotations, matcher=matcher)

    assert result.n_figures == 1
    assert 0.0 <= result.per_figure[0].summary_score <= 1.0
