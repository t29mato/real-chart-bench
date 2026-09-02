"""Human-ceiling agreement metric (戦略メモ「柱B: GTの信頼性を定量化する」).

compare_annotations() must reuse the exact same metric/matcher machinery the
leaderboard scores models with (NormalizedYDistanceMetric +
HungarianCurveMatcher via evaluate_figure) so a human-ceiling score sits on
the same axis as a model's summary_score -- that's the entire point of
publishing it.

NormalizedYDistanceMetric is *asymmetric*: it interpolates the "predicted"
curve onto the "ground truth" curve's own x-grid and normalizes by the
ground truth's y-range (see domain/metrics.py). Two independent
digitizations of the same figure have no natural predicted/ground-truth
ordering, so compare_annotations() symmetrizes by averaging both directions
-- see test_symmetrized_score_is_order_independent below.

classify_ceiling_label()/require_human_ceiling() are the structural guard
against ever presenting a non-human-sourced agreement number as a "human
ceiling" (project rule: an LLM judgment must never be presented as a human
one).
"""

import pytest

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.human_ceiling import (
    AnnotationSource,
    CeilingLabel,
    NotAHumanCeilingError,
    classify_ceiling_label,
    compare_annotations,
    require_human_ceiling,
)
from real_chart_bench.domain.matching import HungarianCurveMatcher
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric


@pytest.fixture
def matcher():
    return HungarianCurveMatcher(metric=NormalizedYDistanceMetric())


def _line(y_values, x_values=None):
    x = x_values or tuple(float(i) for i in range(len(y_values)))
    return Curve(x_values=x, y_values=tuple(float(v) for v in y_values))


# --- compare_annotations() -------------------------------------------------


def test_identical_annotations_score_a_perfect_agreement(matcher):
    curve = _line([1, 2, 3])

    result = compare_annotations(
        figure_id="4173-20120", annotation_a=[curve], annotation_b=[curve], matcher=matcher
    )

    assert result.mean_summary_score == pytest.approx(1.0)
    assert result.a_as_predicted.summary_score == pytest.approx(1.0)
    assert result.b_as_predicted.summary_score == pytest.approx(1.0)


def test_uses_the_same_metric_family_as_a_direct_evaluate_figure_call(matcher):
    # Not a re-implementation: b_as_predicted must be exactly what
    # evaluate_figure(predicted=b, ground_truth=a, matcher) returns, and
    # a_as_predicted the mirror -- this is the "reuse, don't reinvent" check.
    from real_chart_bench.domain.evaluation import evaluate_figure

    a = [_line([0.0, 0.0, 0.0])]
    b = [_line([0.0, 0.0, 1.0])]

    result = compare_annotations(figure_id="f", annotation_a=a, annotation_b=b, matcher=matcher)

    expected_a_as_predicted = evaluate_figure(predicted=a, ground_truth=b, matcher=matcher)
    expected_b_as_predicted = evaluate_figure(predicted=b, ground_truth=a, matcher=matcher)
    assert result.a_as_predicted.summary_score == pytest.approx(
        expected_a_as_predicted.summary_score
    )
    assert result.b_as_predicted.summary_score == pytest.approx(
        expected_b_as_predicted.summary_score
    )
    expected_mean = (
        expected_a_as_predicted.summary_score + expected_b_as_predicted.summary_score
    ) / 2
    assert result.mean_summary_score == pytest.approx(expected_mean)


def test_symmetrized_score_is_order_independent_despite_asymmetric_metric(matcher):
    # Craft a case where the underlying metric genuinely disagrees by
    # direction (different y-range normalizers), so a naive "just call
    # evaluate_figure once" implementation would silently depend on which
    # annotation happened to be passed first.
    a = [_line([0.0, 10.0])]  # y-range 10
    b = [_line([0.0, 1.0])]  # y-range 1 -- normalizing by this is much stricter

    forward = compare_annotations(figure_id="f", annotation_a=a, annotation_b=b, matcher=matcher)
    backward = compare_annotations(figure_id="f", annotation_a=b, annotation_b=a, matcher=matcher)

    # The underlying metric really is asymmetric (sanity-check the premise).
    assert forward.a_as_predicted.summary_score != pytest.approx(
        forward.b_as_predicted.summary_score
    )
    # But the symmetrized, order-independent result is not.
    assert forward.mean_summary_score == pytest.approx(backward.mean_summary_score)


def test_differing_series_counts_does_not_crash_and_stays_in_bounds(matcher):
    a = [_line([1, 2, 3]), _line([4, 5, 6])]
    b = [_line([1, 2, 3])]

    result = compare_annotations(figure_id="f", annotation_a=a, annotation_b=b, matcher=matcher)

    assert 0.0 <= result.mean_summary_score <= 1.0
    # b has one fewer series than a: from b's perspective (as predicted
    # against a as ground truth) there's a detection miss, lowering match_rate.
    assert result.b_as_predicted.match_rate < 1.0


def test_annotation_missing_a_series_the_other_has_is_reported_not_dropped(matcher):
    shared = _line([1, 2, 3])
    only_in_a = _line([9, 9, 9])

    result = compare_annotations(
        figure_id="f", annotation_a=[shared, only_in_a], annotation_b=[shared], matcher=matcher
    )

    # a_as_predicted: a (2 series) predicted against b (1 series) ground
    # truth -> a has a false-positive series.
    assert any(m.is_false_positive for m in result.a_as_predicted.matches)
    # b_as_predicted: b (1 series) predicted against a (2 series) ground
    # truth -> a's extra series is now an unmatched (missed) ground truth.
    assert any(m.is_detection_miss for m in result.b_as_predicted.matches)


def test_non_overlapping_x_ranges_scores_as_worst_case_in_both_directions(matcher):
    a = [_line([1.0, 2.0], x_values=(0.0, 1.0))]
    b = [_line([1.0, 2.0], x_values=(100.0, 101.0))]

    result = compare_annotations(figure_id="f", annotation_a=a, annotation_b=b, matcher=matcher)

    assert result.a_as_predicted.mean_coverage_ratio == pytest.approx(0.0)
    assert result.b_as_predicted.mean_coverage_ratio == pytest.approx(0.0)


# --- classify_ceiling_label() / require_human_ceiling() -------------------


def test_all_human_sources_classify_as_human_ceiling():
    assert (
        classify_ceiling_label([AnnotationSource.HUMAN, AnnotationSource.HUMAN])
        is CeilingLabel.HUMAN_CEILING
    )


def test_all_llm_sources_classify_as_machine_agreement():
    assert (
        classify_ceiling_label([AnnotationSource.LLM, AnnotationSource.LLM])
        is CeilingLabel.MACHINE_AGREEMENT
    )


def test_all_automated_sources_classify_as_machine_agreement():
    assert (
        classify_ceiling_label([AnnotationSource.AUTOMATED, AnnotationSource.AUTOMATED])
        is CeilingLabel.MACHINE_AGREEMENT
    )


def test_mixed_human_and_llm_sources_classify_as_mixed_source_agreement():
    assert (
        classify_ceiling_label([AnnotationSource.HUMAN, AnnotationSource.LLM])
        is CeilingLabel.MIXED_SOURCE_AGREEMENT
    )


def test_mixed_llm_and_automated_sources_classify_as_mixed_source_agreement():
    assert (
        classify_ceiling_label([AnnotationSource.LLM, AnnotationSource.AUTOMATED])
        is CeilingLabel.MIXED_SOURCE_AGREEMENT
    )


def test_classify_ceiling_label_rejects_empty_input():
    with pytest.raises(ValueError, match="at least one"):
        classify_ceiling_label([])


def test_require_human_ceiling_passes_silently_for_all_human_sources():
    require_human_ceiling([AnnotationSource.HUMAN, AnnotationSource.HUMAN, AnnotationSource.HUMAN])


def test_require_human_ceiling_refuses_a_single_llm_annotation():
    with pytest.raises(NotAHumanCeilingError):
        require_human_ceiling([AnnotationSource.HUMAN, AnnotationSource.LLM])


def test_require_human_ceiling_refuses_all_automated_sources():
    with pytest.raises(NotAHumanCeilingError):
        require_human_ceiling([AnnotationSource.AUTOMATED, AnnotationSource.AUTOMATED])


def test_require_human_ceiling_error_names_the_offending_sources():
    with pytest.raises(NotAHumanCeilingError, match="llm"):
        require_human_ceiling([AnnotationSource.HUMAN, AnnotationSource.LLM])
