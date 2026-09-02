"""Inter-annotator agreement for the "human ceiling" (戦略メモ「柱B: GTの信頼性
を定量化する」): how well two *independent* digitizations of the same figure
agree with each other, scored on the exact same axis the leaderboard scores
models on.

Design note -- reusing the leaderboard metric: ``compare_annotations()``
delegates entirely to ``domain.evaluation.evaluate_figure`` with the project's
existing ``CurveMatcher``/``MetricStrategy`` (in practice
``HungarianCurveMatcher(NormalizedYDistanceMetric())``, the same objects
``usecase/evaluate_dataset.py`` uses to score models). This is deliberate: a
human-ceiling number computed with a different metric would not be
comparable to a model's summary_score, defeating the entire point of
publishing it as a ceiling.

Caveat verified against the metric's actual code (domain/metrics.py) rather
than assumed: ``NormalizedYDistanceMetric`` is *asymmetric*. It interpolates
the "predicted" curve onto the "ground truth" curve's x-grid and normalizes
the y-error by the ground truth's own y-range, so
``distance(A, B) != distance(B, A)`` in general (see
test_human_ceiling.py::test_symmetrized_score_is_order_independent...).
That asymmetry is fine, even necessary, for scoring a model against a fixed
ground truth -- but two independent digitizations of the same figure have no
natural "which one is ground truth" ordering. ``compare_annotations()``
therefore evaluates *both* directions (A predicted against B, and B
predicted against A) with the unmodified metric/matcher and averages the two
summary_scores, so the published human-ceiling figure does not silently
depend on which annotation happened to be passed first.

``classify_ceiling_label()`` / ``require_human_ceiling()`` are the structural
guard for a separate, non-negotiable project rule: an LLM-produced judgment
must never be presented as a human one. A ceiling computed from any
non-``human`` annotation_source is a *machine* agreement number, not a human
ceiling, and code must not be able to label it as one by accident -- see
``usecase/build_human_ceiling_result.py`` for the single call site that is
allowed to assign the "human_ceiling" leaderboard identity, gated on
``require_human_ceiling()`` not raising.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.evaluation import EvaluationResult, evaluate_figure
from real_chart_bench.domain.matching import CurveMatcher


class AnnotationSource(Enum):
    """Who/what produced a digitization. ``HUMAN`` is the only source a
    result may ever be labeled "human ceiling" from -- see
    ``require_human_ceiling()``."""

    HUMAN = "human"
    LLM = "llm"
    AUTOMATED = "automated"


class CeilingLabel(Enum):
    """What kind of agreement number a set of annotation sources yields."""

    HUMAN_CEILING = "human_ceiling"
    MIXED_SOURCE_AGREEMENT = "mixed_source_agreement"
    MACHINE_AGREEMENT = "machine_agreement"


class NotAHumanCeilingError(ValueError):
    """Raised by ``require_human_ceiling()`` when at least one contributing
    annotation is not ``annotation_source=human``. Never catch this to
    silently fall back to presenting the result as a human ceiling anyway --
    catch it only to choose an honestly different label (see
    ``usecase/build_human_ceiling_result.py``)."""


@dataclass(frozen=True)
class PairwiseAgreementResult:
    """Full detail behind one figure's inter-annotator agreement score."""

    figure_id: str
    a_as_predicted: EvaluationResult
    b_as_predicted: EvaluationResult
    mean_summary_score: float


def compare_annotations(
    *,
    figure_id: str,
    annotation_a: Sequence[Curve],
    annotation_b: Sequence[Curve],
    matcher: CurveMatcher,
) -> PairwiseAgreementResult:
    """Agreement between two independent digitizations of the same figure,
    using the project's normal figure-scoring machinery in both directions
    (see module docstring for why symmetrizing is necessary)."""
    a_as_predicted = evaluate_figure(
        predicted=annotation_a, ground_truth=annotation_b, matcher=matcher
    )
    b_as_predicted = evaluate_figure(
        predicted=annotation_b, ground_truth=annotation_a, matcher=matcher
    )
    mean_summary_score = (a_as_predicted.summary_score + b_as_predicted.summary_score) / 2
    return PairwiseAgreementResult(
        figure_id=figure_id,
        a_as_predicted=a_as_predicted,
        b_as_predicted=b_as_predicted,
        mean_summary_score=mean_summary_score,
    )


def classify_ceiling_label(sources: Iterable[AnnotationSource]) -> CeilingLabel:
    """Pure classification, never raises for a non-human mix -- it only
    raises when there is nothing to classify at all (that's a caller bug,
    not a labeling decision)."""
    sources = list(sources)
    if not sources:
        raise ValueError("classify_ceiling_label() requires at least one annotation source")
    if all(s is AnnotationSource.HUMAN for s in sources):
        return CeilingLabel.HUMAN_CEILING
    if all(s is sources[0] for s in sources):
        return CeilingLabel.MACHINE_AGREEMENT
    return CeilingLabel.MIXED_SOURCE_AGREEMENT


def require_human_ceiling(sources: Iterable[AnnotationSource]) -> None:
    """Refusal gate: raises ``NotAHumanCeilingError`` unless every source is
    ``annotation_source=human``. Call this immediately before presenting or
    labeling any result as "human ceiling" -- do not gate on
    ``classify_ceiling_label()`` directly for that decision, so there is
    exactly one code path that can grant the "human ceiling" identity."""
    sources = list(sources)
    label = classify_ceiling_label(sources)
    if label is not CeilingLabel.HUMAN_CEILING:
        non_human = sorted({s.value for s in sources if s is not AnnotationSource.HUMAN})
        raise NotAHumanCeilingError(
            "Refusing to present this result as a human ceiling: contributing "
            f"annotation_source value(s) include non-human source(s) {non_human}. "
            f"This computes as {label.value}, not human_ceiling -- label it "
            "accordingly instead of discarding or renaming it as human."
        )
