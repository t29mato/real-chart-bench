"""Runs a ModelRunnerPort against a dataset of (task, ground truth) pairs
and scores each figure with the domain evaluation metric (design §7.15).

A model's extraction failing (exception, timeout, malformed output) must
not abort the whole run — it's scored as a total miss (summary_score=0,
error recorded) so a leaderboard run over hundreds of figures survives a
handful of bad ones.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from real_chart_bench.domain.curve import Curve, ScaleType
from real_chart_bench.domain.evaluation import EvaluationResult, evaluate_figure
from real_chart_bench.domain.matching import CurveMatcher, HungarianCurveMatcher
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric
from real_chart_bench.usecase.model_runner import ExtractionTask, ModelRunnerPort


@dataclass(frozen=True)
class DatasetItem:
    figure_id: str
    task: ExtractionTask
    ground_truth: list[Curve]


@dataclass(frozen=True)
class FigureResult:
    figure_id: str
    evaluation: EvaluationResult
    error: str | None = None


_ZERO_SCORE_EVALUATION = EvaluationResult(
    matches=(), match_rate=0.0, mean_curve_distance=1.0, mean_coverage_ratio=0.0, summary_score=0.0
)


# design §7.66: the normalizing span of the y-error never drops below this
# fraction of a linear y axis's range. Log y axes are left unfloored -- a
# floor in linear units means nothing there.
Y_SPAN_FLOOR_FRACTION = 0.05


def matcher_for_task(task: ExtractionTask) -> HungarianCurveMatcher:
    """The benchmark's scoring matcher for one figure."""
    if task.y_scale is ScaleType.LOG:
        floor = 0.0
    else:
        floor = Y_SPAN_FLOOR_FRACTION * abs(task.y_range[1] - task.y_range[0])
    return HungarianCurveMatcher(metric=NormalizedYDistanceMetric(min_y_span=floor))


def evaluate_model_on_dataset(
    model: ModelRunnerPort,
    items: Sequence[DatasetItem],
    *,
    matcher: CurveMatcher | None = None,
    matcher_for: Callable[[ExtractionTask], CurveMatcher] | None = None,
) -> list[FigureResult]:
    """Score every item. Pass ``matcher`` for one matcher throughout, or
    ``matcher_for`` to build one per task (the benchmark's own scoring uses
    ``matcher_for=matcher_for_task``)."""
    if (matcher is None) == (matcher_for is None):
        raise ValueError("pass exactly one of matcher / matcher_for")
    results: list[FigureResult] = []
    for item in items:
        try:
            predicted = model.extract(item.task)
            figure_matcher = matcher if matcher is not None else matcher_for(item.task)
            evaluation = evaluate_figure(predicted, item.ground_truth, figure_matcher)
            results.append(FigureResult(figure_id=item.figure_id, evaluation=evaluation))
        except Exception as exc:  # noqa: BLE001 - a single bad figure must not abort the run
            results.append(
                FigureResult(
                    figure_id=item.figure_id,
                    evaluation=_ZERO_SCORE_EVALUATION,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
    return results
