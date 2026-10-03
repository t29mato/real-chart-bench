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
from real_chart_bench.domain.point_metrics import AxisFrame, PointEvaluation, evaluate_points
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
    # design §7.67: point-level metrics, one per tau in POINT_TAUS order
    points: tuple[PointEvaluation, ...] = ()


_ZERO_SCORE_EVALUATION = EvaluationResult(
    matches=(), match_rate=0.0, mean_curve_distance=1.0, mean_coverage_ratio=0.0, summary_score=0.0
)


# design §7.66: the normalizing span of the y-error never drops below this
# fraction of a linear y axis's range. Log y axes are left unfloored -- a
# floor in linear units means nothing there.
Y_SPAN_FLOOR_FRACTION = 0.05


# design §7.67: the point-level metric's thresholds, in axis-normalized units.
# 0.02 is the primary value; 0.01 and 0.05 are reported alongside.
POINT_TAUS = (0.01, 0.02, 0.05)
PRIMARY_POINT_TAU = 0.02
POINT_NORM = "euclidean"


def axis_frame_for_task(task: ExtractionTask) -> AxisFrame:
    """The axis extent a figure's points are normalized by (design §7.67)."""
    return AxisFrame(
        x_range=task.x_range, y_range=task.y_range, x_scale=task.x_scale, y_scale=task.y_scale
    )


def _point_evaluations(
    predicted: Sequence[Curve],
    item: DatasetItem,
    taus: Sequence[float],
    norm: str,
) -> tuple[PointEvaluation, ...]:
    frame = axis_frame_for_task(item.task)
    return tuple(evaluate_points(predicted, item.ground_truth, frame, tau, norm) for tau in taus)


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
    point_taus: Sequence[float] = POINT_TAUS,
    point_norm: str = POINT_NORM,
) -> list[FigureResult]:
    """Score every item. Pass ``matcher`` for one matcher throughout, or
    ``matcher_for`` to build one per task (the benchmark's own scoring uses
    ``matcher_for=matcher_for_task``).

    Every figure also gets the point-level metrics (design §7.67) at each of
    ``point_taus``, normalized by the task's axis range. A failed extraction
    scores as an empty answer there too: every ground-truth point missed."""
    if (matcher is None) == (matcher_for is None):
        raise ValueError("pass exactly one of matcher / matcher_for")
    results: list[FigureResult] = []
    for item in items:
        try:
            predicted = model.extract(item.task)
            figure_matcher = matcher if matcher is not None else matcher_for(item.task)
            evaluation = evaluate_figure(predicted, item.ground_truth, figure_matcher)
        except Exception as exc:  # noqa: BLE001 - a single bad figure must not abort the run
            results.append(
                FigureResult(
                    figure_id=item.figure_id,
                    evaluation=_ZERO_SCORE_EVALUATION,
                    error=f"{type(exc).__name__}: {exc}",
                    points=_point_evaluations([], item, point_taus, point_norm),
                )
            )
            continue
        results.append(
            FigureResult(
                figure_id=item.figure_id,
                evaluation=evaluation,
                points=_point_evaluations(predicted, item, point_taus, point_norm),
            )
        )
    return results
