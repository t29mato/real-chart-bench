"""Runs a ModelRunnerPort against a dataset of (task, ground truth) pairs
and scores each figure with the domain evaluation metric (design §7.15).

A model's extraction failing (exception, timeout, malformed output) must
not abort the whole run — it's scored as a total miss (summary_score=0,
error recorded) so a leaderboard run over hundreds of figures survives a
handful of bad ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.evaluation import EvaluationResult, evaluate_figure
from real_chart_bench.domain.matching import CurveMatcher
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


def evaluate_model_on_dataset(
    model: ModelRunnerPort,
    items: Sequence[DatasetItem],
    *,
    matcher: CurveMatcher,
) -> list[FigureResult]:
    results: list[FigureResult] = []
    for item in items:
        try:
            predicted = model.extract(item.task)
            evaluation = evaluate_figure(predicted, item.ground_truth, matcher)
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
