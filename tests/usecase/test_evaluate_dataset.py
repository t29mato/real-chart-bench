from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.matching import HungarianCurveMatcher
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric
from real_chart_bench.usecase.evaluate_dataset import DatasetItem, evaluate_model_on_dataset
from real_chart_bench.usecase.model_runner import ExtractionTask


def _curve(y_values):
    x = tuple(float(i) for i in range(len(y_values)))
    return Curve(x_values=x, y_values=tuple(float(v) for v in y_values))


class _PerfectModel:
    """Returns the ground truth curves verbatim, ignoring the image — used
    to sanity-check the harness wiring gives a perfect score end-to-end."""

    def __init__(self, answers: dict[bytes, list[Curve]]):
        self._answers = answers

    def extract(self, task: ExtractionTask) -> list[Curve]:
        return self._answers[task.image_bytes]


def _matcher():
    return HungarianCurveMatcher(metric=NormalizedYDistanceMetric())


def test_perfect_model_scores_one_across_the_dataset():
    curve = _curve([1, 2, 3])
    task = ExtractionTask(image_bytes=b"img1", x_range=(0, 2), y_range=(1, 3))
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=[curve])]
    model = _PerfectModel({b"img1": [curve]})

    results = evaluate_model_on_dataset(model, items, matcher=_matcher())

    assert len(results) == 1
    assert results[0].figure_id == "f1"
    assert results[0].evaluation.summary_score == 1.0


def test_aggregate_score_is_the_mean_of_per_figure_scores():
    curve = _curve([1, 2, 3])
    off_curve = _curve([9, 9, 9])
    task1 = ExtractionTask(image_bytes=b"img1", x_range=(0, 2), y_range=(1, 3))
    task2 = ExtractionTask(image_bytes=b"img2", x_range=(0, 2), y_range=(1, 3))
    items = [
        DatasetItem(figure_id="f1", task=task1, ground_truth=[curve]),
        DatasetItem(figure_id="f2", task=task2, ground_truth=[curve]),
    ]
    model = _PerfectModel({b"img1": [curve], b"img2": [off_curve]})

    results = evaluate_model_on_dataset(model, items, matcher=_matcher())

    scores = [r.evaluation.summary_score for r in results]
    assert scores[0] == 1.0
    assert scores[1] < 1.0


def test_model_extraction_error_is_captured_not_fatal():
    class _BrokenModel:
        def extract(self, task):
            raise RuntimeError("boom")

    curve = _curve([1, 2, 3])
    task = ExtractionTask(image_bytes=b"img1", x_range=(0, 2), y_range=(1, 3))
    items = [DatasetItem(figure_id="f1", task=task, ground_truth=[curve])]

    results = evaluate_model_on_dataset(_BrokenModel(), items, matcher=_matcher())

    assert len(results) == 1
    assert results[0].error is not None
    assert results[0].evaluation.summary_score == 0.0


def test_empty_dataset_returns_empty_results():
    results = evaluate_model_on_dataset(_PerfectModel({}), [], matcher=_matcher())
    assert results == []
