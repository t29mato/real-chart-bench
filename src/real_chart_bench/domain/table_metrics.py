"""Chart-as-table metrics (design §7.85): RMS and NMS, reported *alongside*
the primary point-level metric (``point_metrics.py``), never instead of it.

RMS -- Relative Mapping Similarity, DePlot (Liu et al., 2023) -- is the de
facto standard for chart-to-table, so a row here can be read next to prior
work. It views a chart as a set of ``(row header, column header, value)``
triples: the row header is the x value, the column header is the series name.
Predicted and ground-truth triples are matched one to one by minimum-cost
bipartite assignment on the *header* distance, and a matched pair scores

    header score x value score

with ``header score = ANLS(key_g, key_p)`` (1 - normalized Levenshtein
distance, thresholded to 0 beyond ``text_theta``) and
``value score = 1 - min(1, |g - p| / |g|)``, thresholded to 0 beyond
``number_theta``. Precision divides the summed score by the number of
predicted triples, recall by the number of ground-truth triples. The metric is
threshold-free in the sense that matters here: it is graded, not binary.

Three numbers are reported, which differ in exactly two independent knobs --
``header_term`` and ``value_norm``:

1. ``rms_f1`` -- faithful DePlot: header term on, relative denominator.
2. ``rms_f1_value_only`` -- header term off, relative denominator. The project
   leads with this one. ``series_label`` is Starrydata's *database* name; for
   some figures it is the string the legend prints and for others it is not,
   so scoring the header text would reward only the figures whose legend
   happens to agree with the database (owner decision, §7.85). Neutralizing
   the term -- header score = 1 for every matched pair -- leaves the numeric
   accuracy alone. The *pairing* is still header-driven: a predicted triple is
   still compared against the ground-truth triple whose (x, series) label is
   closest to it, so this is RMS with one term removed, not a different
   matching.
3. ``nms_f1`` -- header term off, axis-range denominator. RMS normalizes the
   value error by ``|g|``, which is wrong for charts (paper §3.5.2/§3.5.4);
   PaperUnPlot's NMS divides by the axis range instead and handles log axes.
   This variant makes that one substitution -- same matching, same
   ``number_theta`` -- so the difference between (2) and (3) is the
   denominator and nothing else. The defect is *not* fixed in (1): RMS must
   stay faithful to the published definition to remain comparable.

Deliberate deviations from the reference implementation
(``scripts/eval/repro/third_party/deplot_metrics.py``), each for a reason the
published definition does not speak to:

- **Duplicate headers stay separate triples.** DePlot collapses the table into
  a ``dict`` keyed by the header, so two points with the same x in the same
  series silently become one. The definition is a *set of triples* matched one
  to one; collapsing would make the denominators depend on that implementation
  detail and would drop ground-truth points. A chart's table is treated as a
  multiset here.
- **Ties in the header distance are broken toward the better entry score.**
  With duplicate or missing headers the header cost matrix has ties, and which
  optimal assignment the solver returns would otherwise depend on the input
  order -- the metric must not. The tie-break weight is ~1e-9, far below any
  reported digit, so it cannot change a non-tied assignment.
- **A ground-truth value of exactly 0.** ``|g| = 0`` leaves RMS's relative
  error undefined. DePlot falls through to comparing the two value *cells* as
  text; this pipeline compares parsed floats, and ANLS over rendered floats
  would depend on the formatting, so the numeric half of that branch is kept:
  value score 1 when the prediction is exactly 0, else 0. ``nms_f1`` has no
  such hole -- it divides by the axis range -- which is one more reason the
  project does not lead with ``rms_f1``.
- **Two empty strings compare equal.** ANLS divides by the longer length and
  so raises on ``("", "")`` upstream; here it is 1.

A missing series label becomes an empty column header: the key degrades to the
x value alone. Both sides unlabeled therefore still match; one side labeled and
the other not is usually beyond ``text_theta`` and so scores 0 in (1) -- again
what (2) and (3) exist to avoid.

Pure: numpy, scipy and editdistance only, like ``point_metrics.py``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import editdistance
import numpy as np
from scipy.optimize import linear_sum_assignment

from real_chart_bench.domain.curve import Curve, ScaleType
from real_chart_bench.domain.point_metrics import AxisFrame

# DePlot's defaults (Liu et al., 2023; deplot_metrics.py)
TEXT_THETA = 0.5
NUMBER_THETA = 0.1

# RMS: |g - p| / |g|. NMS: |g - p| / axis range, log axes in log10 space.
VALUE_NORM_RELATIVE = "relative"
VALUE_NORM_AXIS_RANGE = "axis_range"
VALUE_NORMS = (VALUE_NORM_RELATIVE, VALUE_NORM_AXIS_RANGE)

# only ever decides between assignments whose header cost is equal
_TIE_BREAK = 1e-9

TABLE_METRIC_LABEL = (
    "chart-as-table similarity over (x, series, value) triples, one-to-one "
    "assignment on the header distance; rms_f1 = faithful DePlot RMS (Liu et "
    "al. 2023, header ANLS x 1 - min(1, |g-p|/|g|), text theta 0.5, number "
    "theta 0.1); rms_f1_value_only = the same with the header term "
    "neutralized (series_label is the database's name, not necessarily the "
    "printed legend); nms_f1 = value_only with the error divided by the axis "
    "range instead of |g| (PaperUnPlot's NMS, log axes in log10). Secondary "
    "metrics: the primary metric stays macro point_f1 (design 7.67/7.85)"
)


def format_cell(value: float) -> str:
    """The value as a chart's table would print it: a whole number without a
    decimal point, otherwise the shortest round-tripping decimal. RMS compares
    headers as *text*, so x values need one canonical spelling."""
    if not math.isfinite(value):
        return repr(value)
    if value == int(value) and abs(value) < 1e16:
        return str(int(value))
    return repr(value)


@dataclass(frozen=True)
class TableEntry:
    """One ``(row header, column header, value)`` triple of the chart's table."""

    row_header: str
    column_header: str
    value: float

    @property
    def key(self) -> str:
        """DePlot's datapoint key: the two headers joined by a single space."""
        return f"{self.row_header} {self.column_header}"


def entries_from_curves(curves: Sequence[Curve]) -> list[TableEntry]:
    """Every point of every curve as one triple. The column header is the
    series label, stripped and lower-cased (DePlot lower-cases the whole
    table); a missing label becomes an empty column header."""
    entries: list[TableEntry] = []
    for curve in curves:
        label = str(curve.series_label or "").strip().lower()
        for x, y in zip(curve.x_values, curve.y_values, strict=True):
            entries.append(TableEntry(row_header=format_cell(x), column_header=label, value=y))
    return entries


@dataclass(frozen=True)
class TableEvaluation:
    """One figure under one (header_term, value_norm) setting."""

    precision: float
    recall: float
    f1: float
    n_predicted: int
    n_ground_truth: int
    # the summed similarity of the matched triples, so a dataset-level micro
    # aggregate can be rebuilt from the rows alone
    score: float


@dataclass(frozen=True)
class TableMetrics:
    """The three reported variants for one figure."""

    rms: TableEvaluation
    rms_value_only: TableEvaluation
    nms: TableEvaluation


def anls(target: str, prediction: str, theta: float = TEXT_THETA) -> float:
    """1 - normalized Levenshtein distance, 0 once that distance reaches
    ``theta`` (DocVQA's ANLS, as DePlot uses it). Two empty strings are 1."""
    longest = max(len(target), len(prediction))
    if longest == 0:
        return 1.0
    normalized = editdistance.eval(target, prediction) / longest
    return 1.0 - normalized if normalized < theta else 0.0


def _axis_span(frame: AxisFrame) -> float:
    lo, hi = frame.y_range
    if frame.y_scale is ScaleType.LOG:
        lo, hi = math.log10(lo), math.log10(hi)
    return abs(hi - lo)


def _on_axis(value: float, scale: ScaleType) -> float:
    """The value in the space the axis measures distance in; NaN when it
    cannot be placed there (non-finite, or non-positive on a log axis)."""
    if not math.isfinite(value):
        return math.nan
    if scale is ScaleType.LOG:
        return math.log10(value) if value > 0 else math.nan
    return value


def _relative_distance(truth: float, prediction: float) -> float:
    """RMS's ``min(1, |g - p| / |g|)``."""
    if not (math.isfinite(truth) and math.isfinite(prediction)):
        return 1.0
    if truth == 0.0:
        # see the module docstring: the numeric half of DePlot's fallback
        return 0.0 if prediction == 0.0 else 1.0
    return min(abs((truth - prediction) / truth), 1.0)


def _axis_range_distance(truth: float, prediction: float, frame: AxisFrame) -> float:
    """NMS's ``min(1, |g - p| / axis range)``, log axes in log10 space."""
    g = _on_axis(truth, frame.y_scale)
    p = _on_axis(prediction, frame.y_scale)
    if math.isnan(g) or math.isnan(p):
        return 1.0
    return min(abs(g - p) / _axis_span(frame), 1.0)


def _value_score(
    truth: float,
    prediction: float,
    value_norm: str,
    frame: AxisFrame | None,
    number_theta: float,
) -> float:
    if value_norm == VALUE_NORM_RELATIVE:
        distance = _relative_distance(truth, prediction)
    else:
        assert frame is not None  # checked by evaluate_table
        distance = _axis_range_distance(truth, prediction, frame)
    return 1.0 - distance if distance < number_theta else 0.0


def evaluate_table(
    predicted: Sequence[TableEntry],
    ground_truth: Sequence[TableEntry],
    *,
    header_term: bool = True,
    value_norm: str = VALUE_NORM_RELATIVE,
    frame: AxisFrame | None = None,
    text_theta: float = TEXT_THETA,
    number_theta: float = NUMBER_THETA,
) -> TableEvaluation:
    """RMS precision / recall / F1 for one figure's table.

    ``header_term=False`` neutralizes the header score (1 for every matched
    pair) without changing the matching. ``value_norm=VALUE_NORM_AXIS_RANGE``
    divides the value error by ``frame``'s y-axis range instead of by ``|g|``.

    Empty sides follow DePlot: nothing predicted -> precision 1, recall 0
    (an empty answer must not raise a macro-averaged precision); nothing in
    the ground truth -> precision 0, recall 1; both empty -> 1 / 1 / 1. A
    figure where no triple scores at all is 0 / 0 / 0.
    """
    if value_norm not in VALUE_NORMS:
        raise ValueError(f"value_norm must be one of {VALUE_NORMS}, got {value_norm!r}")
    if value_norm == VALUE_NORM_AXIS_RANGE and frame is None:
        raise ValueError("value_norm 'axis_range' needs a frame")

    n_predicted, n_ground_truth = len(predicted), len(ground_truth)
    if n_predicted == 0 or n_ground_truth == 0:
        both_empty = n_predicted == 0 and n_ground_truth == 0
        return TableEvaluation(
            precision=1.0 if n_predicted == 0 else 0.0,
            recall=1.0 if n_ground_truth == 0 else 0.0,
            f1=1.0 if both_empty else 0.0,
            n_predicted=n_predicted,
            n_ground_truth=n_ground_truth,
            score=0.0,
        )

    header = np.array([[anls(g.key, p.key, text_theta) for p in predicted] for g in ground_truth])
    value = np.array(
        [
            [_value_score(g.value, p.value, value_norm, frame, number_theta) for p in predicted]
            for g in ground_truth
        ]
    )
    entry = value * (header if header_term else 1.0)
    # DePlot assigns on the header distance alone; the tie-break only orders
    # assignments whose header cost is equal (see the module docstring)
    rows, cols = linear_sum_assignment((1.0 - header) - _TIE_BREAK * entry)
    score = float(entry[rows, cols].sum())
    if score <= 0.0:
        return TableEvaluation(0.0, 0.0, 0.0, n_predicted, n_ground_truth, 0.0)
    precision = score / n_predicted
    recall = score / n_ground_truth
    return TableEvaluation(
        precision=precision,
        recall=recall,
        f1=2 * precision * recall / (precision + recall),
        n_predicted=n_predicted,
        n_ground_truth=n_ground_truth,
        score=score,
    )


def evaluate_chart_table(
    predicted: Sequence[Curve],
    ground_truth: Sequence[Curve],
    frame: AxisFrame,
) -> TableMetrics:
    """The three reported variants for one figure (design §7.85).

    ``frame`` is the figure's axis extent -- the same frame the point metric
    normalizes by -- and is used only by ``nms``. Unlike ``point_metrics``,
    every figure is scored: RMS matches on headers, so a dense-marker figure
    (§7.72) is no more ambiguous here than a sparse one.
    """
    pred = entries_from_curves(predicted)
    truth = entries_from_curves(ground_truth)
    return TableMetrics(
        rms=evaluate_table(pred, truth),
        rms_value_only=evaluate_table(pred, truth, header_term=False),
        nms=evaluate_table(
            pred, truth, header_term=False, value_norm=VALUE_NORM_AXIS_RANGE, frame=frame
        ),
    )
