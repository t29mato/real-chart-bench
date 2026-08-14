"""§3.3 boundary case: x値がカテゴリ変数(数値でない)の場合の距離定義.

Decision (documented, not silently dropped): Curve.x_values is numeric-only
(see design §4.2 class diagram: ``List~float~ xValues``). Categorical x-axes
(e.g. discrete sample names) are out of v0 scope per design §0 ("非XY系グラフ
...将来拡張候補"); callers that need to compare categorical-x series must
ordinal-encode the categories to floats (0.0, 1.0, 2.0, ...) upstream of the
domain layer — the metric then treats it like any other numeric axis. This
test pins down that this workaround produces sane, deterministic results, so
the scope decision doesn't silently rot.
"""

import pytest

from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric


def test_ordinal_encoded_categorical_x_axis_behaves_like_numeric_axis():
    categories = ["low", "medium", "high"]
    encode = {label: float(i) for i, label in enumerate(categories)}

    ground_truth = Curve(
        x_values=tuple(encode[c] for c in categories),
        y_values=(1.0, 2.0, 3.0),
    )
    predicted = Curve(
        x_values=tuple(encode[c] for c in categories),
        y_values=(1.0, 2.0, 3.0),
    )

    metric = NormalizedYDistanceMetric()
    assert metric.distance(predicted, ground_truth) == pytest.approx(0.0)


def test_curve_rejects_non_numeric_x_values_explicitly():
    with pytest.raises((TypeError, ValueError)):
        Curve(x_values=("low", "medium", "high"), y_values=(1.0, 2.0, 3.0))  # type: ignore[arg-type]
