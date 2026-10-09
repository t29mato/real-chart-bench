"""MarkerNet v2 post-processing (docs/design/local-model.md「方式A v2」): the
predicted marker size widens same-series duplicate suppression, and the
predicted plot / legend regions drop peaks outside the plot or in a legend."""

from real_chart_bench.domain.marker_detection import (
    Detection,
    PostConfig,
    gate_by_region,
    postprocess,
    suppress_by_size,
)


def _d(x, y, score=0.9, emb=(0.0,), size=None, plot=None, legend=None):
    return Detection(x=x, y=y, score=score, marker="circle", embedding=emb,
                     size=size, plot=plot, legend=legend)


def test_v1_detections_still_construct_without_the_new_fields():
    d = Detection(x=1, y=2, score=0.5, marker="circle", embedding=(0.0,))
    assert d.size is None and d.plot is None and d.legend is None


def test_large_marker_extra_peaks_are_suppressed_by_its_size():
    big = [_d(100, 100, 0.9, size=20), _d(106, 100, 0.8, size=20), _d(100, 107, 0.7, size=20)]
    assert suppress_by_size(big, factor=0.5, embed_threshold=0.5) == [big[0]]


def test_small_markers_of_one_series_further_than_their_size_stay():
    small = [_d(100, 100, size=4), _d(106, 100, size=4)]
    assert suppress_by_size(small, factor=0.5, embed_threshold=0.5) == small


def test_coincident_markers_of_another_series_stay():
    a, b = _d(100, 100, 0.9, emb=(0.0,), size=20), _d(103, 100, 0.8, emb=(3.0,), size=20)
    assert suppress_by_size([a, b], factor=0.5, embed_threshold=0.5) == [a, b]


def test_unknown_size_is_never_suppressed_by_size():
    a, b = _d(100, 100, 0.9), _d(101, 100, 0.8)
    assert suppress_by_size([a, b], factor=0.5, embed_threshold=0.5) == [a, b]


def test_suppression_uses_the_larger_of_the_two_sizes_and_keeps_input_order():
    weak_big, strong_small = _d(100, 100, 0.5, size=20), _d(108, 100, 0.9, size=2)
    assert suppress_by_size([weak_big, strong_small], factor=0.5, embed_threshold=0.5) == [
        strong_small]


def test_region_gate_drops_outside_plot_and_inside_legend():
    inside = _d(1, 1, plot=0.9, legend=0.1)
    outside = _d(2, 2, plot=0.2, legend=0.0)
    in_legend = _d(3, 3, plot=0.9, legend=0.8)
    unknown = _d(4, 4)
    kept = gate_by_region([inside, outside, in_legend, unknown], plot_min=0.5, legend_max=0.5)
    assert kept == [inside, unknown]


def test_region_gate_with_none_thresholds_keeps_everything():
    ds = [_d(1, 1, plot=0.0, legend=1.0)]
    assert gate_by_region(ds, plot_min=None, legend_max=None) == ds


def test_postprocess_applies_v2_steps_only_when_configured():
    dets = [_d(100, 100, 0.9, size=20, plot=0.9), _d(106, 100, 0.8, size=20, plot=0.9),
            _d(300, 300, 0.9, size=20, plot=0.9), _d(5, 5, 0.9, size=20, plot=0.1)]
    v1 = postprocess(dets, (1000, 1000), None, PostConfig(threshold=0.3, group_threshold=0.5,
                                                          min_points=1))
    v2 = postprocess(dets, (1000, 1000), None, PostConfig(
        threshold=0.3, group_threshold=0.5, min_points=1, size_factor=0.5, plot_min=0.5))
    assert sum(len(s["x"]) for s in v1) == 4
    assert sum(len(s["x"]) for s in v2) == 2
