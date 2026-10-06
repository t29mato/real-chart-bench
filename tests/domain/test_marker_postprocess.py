"""方式C (docs/design/local-model.md): detector post-processing -- keeping
detections inside the detected plot frame, and a duplicate-suppression
radius that grows with the figure's marker size. Settings are chosen on the
training data's validation split, never on the benchmark."""

from real_chart_bench.domain.marker_detection import (
    Detection,
    duplicate_radius,
    inside_frame,
)


def _d(x, y, s=0.9):
    return Detection(x=x, y=y, score=s, marker="circle", embedding=(0.0,))


def test_inside_frame_keeps_points_in_the_frame_with_margin():
    ds = [_d(50, 50), _d(5, 50), _d(103, 50), _d(50, 120)]
    kept = inside_frame(ds, (10, 10, 100, 100), margin_fraction=0.05)
    # margin 0.05 * 90 = 4.5 px: x=103 stays, x=5 is 5 px outside -> dropped
    assert [(d.x, d.y) for d in kept] == [(50, 50), (103, 50)]


def test_away_from_frame_edges_drops_peaks_on_the_frame_lines():
    from real_chart_bench.domain.marker_detection import away_from_frame_edges

    frame = (0, 0, 200, 100)
    ds = [_d(100, 1), _d(199, 50), _d(100, 50), _d(3, 50), _d(100, 97), _d(5, 50)]
    kept = away_from_frame_edges(ds, frame, band_fraction=0.02)
    # band = 0.02 * min(200, 100) = 2 px: inward tick marks on the frame lines go
    assert [(d.x, d.y) for d in kept] == [(100, 50), (3, 50), (100, 97), (5, 50)]
    kept = away_from_frame_edges(ds, frame, band_fraction=0.04)
    assert [(d.x, d.y) for d in kept] == [(100, 50), (5, 50)]


def test_away_from_frame_edges_without_frame_keeps_everything():
    from real_chart_bench.domain.marker_detection import away_from_frame_edges

    ds = [_d(1, 1)]
    assert away_from_frame_edges(ds, None, band_fraction=0.05) == ds


def test_inside_frame_without_frame_keeps_everything():
    ds = [_d(1, 1)]
    assert inside_frame(ds, None, margin_fraction=0.0) == ds


def test_duplicate_radius_floor_when_no_size():
    assert duplicate_radius((1000, 500), marker_px=None, frac=0.004, size_factor=0.6) == 4.0


def test_duplicate_radius_grows_with_marker_size():
    assert duplicate_radius((1000, 500), marker_px=20.0, frac=0.004, size_factor=0.6) == 12.0


def test_duplicate_radius_never_below_floor():
    assert duplicate_radius((100, 50), marker_px=1.0, frac=0.004, size_factor=0.6) == 1.5


def _e(x, y, s, emb):
    return Detection(x=x, y=y, score=s, marker="circle", embedding=emb)


def test_same_series_duplicates_are_dropped_within_radius():
    from real_chart_bench.domain.marker_detection import suppress_same_series_duplicates

    ds = [_e(0, 0, 0.9, (0.0, 0.0)), _e(6, 0, 0.5, (0.1, 0.0)), _e(20, 0, 0.8, (0.0, 0.1))]
    kept = suppress_same_series_duplicates(ds, radius=10, embed_threshold=0.5)
    assert [(d.x, d.y) for d in kept] == [(0, 0), (20, 0)]


def test_coincident_markers_of_other_series_survive():
    from real_chart_bench.domain.marker_detection import suppress_same_series_duplicates

    ds = [_e(0, 0, 0.9, (0.0, 0.0)), _e(3, 0, 0.6, (2.0, 0.0))]
    kept = suppress_same_series_duplicates(ds, radius=10, embed_threshold=0.5)
    assert len(kept) == 2


def test_same_series_suppression_keeps_input_order():
    from real_chart_bench.domain.marker_detection import suppress_same_series_duplicates

    ds = [_e(5, 0, 0.3, (0.0,)), _e(0, 0, 0.9, (0.0,))]
    assert suppress_same_series_duplicates(ds, radius=10, embed_threshold=1.0) == [ds[1]]


def test_postprocess_chains_threshold_frame_suppression_and_grouping():
    from real_chart_bench.domain.marker_detection import PostConfig, postprocess

    ds = [
        _e(50, 50, 0.9, (0.0,)), _e(54, 50, 0.5, (0.1,)),  # one big marker, two peaks
        _e(80, 60, 0.9, (0.0,)),
        _e(5, 50, 0.9, (0.0,)),  # a tick label left of the frame
        _e(60, 70, 0.2, (0.0,)),  # below the peak threshold
    ]
    cfg = PostConfig(threshold=0.4, group_threshold=0.5, dup_frac=0.004,
                     same_series_frac=0.08, embed_gate=0.5, frame_margin=0.02)
    ans = postprocess(ds, (100, 100), (10, 10, 100, 100), cfg)
    assert ans == [{"label": "series_1", "x": [50, 80], "y": [50, 60]}]


def test_postprocess_without_frame_or_extra_suppression_is_method_a():
    from real_chart_bench.domain.marker_detection import PostConfig, postprocess

    ds = [_e(50, 50, 0.9, (0.0,)), _e(54, 50, 0.5, (0.1,)), _e(5, 50, 0.9, (0.0,))]
    cfg = PostConfig(threshold=0.4, group_threshold=0.5)
    ans = postprocess(ds, (100, 100), None, cfg)
    assert ans == [{"label": "series_1", "x": [5, 50, 54], "y": [50, 50, 50]}]
