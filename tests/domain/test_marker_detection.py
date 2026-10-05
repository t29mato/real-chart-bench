"""Marker detector (approach A, docs/design/local-model.md): the pure parts --
image <-> network-input geometry, the train/validation split, turning peak
detections into series of pixel points, and the pixel-space point F1 used
for tuning on the validation split (never on the benchmark)."""

import pytest

from real_chart_bench.domain.marker_detection import (
    Detection,
    Letterbox,
    group_into_series,
    is_validation,
    pixel_answer,
    pixel_point_f1,
    split_key,
    suppress_duplicates,
)

# --- Letterbox ---------------------------------------------------------------


def test_letterbox_scales_long_side_and_pads_to_stride():
    lb = Letterbox.fit(1000, 500, long_side=800, stride=32)
    assert lb.scale == pytest.approx(0.8)
    assert (lb.out_w, lb.out_h) == (800, 416)  # 400 rounded up to 416


def test_letterbox_round_trip_is_identity():
    lb = Letterbox.fit(1234, 777, long_side=1024, stride=32)
    x, y = lb.to_input(321.5, 600.25)
    assert lb.to_image(x, y) == pytest.approx((321.5, 600.25))


def test_letterbox_upscales_small_images():
    lb = Letterbox.fit(400, 300, long_side=800, stride=32)
    assert lb.scale == pytest.approx(2.0)
    assert (lb.out_w, lb.out_h) == (800, 608)


def test_letterbox_rejects_empty_image():
    with pytest.raises(ValueError):
        Letterbox.fit(0, 300, long_side=800, stride=32)


# --- validation split ----------------------------------------------------------


def test_split_key_is_paper_for_real_figures_and_image_otherwise():
    assert split_key({"paper_id": 123, "image": "a.png", "source": "starrydata"}) == "paper:123"
    assert (
        split_key({"paper_id": None, "image": "a.png", "source": "plotqa"}) == "image:plotqa/a.png"
    )


def test_is_validation_is_deterministic_and_near_fraction():
    keys = [f"image:k{i}" for i in range(4000)]
    first = [is_validation(k, 0.1) for k in keys]
    assert first == [is_validation(k, 0.1) for k in keys]
    assert 0.08 < sum(first) / len(keys) < 0.12


def test_is_validation_edges():
    assert not is_validation("image:a", 0.0)
    assert is_validation("image:a", 1.0)


# --- duplicate suppression ------------------------------------------------------


def _d(x, y, score=1.0, marker="circle", emb=(0.0,)):
    return Detection(x=x, y=y, score=score, marker=marker, embedding=emb)


def test_suppress_duplicates_keeps_the_stronger_of_two_close_peaks():
    kept = suppress_duplicates([_d(10, 10, 0.5), _d(11, 10, 0.9), _d(40, 10, 0.3)], radius=3)
    assert [(d.x, d.score) for d in kept] == [(11, 0.9), (40, 0.3)]


def test_suppress_duplicates_keeps_points_exactly_radius_apart():
    kept = suppress_duplicates([_d(10, 10, 0.5), _d(13, 10, 0.9)], radius=3)
    assert len(kept) == 2


def test_suppress_duplicates_empty():
    assert suppress_duplicates([], radius=3) == []


# --- grouping into series -----------------------------------------------------


def test_group_into_series_splits_by_embedding():
    dets = [_d(i, 0, emb=(0.0, 0.0)) for i in range(3)] + [
        _d(i, 5, emb=(1.0, 1.0)) for i in range(2)
    ]
    groups = group_into_series(dets, threshold=0.5)
    assert sorted(len(g) for g in groups) == [2, 3]


def test_group_into_series_merges_within_threshold():
    dets = [_d(0, 0, emb=(0.0,)), _d(1, 0, emb=(0.2,)), _d(2, 0, emb=(0.4,))]
    # running mean 0.1 after two; 0.4 is 0.3 from it
    assert len(group_into_series(dets, threshold=0.35)) == 1


def test_group_into_series_largest_group_first():
    dets = [_d(0, 0, emb=(5.0,))] + [_d(i, 1, emb=(0.0,)) for i in range(4)]
    groups = group_into_series(dets, threshold=0.5)
    assert [len(g) for g in groups] == [4, 1]


def test_group_into_series_empty():
    assert group_into_series([], threshold=0.5) == []


# --- answer shape -------------------------------------------------------------


def test_pixel_answer_sorts_points_by_x_and_labels_series():
    groups = [[_d(30, 1), _d(10, 2), _d(20, 3)], [_d(5, 9)]]
    ans = pixel_answer(groups)
    assert ans == [
        {"label": "series_1", "x": [10, 20, 30], "y": [2, 3, 1]},
        {"label": "series_2", "x": [5], "y": [9]},
    ]


def test_pixel_answer_drops_small_groups_when_asked():
    groups = [[_d(1, 1), _d(2, 2)], [_d(5, 9)]]
    assert len(pixel_answer(groups, min_points=2)) == 1


# --- pixel F1 on the validation split ----------------------------------------


def test_pixel_point_f1_perfect():
    pts = [(0, 0), (10, 10)]
    assert pixel_point_f1(pts, pts, radius=2) == pytest.approx(1.0)


def test_pixel_point_f1_counts_each_truth_once():
    # two predictions on one true point: one hit, one false positive
    f1 = pixel_point_f1([(0, 0), (0.5, 0)], [(0, 0)], radius=2)
    assert f1 == pytest.approx(2 * 0.5 * 1.0 / 1.5)


def test_pixel_point_f1_outside_radius_is_a_miss():
    assert pixel_point_f1([(5, 0)], [(0, 0)], radius=2) == 0.0


def test_pixel_point_f1_empty_cases():
    assert pixel_point_f1([], [], radius=2) == 1.0
    assert pixel_point_f1([(0, 0)], [], radius=2) == 0.0
    assert pixel_point_f1([], [(0, 0)], radius=2) == 0.0
