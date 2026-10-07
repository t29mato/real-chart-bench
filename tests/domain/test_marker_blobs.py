"""marker_blobs (方式D v3): one point per marker from a colour mask, merging
the pieces of a marker split by a white cross and splitting blobs of
overlapping markers, with the marker size taken from the series itself."""

from __future__ import annotations

import math

import numpy as np
import pytest

from real_chart_bench.domain.marker_blobs import marker_centres, unit_marker


def blank(w=300, h=200):
    return np.zeros((h, w), bool)


def disc(m, cx, cy, r):
    h, w = m.shape
    yy, xx = np.mgrid[:h, :w]
    m[(xx - cx) ** 2 + (yy - cy) ** 2 <= r * r] = True


def near(found, truth, tol):
    """Every true centre has a found centre within tol, one to one."""
    left = list(found)
    for tx, ty in truth:
        if not left:
            return False
        k = min(range(len(left)), key=lambda i: math.hypot(left[i][0] - tx, left[i][1] - ty))
        if math.hypot(left[k][0] - tx, left[k][1] - ty) > tol:
            return False
        left.pop(k)
    return not left


ISOLATED = [(30, 30), (90, 40), (150, 30), (210, 50), (270, 40)]


def test_isolated_discs_one_point_each():
    m = blank()
    for c in ISOLATED:
        disc(m, *c, 6)
    pts, info = marker_centres(m)
    assert near(pts, ISOLATED, 0.6)
    assert info["diameter_px"] == pytest.approx(13, abs=1)
    assert info["n_split"] == 0 and info["n_merged"] == 0


def test_unit_marker_from_the_series_ignores_merged_blobs():
    sizes = [(13, 13, 133)] * 5 + [(20, 13, 230)]
    d, area = unit_marker(sizes)
    assert d == 13 and area == 133


@pytest.mark.parametrize("offset", [0.5, 0.75, 1.0])
def test_two_overlapping_discs_are_split(offset):
    m = blank()
    for c in ISOLATED:
        disc(m, *c, 6)
    a, b = (100, 140), (100 + offset * 12, 140)
    disc(m, *a, 6)
    disc(m, *b, 6)
    pts, info = marker_centres(m)
    assert near(pts, [*ISOLATED, a, b], 1.5)
    assert info["n_split"] == 1


def test_three_overlapping_in_a_row_and_a_diagonal_pair():
    m = blank()
    for c in ISOLATED:
        disc(m, *c, 6)
    row = [(60, 150), (69, 150), (78, 150)]
    diag = [(200, 140), (208, 147)]
    for c in row + diag:
        disc(m, *c, 6)
    pts, _ = marker_centres(m)
    assert near(pts, [*ISOLATED, *row, *diag], 2.0)


def test_marker_split_by_a_white_cross_is_one_point():
    m = blank()
    truth = []
    for cx, cy in ISOLATED + [(60, 140), (140, 150)]:
        disc(m, cx, cy, 8)
        m[cy - 1:cy + 2, cx - 9:cx + 10] = False
        m[cy - 9:cy + 10, cx - 1:cx + 2] = False
        truth.append((cx, cy))
    pts, info = marker_centres(m)
    assert near(pts, truth, 1.0)
    assert info["n_merged"] == len(truth)
    assert info["diameter_px"] >= 15


def test_merge_and_split_can_be_turned_off():
    m = blank()
    for cx, cy in ISOLATED:
        disc(m, cx, cy, 8)
        m[cy - 1:cy + 2, cx - 9:cx + 10] = False
        m[cy - 9:cy + 10, cx - 1:cx + 2] = False
    pts, _ = marker_centres(m, merge=False)
    assert len(pts) == 4 * len(ISOLATED)


def test_size_limits_and_a_given_diameter():
    m = blank()
    for c in ISOLATED:
        disc(m, *c, 6)
    m[150:153, 20:280] = True  # a long line: not a marker
    m[100, 100] = True  # a speck
    pts, _ = marker_centres(m, min_diameter=4, max_diameter=40)
    assert near(pts, ISOLATED, 0.6)
    disc(m, 150, 120, 6)
    disc(m, 156, 120, 6)
    pts, info = marker_centres(m, diameter=13, min_diameter=4, max_diameter=40)
    assert near(pts, [*ISOLATED, (150, 120), (156, 120)], 1.5)


def test_no_ink():
    pts, info = marker_centres(blank())
    assert pts == [] and info["diameter_px"] is None
