"""The starry-digitizer port (domain/starry_extract.py): point-for-point
equal to the original JavaScript on recorded cases, plus the rules it keeps."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from real_chart_bench.domain.starry_extract import (
    js_to_fixed_1,
    line_extract,
    mask_from_rgba,
    match_color,
    match_color_mask,
    symbol_extract,
)

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures/starry_extract_js_cases.json"


def _cases():
    return json.loads(FIXTURE.read_text())["cases"]


@pytest.mark.parametrize(
    "case", _cases(), ids=lambda c: f"{c['strategy']}-{c['width']}x{c['height']}"
)
def test_port_equals_original_javascript(case):
    rgba = np.asarray(case["rgba"], dtype=np.uint8).reshape(case["height"], case["width"], 4)
    mask = None
    if "mask_rgba" in case:
        mr = np.asarray(case["mask_rgba"], dtype=np.uint8).reshape(rgba.shape)
        mask = mask_from_rgba(mr)
    target = tuple(case["target"])
    if case["strategy"] == "Symbol Extract":
        got = symbol_extract(rgba[..., :3], target, case["distance_pct"], mask,
                             **case["params"])
    else:
        got = line_extract(rgba[..., :3], target, case["distance_pct"], mask, **case["params"])
    assert [list(p) for p in got] == case["expected"]


def test_fixture_covers_both_strategies_masks_and_points():
    cases = _cases()
    assert {c["strategy"] for c in cases} == {"Symbol Extract", "Line Extract"}
    assert any("mask_rgba" in c for c in cases)
    assert sum(len(c["expected"]) for c in cases) >= 50


@pytest.mark.parametrize(
    ("value", "expected"),
    [(2.25, 2.3), (2.35, 2.4), (0.05, 0.1), (1.0, 1.0), (3.04999, 3.0), (10.75, 10.8),
     (7.15, 7.2)],
)
def test_to_fixed_rounds_like_javascript(value, expected):
    # 2.25 and 10.75 are exact binary ties: JS rounds up, Python's round() to even
    # 2.35 is 2.35000000000000008882 and 7.15 is 7.15000000000000035527 in binary
    assert js_to_fixed_1(value) == expected


def test_match_color_threshold_is_strict():
    # one channel 255 apart: 65025 / 195075 * 100 = 33.33...
    assert match_color((255, 0, 0), (0, 0, 0), 34)
    assert not match_color((255, 0, 0), (0, 0, 0), 33)
    assert not match_color((5, 5, 5), (5, 5, 5), 0)  # "< pct": nothing matches at 0


def test_vectorised_match_equals_scalar():
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(7, 9, 3), dtype=np.uint8)
    target = (120, 30, 200)
    m = match_color_mask(img, target, 20)
    for y in range(7):
        for x in range(9):
            assert m[y, x] == match_color(tuple(int(v) for v in img[y, x]), target, 20)


def _canvas(h, w):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_symbol_extract_centroid_and_diameter_filter():
    img = _canvas(20, 30)
    img[2:5, 3:6] = (255, 0, 0)  # 9 px -> diameter 3.39
    img[10:16, 20:26] = (255, 0, 0)  # 36 px -> diameter 6.77
    pts = symbol_extract(img, (255, 0, 0), 1, None, min_diameter_px=5, max_diameter_px=100)
    assert pts == [(23.0, 13.0)]  # mean index 22.5 / 12.5, + 0.5
    pts = symbol_extract(img, (255, 0, 0), 1, None, min_diameter_px=1, max_diameter_px=5)
    assert pts == [(4.5, 3.5)]


def test_symbol_extract_is_eight_connected():
    img = _canvas(10, 10)
    for i in range(6):
        img[i, i] = (0, 0, 0)  # a diagonal: one blob
    pts = symbol_extract(img, (0, 0, 0), 1, None, min_diameter_px=1)
    assert pts == [(3.0, 3.0)]


def test_mask_restricts_extraction():
    img = _canvas(10, 20)
    img[2:6, 2:6] = (0, 0, 255)
    img[2:6, 12:16] = (0, 0, 255)
    mask = np.zeros((10, 20), bool)
    mask[:, 10:] = True
    assert symbol_extract(img, (0, 0, 255), 1, mask, min_diameter_px=1) == [(14.0, 4.0)]


def test_line_extract_samples_in_patches():
    img = _canvas(5, 30)
    img[2, :] = (0, 0, 0)  # a horizontal line 30 px long
    pts = line_extract(img, (0, 0, 0), 1, None, dx_px=4, dy_px=4)
    # column-major seeds: patches x 0..4, 5..9, ... (|dx| <= 4 from the seed)
    assert pts == [(2.5, 2.5), (7.5, 2.5), (12.5, 2.5), (17.5, 2.5), (22.5, 2.5), (27.5, 2.5)]


def test_rejects_mismatched_mask():
    with pytest.raises(ValueError):
        symbol_extract(_canvas(4, 4), (0, 0, 0), 1, np.ones((3, 4), bool))
