"""The synthetic generator's labels match what it renders
(scripts/train/gen_synth_materials.py, docs/design/local-model.md).

For isolated point-symmetric markers (no line, no error bar, no grid, no
neighbour within reach), the ink centroid of the rendered marker must sit on
the labelled pixel point. Also: every label is valid and self-consistent with
its own ticks.
"""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from real_chart_bench.domain.training_data import max_axis_residual_px, validate_label

pytest.importorskip("matplotlib")

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "train" / "gen_synth_materials.py"
SYMMETRIC = {"o", "s", "D", "d", "x", "+", "h"}


@pytest.fixture(scope="module")
def gen():
    spec = importlib.util.spec_from_file_location("gen_synth_materials", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def rendered(gen):
    return [gen.render(i) for i in range(40)]


@pytest.fixture(scope="module")
def plain(gen):
    """Images with markers only: no lines, error bars, grid or inset."""
    force = {"style": "markers", "errorbars": False, "grid": False, "inset": False}
    return [gen.render(i, force=force) for i in range(1000, 1030)]


def test_every_label_is_valid_and_agrees_with_its_ticks(rendered):
    with_axes = 0
    for _, label, img, _ in rendered:
        assert validate_label(label) == []
        assert img.size == (label["width"], label["height"])
        r = max_axis_residual_px(label)
        if r is not None:
            with_axes += 1
            assert r < 0.01
    assert with_axes >= 34


def _isolated_points(label):
    ms = label["synth"]["marker_size_px"]
    pts = [(p, s["mpl_marker"]) for s in label["series"] for p in s["points_px"]]
    allp = np.array([p for p, _ in pts])
    x0, y0, x1, y1 = label["plot_bbox"]
    legend = label["synth"]["legend_bbox"]
    reach = ms + 6
    for p, mk in pts:
        if mk not in SYMMETRIC:
            continue
        d = np.hypot(*(allp - p).T)
        if np.sum(d < 2 * reach) > 1:
            continue
        # the ink window (radius `reach`) must not touch the axes frame or ticks
        if not (x0 + 2 * reach < p[0] < x1 - 2 * reach and y0 + 2 * reach < p[1] < y1 - 2 * reach):
            continue
        if legend and (legend[0] - 2 * reach < p[0] < legend[2] + 2 * reach
                       and legend[1] - 2 * reach < p[1] < legend[3] + 2 * reach):
            continue
        yield p, reach


def test_rendered_marker_centres_match_the_labelled_pixels(plain):
    errors = []
    for _, label, img, _ in plain:  # img is the lossless render (JPEG is applied on save)
        ink = 255.0 - np.asarray(img.convert("L"), dtype=float)
        for (px, py), reach in _isolated_points(label):
            r = int(np.ceil(reach))
            cx, cy = int(px), int(py)
            win = ink[cy - r: cy + r + 1, cx - r: cx + r + 1]
            ys, xs = np.mgrid[cy - r: cy + r + 1, cx - r: cx + r + 1]
            w = win.sum()
            assert w > 0, f"no ink at {label['image']} {px, py}"
            errors.append(np.hypot((win * (xs + 0.5)).sum() / w - px,
                                   (win * (ys + 0.5)).sum() / w - py))
    assert len(errors) >= 20
    # supersampled drawing: markers land within a fraction of a pixel
    assert np.median(errors) < 0.35, np.median(errors)
    assert max(errors) < 1.0, sorted(errors)[-5:]
