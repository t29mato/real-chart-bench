"""The tick-calibrated LineFormer row (design §7.66) maps LineFormer's pixels
through the owner-reviewed tick-mark positions instead of the full image
frame. Those positions are only trusted when they still describe the figure
as scored: same image, tick labels equal to the registry's ranges, not
marked stale."""

import json

from real_chart_bench.adapter.tick_plot_areas import load_tick_plot_areas
from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.verified_pairing import VerifiedPairing


def _pairing(**overrides):
    fields = dict(
        paper_id="1",
        figure_id="2",
        image_path="data/verified_pairs/crops/1/a.png",
        x_range=(300.0, 900.0),
        y_range=(0.0, 40.0),
        x_scale=ScaleType.LINEAR,
        y_scale=ScaleType.LINEAR,
    )
    fields.update(overrides)
    return fields


def _entry(**overrides):
    entry = {
        "paper_id": "1",
        "figure_id": "2",
        "image_path": "data/verified_pairs/crops/1/a.png",
        "x_min_label": 300,
        "x_max_label": 900,
        "y_min_label": 0,
        "y_max_label": 40,
        "pixel_bbox_mean": {"x_min_px": 10, "x_max_px": 410, "y_min_px": 300, "y_max_px": 20},
    }
    entry.update(overrides)
    return entry


def _load(tmp_path, entry, pairing_fields):
    path = tmp_path / "axis.json"
    path.write_text(json.dumps([{"_meta": {}}, entry]))
    pairing = _make(pairing_fields)
    return load_tick_plot_areas(path, [pairing])


def _make(fields):
    # VerifiedPairing has many required fields; build through the registry
    # parser's own defaults would couple this test to the file format, so
    # use a minimal stand-in exposing what the loader reads.
    class _P:
        pass

    p = _P()
    for k, v in fields.items():
        setattr(p, k, v)
    return p


def test_valid_entry_gives_the_tick_bbox_as_pixel_bbox(tmp_path):
    areas = _load(tmp_path, _entry(), _pairing())

    # (x0, y0, x1, y1) with y0 the top (y_max tick) and y1 the bottom
    assert areas == {("1", "2"): (10.0, 20.0, 410.0, 300.0)}


def test_stale_entry_is_not_used(tmp_path):
    entry = _entry(pixel_coords_stale_since="2026-09-30")
    assert _load(tmp_path, entry, _pairing()) == {}


def test_labels_that_differ_from_the_scored_ranges_are_not_used(tmp_path):
    assert _load(tmp_path, _entry(), _pairing(x_range=(280.0, 900.0))) == {}


def test_a_different_image_is_not_used(tmp_path):
    entry = _entry(image_path="data/verified_pairs/crops/1/b.png")
    assert _load(tmp_path, entry, _pairing()) == {}


def test_missing_labels_are_not_used(tmp_path):
    assert _load(tmp_path, _entry(y_max_label=None), _pairing()) == {}


def test_verified_pairing_exposes_the_fields_the_loader_reads():
    names = VerifiedPairing.__dataclass_fields__
    for field in ("paper_id", "figure_id", "image_path", "x_range", "y_range"):
        assert field in names
