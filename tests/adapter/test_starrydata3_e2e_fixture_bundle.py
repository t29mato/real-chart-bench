"""Guards the fixture bundle handed to Starrydata3 on 2026-09-09
(``docs/interop/starrydata3-e2e-fixtures-2026-09-09.json``, built by
``scripts/export/build_starrydata3_e2e_fixtures.py``).

Two things are worth a test rather than a comment:

1. **Staleness.** The bundle hands another project a pixel-level oracle copied
   out of ``axis_pixel_candidates.json``. Those pixel positions get corrected
   -- five commits did exactly that in the week before this bundle was cut --
   and a correction silently invalidates every point pixel we handed over.
   These tests fail when the committed bundle drifts from the repo, which is
   the signal to rebuild and tell them.

2. **Self-consistency.** The whole value of the bundle is that mapping a point
   pixel back through the shipped calibration returns the shipped value. That
   is a property of the file, so it is checked on the file.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BUNDLE = REPO / "docs" / "interop" / "starrydata3-e2e-fixtures-2026-09-09.json"
VP = REPO / "data" / "verified_pairs"


@pytest.fixture(scope="module")
def bundle() -> dict:
    return json.loads(BUNDLE.read_text())


@pytest.fixture(scope="module")
def axis_entries() -> dict:
    entries = json.loads((VP / "axis_pixel_candidates.json").read_text())
    return {(e["paper_id"], e["figure_id"]): e for e in entries if "_meta" not in e}


def _to_data(axis: dict, pixel: float) -> float:
    lo_value, lo_px = axis["p1"]["value"], axis["p1"]["px"]
    hi_value, hi_px = axis["p2"]["value"], axis["p2"]["px"]
    frac = (pixel - lo_px) / (hi_px - lo_px)
    if axis["scale"] == "log":
        return 10 ** (math.log10(lo_value) + frac * (math.log10(hi_value) - math.log10(lo_value)))
    return lo_value + frac * (hi_value - lo_value)


def test_every_shipped_point_pixel_maps_back_to_its_shipped_value(bundle):
    # 1e-3 relative, not exact: px/py are rounded to 3 decimals in the file.
    for figure in bundle["figures"]:
        calibration = figure["axis_calibration"]
        for series in figure["series"]:
            for point in series["points"]:
                assert _to_data(calibration["x"], point["px"]) == pytest.approx(
                    point["x"], rel=1e-3, abs=1e-6
                )
                assert _to_data(calibration["y"], point["py"]) == pytest.approx(
                    point["y"], rel=1e-3, abs=1e-6
                )


def test_shipped_calibration_still_matches_the_repo_axis_data(bundle, axis_entries):
    """Fails when an axis correction lands after the bundle was handed over."""
    for figure in bundle["figures"]:
        axis = axis_entries[(figure["paper_id"], figure["figure_id"])]
        bbox = axis["pixel_bbox_mean"]
        shipped = figure["axis_calibration"]

        assert shipped["x"]["p1"] == {"value": axis["x_min_label"], "px": bbox["x_min_px"]}
        assert shipped["x"]["p2"] == {"value": axis["x_max_label"], "px": bbox["x_max_px"]}
        assert shipped["y"]["p1"] == {"value": axis["y_min_label"], "px": bbox["y_min_px"]}
        assert shipped["y"]["p2"] == {"value": axis["y_max_label"], "px": bbox["y_max_px"]}
        assert shipped["x"]["scale"] == axis["x_scale"]
        assert shipped["y"]["scale"] == axis["y_scale"]


def test_shipped_points_still_match_the_repo_ground_truth(bundle):
    ground_truth = json.loads((VP / "ground_truth.json").read_text())
    for figure in bundle["figures"]:
        curves = ground_truth[figure["figure_id"]]
        assert len(curves) == len(figure["series"])
        for curve, series in zip(curves, figure["series"], strict=True):
            assert [p["x"] for p in series["points"]] == curve["x"]
            assert [p["y"] for p in series["points"]] == curve["y"]


def test_only_owner_reviewed_axis_readings_were_handed_over(bundle, axis_entries):
    for figure in bundle["figures"]:
        axis = axis_entries[(figure["paper_id"], figure["figure_id"])]
        assert axis["status"] == "owner_reviewed", (
            f"{figure['paper_id']}/{figure['figure_id']} was exported with an unreviewed "
            "axis reading"
        )


def test_the_figure_withheld_for_a_gt_mismatch_is_not_also_shipped(bundle):
    shipped = {(f["paper_id"], f["figure_id"]) for f in bundle["figures"]}
    withheld = {(e["paper_id"], e["figure_id"]) for e in bundle["excluded"]}

    assert ("83", "9049") in withheld
    assert not (shipped & withheld)


def test_every_figure_carries_the_attribution_a_cc_by_redistribution_needs(bundle):
    for figure in bundle["figures"]:
        attribution = figure["attribution"]
        assert attribution["doi"]
        assert attribution["license_id"] == "cc-by"
        # No version: OpenAlex records CC BY 1.0/2.0/3.0/4.0 identically, so
        # claiming one would be an unverified upgrade (see docs/interop/README.md).
        assert attribution["license_display"] == "CC BY"
        assert "4.0" not in attribution["line"]
        assert attribution["modified_from_original"] in {"yes", "no"}


def test_no_shipped_figure_has_since_been_excluded_or_flagged(bundle):
    """If a figure we handed over is later taken out of scoring or flagged
    gt_suspect, Starrydata3 is testing against an oracle we no longer trust --
    they have to be told. This fails so that someone tells them.
    """
    registry = {
        (e["paper_id"], e["figure_id"]): e
        for e in json.loads((VP / "registry.json").read_text())
    }
    for figure in bundle["figures"]:
        entry = registry[(figure["paper_id"], figure["figure_id"])]
        assert not entry.get("excluded_reason"), (
            f"{figure['paper_id']}/{figure['figure_id']} was handed to Starrydata3 and has "
            f"since been excluded: {entry['excluded_reason']}"
        )
        assert entry.get("gt_suspect_status") is None, (
            f"{figure['paper_id']}/{figure['figure_id']} was handed to Starrydata3 and has "
            f"since been flagged gt_suspect: {entry['gt_suspect_status']}"
        )
