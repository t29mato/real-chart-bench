"""Converts registry.json's x_range/y_range and ground_truth.json's curve
values from Starrydata's SI units to each paper's own printed display
units (design 7.47, owner decision 2026-09-01).

Why: Starrydata always stores digitized curves in SI units, but almost
every paper displays a rescaled unit on its own axis (uV/K, S/cm,
mOhm.cm, ...). Every "unit bug" found in this project so far (design
7.44/7.45) came from registry.json's x_range/y_range needing to track
that SI<->display distinction by hand, with no independent check at
authoring time -- an error-prone process. Storing ground_truth.json
directly in each paper's own display units removes the whole category of
bug: there is no second unit space to get wrong, and a human comparing
ground truth against the original chart does it by eye with no mental
unit conversion, which is also a much easier way to *verify* the
benchmark (the owner's stated motivation).

This does NOT change what "correct" means for scoring: NormalizedYDistanceMetric
normalizes error by ground_truth's own observed range, so it is invariant
to a uniform rescaling of x_range/y_range and ground_truth curves together
-- this is a re-parameterization, not a change to the metric's semantics.
(design 7.47 also fixed a related latent bug: the metric's zero-range
epsilon was a fixed absolute value that assumed SI-scale magnitudes; see
domain/metrics.py's _is_negligible.)

Conversion factor per entry, reusing exactly the same trust hierarchy as
generate_verified_pairs_visual_audit.py's `_derive_factor`:
  1. axis_pixel_candidates.json confident factor (multiplicative or
     additive kind) -- the only independently-verified source (two
     vision models' printed-tick-label reads, cross-checked against the
     registry's own range).
  2. evidence-text-derived factor (y-axis only) -- a human/agent already
     validated this once while writing the entry, just not independently
     re-derived the way (1) is.
  3. Neither available, or the axis-pixel factor is "indeterminate" (the
     two hypotheses this project checks -- pure multiplicative, pure
     additive -- didn't confidently fit) or "log10" (paper 46278's raw-
     log10-printed-axis case, where the printed axis isn't a rescaling of
     the physical quantity at all, it's its logarithm -- converting the
     stored curve to "look like" a log10 value would corrupt the actual
     physical data): entry is left in SI, unconverted, and reported as such.

Run: python scripts/eval/convert_ground_truth_to_display_units.py
Writes updated registry.json + ground_truth.json in place; prints a
summary of how many entries were converted vs left in SI and why.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts/eval"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from generate_verified_pairs_visual_audit import (  # noqa: E402
    _derive_factor,
    _evidence_text_factor,
)

REGISTRY_PATH = REPO_ROOT / "data/verified_pairs/registry.json"
GROUND_TRUTH_PATH = REPO_ROOT / "data/verified_pairs/ground_truth.json"
AXIS_PIXEL_CANDIDATES_PATH = REPO_ROOT / "data/verified_pairs/axis_pixel_candidates.json"


def _resolve_factor(entry: dict, axp: dict | None) -> dict:
    """Returns {"x": {...factor...}, "y": {...factor...}, "source": str}."""
    x_scale = entry.get("x_scale", "linear")
    y_scale = entry.get("y_scale", "linear")
    no_label = {
        "kind": "indeterminate",
        "factor": 1.0,
        "offset": None,
        "confident": None,
        "detail": "no printed label for this axis",
    }

    no_op = {"kind": "multiplicative", "factor": 1.0, "offset": None, "confident": None}

    if axp is not None and axp.get("status") != "excluded":
        fx = (
            _derive_factor(entry["x_range"], axp["x_min_label"], axp["x_max_label"], x_scale)
            if axp.get("x_min_label") is not None and axp.get("x_max_label") is not None
            else no_label
        )
        fy = (
            _derive_factor(entry["y_range"], axp["y_min_label"], axp["y_max_label"], y_scale)
            if axp.get("y_min_label") is not None and axp.get("y_max_label") is not None
            else no_label
        )
        y_ok = fy["kind"] in ("multiplicative", "additive") and fy["confident"] is True
        if not y_ok:
            return {"x": fx, "y": fy, "source": f"axis-pixel-y-blocked:{fy['kind']}"}

        # x in this domain is always Temperature: the only two physically
        # sensible transformations are "no change" (k=1) or a degC<->K
        # offset -- there is no such thing as "the x-axis is displayed at
        # 99.3% Kelvin scale". A confident *multiplicative* factor merely
        # close to 1.0 is exactly the benign "registry range extends a
        # touch past the outermost printed label" pattern landing inside
        # _derive_factor's 3% tolerance by coincidence, not a real
        # conversion -- accepting it here would apply a physically
        # meaningless shrink/stretch to real temperature values. Require
        # near-exact 1.0 for the multiplicative case; additive (the actual
        # degC<->K case) is unaffected by this guard.
        x_is_trivial = fx["kind"] == "multiplicative" and abs(fx["factor"] - 1.0) < 0.001
        x_ok = fx["confident"] is True and (fx["kind"] == "additive" or x_is_trivial)
        if x_ok:
            return {"x": fx, "y": fy, "source": "axis-pixel"}
        # x's factor-agreement check failed, but in this domain that's
        # overwhelmingly the benign "registry range extends a bit past the
        # outermost printed label" pattern (design 7.44/7.45 spot-checked
        # many of these directly against the source images) -- essentially
        # never a case where x actually needs a nontrivial rescale. Default
        # to "no conversion" for x rather than blocking y's conversion too:
        # this changes no x values (leaves registry's existing x_range
        # exactly as-is), so it cannot introduce wrong data, only decline
        # to also re-express x in a possibly-cleaner printed unit.
        return {"x": no_op, "y": fy, "source": f"axis-pixel-x-defaulted:{fx['kind']}"}

    fy = _evidence_text_factor(entry["evidence"])
    if fy is not None:
        fx = {"kind": "multiplicative", "factor": 1.0, "offset": None, "confident": None}
        return {"x": fx, "y": fy, "source": "evidence-text"}

    return {"x": None, "y": None, "source": "none"}


def main() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    axp_raw = json.loads(AXIS_PIXEL_CANDIDATES_PATH.read_text())
    axp_by_key = {(a["paper_id"], a["figure_id"]): a for a in axp_raw if "_meta" not in a}

    converted = []
    skipped = []

    for entry in registry:
        if entry.get("status") != "verified":
            continue
        key = (entry["paper_id"], entry["figure_id"])
        axp = axp_by_key.get(key)
        resolution = _resolve_factor(entry, axp)

        source = resolution["source"]
        is_convertible = source in ("evidence-text", "axis-pixel") or source.startswith(
            "axis-pixel-x-defaulted"
        )
        if not is_convertible:
            skipped.append((key, entry["figure_reference"], resolution["source"]))
            continue

        fx, fy = resolution["x"], resolution["y"]
        k_x = fx["factor"] if fx else 1.0
        off_x = (fx["offset"] or 0.0) if fx else 0.0
        k_y = fy["factor"]
        off_y = fy["offset"] or 0.0

        old_x_range = list(entry["x_range"])
        old_y_range = list(entry["y_range"])
        entry["x_range"] = [v * k_x + off_x for v in entry["x_range"]]
        entry["y_range"] = [v * k_y + off_y for v in entry["y_range"]]

        curves = ground_truth.get(entry["figure_id"], [])
        for curve in curves:
            curve["x"] = [v * k_x + off_x for v in curve["x"]]
            curve["y"] = [v * k_y + off_y for v in curve["y"]]
            if axp and axp.get("x_axis_unit") and axp["x_axis_unit"] != "-":
                curve["unit_x"] = axp["x_axis_unit"]
            if axp and axp.get("y_axis_unit") and axp["y_axis_unit"] != "-":
                curve["unit_y"] = axp["y_axis_unit"]

        converted.append(
            {
                "key": key,
                "figure_reference": entry["figure_reference"],
                "source": resolution["source"],
                "k_x": k_x,
                "off_x": off_x,
                "k_y": k_y,
                "off_y": off_y,
                "old_x_range": old_x_range,
                "new_x_range": entry["x_range"],
                "old_y_range": old_y_range,
                "new_y_range": entry["y_range"],
            }
        )

    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")
    GROUND_TRUTH_PATH.write_text(json.dumps(ground_truth, indent=2, ensure_ascii=False) + "\n")

    print(f"converted: {len(converted)}")
    print(f"skipped (left in SI): {len(skipped)}")
    print()
    print("--- skipped entries and why ---")
    for key, ref, reason in skipped:
        print(f"  {key} ({ref}): {reason}")
    print()
    print("--- converted entries ---")
    for c in converted:
        print(
            f"  {c['key']} ({c['figure_reference']}) [{c['source']}] "
            f"x: k={c['k_x']:.6g} off={c['off_x']:.6g} {c['old_x_range']} -> {c['new_x_range']} | "
            f"y: k={c['k_y']:.6g} off={c['off_y']:.6g} {c['old_y_range']} -> {c['new_y_range']}"
        )

    with open("/tmp/gt_conversion_report.json", "w") as f:
        json.dump({"converted": converted, "skipped": skipped}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
