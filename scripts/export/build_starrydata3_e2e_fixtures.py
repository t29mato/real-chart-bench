"""Builds the E2E-test fixture bundle handed to Starrydata3 (2026-09-09 request).

Starrydata3's digitizer E2E tests need, per figure: the image, the two-point
axis calibration (value + PIXEL position, per axis, plus linear/log), and every
ground-truth point in BOTH data coordinates and PIXEL coordinates -- so a test
can calibrate at known pixels, click known pixels, and assert known values.

This repo stores none of the point pixel positions: ground_truth.json is data
space only. They are DERIVED here by mapping each GT point through the axis
calibration in axis_pixel_candidates.json (linear, or log10 for log axes).
The derivation is therefore exact by construction for the round trip
(pixel -> value); what it does NOT guarantee is that the derived pixel lands on
the marker actually drawn. That has to be checked against the image, which is
why every figure in FIGURES was inspected on a rendered overlay before being
listed here, and why 83/9049 is excluded (see EXCLUDED).

Selection is restricted to entries where all of the following hold, so that the
calibration handed over is human-reviewed rather than raw LLM output:
  - registry.json status == "verified" and tick_range_source == "owner_reviewed"
  - axis_pixel_candidates.json status == "owner_reviewed"
  - registry's x_tick_range/y_tick_range agree with the axis file's tick labels
  - every GT point maps inside the plot frame (catches unit-space mismatches --
    a registry in K against a printed degC axis, or SI against printed units)
  - no excluded_reason and no gt_suspect flag: a figure this repo has already
    taken out of its own scoring, or flagged as GT-suspect, must not be handed
    to anyone else as an oracle. Enforced in build() rather than left to the
    hand-maintained FIGURES list, so a later flag removes a figure by itself.

Run: python scripts/export/build_starrydata3_e2e_fixtures.py <output_dir>
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import shutil
import sys
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(REPO_SRC))

from real_chart_bench.domain.curve import ScaleType  # noqa: E402
from real_chart_bench.domain.pixel_calibration import PixelCalibration  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
VP = REPO / "data" / "verified_pairs"

# (paper_id, figure_id) -> why this figure is in the bundle. Every one of these
# was confirmed on a rendered overlay (GT points mapped to pixels, drawn on the
# source image) before being added.
FIGURES: dict[tuple[str, str], str] = {
    ("27759", "25217"): "log-y over 5 decades (0.1-1e5 S/m), 4 series, clean vector figure.",
    ("5902", "15112"): (
        "log-X (2-300 K) -- the only log-x figure in the reviewed pool. "
        "Single series, 46 points on a dense hollow-marker curve."
    ),
    ("47534", "49581"): (
        "log-y, inverted colour scheme (yellow ink on dark purple), "
        "decimal-comma tick label ('0,1'), 2 sparse series."
    ),
    ("16111", "15452"): (
        "14 series, 1259 points, two-column legend, curves overlapping in a "
        "narrow ZT band. Data extends left of the smallest labelled tick (350 K) "
        "to ~333 K, inside the drawn frame."
    ),
    ("446", "8725"): (
        "Scanned raster figure on a saturated yellow background, 3 series, "
        "x axis in degC."
    ),
    ("10939", "1531"): "4 series of similar warm hues that converge at low temperature.",
    ("17040", "21020"): (
        "Dual y-axis figure (left G_CNP, right V_c) -- calibration below is "
        "the LEFT axis. 4 series, 220 points, x spans negative to positive."
    ),
    ("22102", "21245"): (
        'figure_kind "line_only" (continuous trace, no markers); single series, '
        "52 points, x in seconds up to 20000."
    ),
    ("17038", "20816"): (
        "Sparsest case: 2 series, 6 points total, with error bars drawn on "
        "every point."
    ),
    # --- 2026-09-11: sibling figures, so the bundle has papers carrying more
    # than one figure. Starrydata3's demo needs a multi-figure paper to
    # exercise its figure list and its image-per-figure overlay rendering;
    # the original 9 happened to be one paper each.
    ("28331", "28492"): "Paper 28331, panel 4(a). 5 figures from this paper are in the bundle.",
    ("28331", "28495"): "Paper 28331, panel 4(c) -- sibling of 4(a).",
    ("28331", "28498"): "Paper 28331, panel 4(e) -- sibling of 4(a).",
    ("28331", "28500"): "Paper 28331, panel 5(a) -- a second figure of the same paper.",
    ("28331", "28502"): "Paper 28331, panel 5(c) -- sibling of 5(a).",
    ("446", "8724"): "Sibling of 446 4(b): same scanned yellow figure, panel 4(a).",
    ("446", "8726"): "Sibling of 446 4(b): same scanned yellow figure, panel 4(c).",
    ("10939", "1536"): "Sibling of 10939 4(a): panel 5(b), same 4 samples.",
    ("10939", "1537"): "Sibling of 10939 4(a): panel 5(c), 3 series.",
    ("27759", "25218"): "Sibling of 27759 fig 7: figure 8, linear y, same 4 samples.",
    ("27759", "25222"): "Sibling of 27759 fig 7: figure 16(a), a second log-y axis.",
    ("22102", "21246"): 'Sibling of 22102 3a: panel 3c, also "line_only".',
}

# Deliberately withheld, with the reason, so the request is answered honestly.
EXCLUDED: dict[tuple[str, str], str] = {
    ("83", "9049"): (
        "Withheld: the derived pixels do not sit on the drawn markers closely "
        "enough for a pixel-level oracle -- a median 6.4 px from the nearest "
        "ink, against <=1.4 px for every figure in this bundle. This figure "
        "WAS included in the 2026-09-04 bundle. "
        "CORRECTION (2026-09-11), replacing what the 2026-09-09 bundle said "
        "here: that text called this a systematic ground-truth error of "
        "'GT y ~0.85x the drawn y'. Both halves were wrong. The 0.85 came from "
        "fitting a correction to minimise distance-to-ink, which is degenerate "
        "on this kind of figure -- known-good figures fit the same coefficient "
        "with zero improvement. And the figure was not un-investigated: the "
        "owner adjudicated it on 2026-09-07 as excluded_reason=harness_limit, "
        "finding the axis correct and the residual explained by an old scan "
        "whose marker centres are inherently ambiguous, with NO evidence of a "
        "ground-truth error. Only the practical conclusion stands: not usable "
        "as a pixel-level oracle. See "
        "docs/experiments/2026-09-09-gt-image-alignment-sweep.md."
    ),
}


def _load(name: str):
    return json.loads((VP / name).read_text())


def _calibration(ax: dict) -> PixelCalibration:
    """The axis entry's tick labels + pixels as the domain's calibration object.

    pixel_bbox is (x0, y_top, x1, y_bottom); the axis file stores y_min_px for
    the SMALLEST label, which is the bottom of the plot, hence the swap.
    """
    bbox = ax["pixel_bbox_mean"]
    return PixelCalibration(
        pixel_bbox=(bbox["x_min_px"], bbox["y_max_px"], bbox["x_max_px"], bbox["y_min_px"]),
        x_range=(ax["x_min_label"], ax["x_max_label"]),
        y_range=(ax["y_min_label"], ax["y_max_label"]),
        x_scale=ScaleType(ax["x_scale"]),
        y_scale=ScaleType(ax["y_scale"]),
    )


def _resolution(v0: float, p0: float, v1: float, p1: float, scale: str) -> dict:
    """What one pixel is worth on an axis, so a caller can set a sane tolerance."""
    span_px = abs(p1 - p0)
    if scale == "log":
        return {"decades_per_px": round(abs(math.log10(v1) - math.log10(v0)) / span_px, 6)}
    return {"units_per_px": round(abs(v1 - v0) / span_px, 6)}


def _attribution() -> dict[str, dict[str, str]]:
    """Committed file path -> {doi, license, modified} from ATTRIBUTION.md."""
    rows = {}
    row_re = re.compile(r"^\| \[([^\]]+)\]\([^)]+\) \| (\S+) \| `([^`]+)` \| (\S+) \|$")
    for line in (VP / "ATTRIBUTION.md").read_text().splitlines():
        m = row_re.match(line.strip())
        if m:
            doi, lic, path, modified = m.groups()
            rows[path] = {"doi": doi, "license_id": lic, "modified_from_original": modified}
    return rows


def build(out_dir: Path) -> dict:
    registry = {(e["paper_id"], e["figure_id"]): e for e in _load("registry.json")}
    axes = {
        (e["paper_id"], e["figure_id"]): e
        for e in _load("axis_pixel_candidates.json")
        if "_meta" not in e
    }
    gt = _load("ground_truth.json")
    attribution = _attribution()

    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    figures = []
    for key, why in FIGURES.items():
        reg, ax = registry[key], axes[key]
        if reg.get("excluded_reason") or reg.get("gt_suspect_status"):
            raise SystemExit(
                f"{key[0]}/{key[1]} is excluded from scoring or flagged gt_suspect "
                f"({reg.get('excluded_reason') or reg.get('gt_suspect_status')}) -- "
                "it must not be exported as an oracle. Drop it from FIGURES."
            )
        src = REPO / reg["image_path"]
        dest_name = f"{key[0]}_{key[1]}{src.suffix}"
        shutil.copyfile(src, images_dir / dest_name)
        blob = src.read_bytes()

        from PIL import Image  # noqa: PLC0415 -- optional dep, only needed here

        with Image.open(src) as im:
            width, height = im.size

        bbox = ax["pixel_bbox_mean"]
        calibration = _calibration(ax)

        series = []
        for i, curve in enumerate(gt[key[1]]):
            points = []
            for x, y in zip(curve["x"], curve["y"], strict=True):
                px, py = calibration.to_pixel(x, y)
                points.append({"x": x, "y": y, "px": round(px, 3), "py": round(py, 3)})
            series.append(
                {
                    "series_index": i,
                    "series_label": None,  # not recoverable from this repo's data
                    "prop_x": curve["prop_x"],
                    "prop_y": curve["prop_y"],
                    "unit_x": curve["unit_x"],
                    "unit_y": curve["unit_y"],
                    "n_points": len(points),
                    "points": points,
                }
            )

        attr = attribution.get(reg["image_path"], {})
        figures.append(
            {
                "paper_id": key[0],
                "figure_id": key[1],
                "figure_reference": reg.get("figure_reference"),
                "figure_kind": reg.get("figure_kind"),
                "image": {
                    "file": f"images/{dest_name}",
                    "width_px": width,
                    "height_px": height,
                    "bytes": len(blob),
                    "sha256": hashlib.sha256(blob).hexdigest(),
                    "source_path_in_real_chart_bench": reg["image_path"],
                },
                "axis_calibration": {
                    "x": {
                        "scale": ax["x_scale"],
                        "unit": ax.get("x_axis_unit"),
                        "axis_label_raw": ax.get("x_axis_label_raw"),
                        "p1": {"value": ax["x_min_label"], "px": bbox["x_min_px"]},
                        "p2": {"value": ax["x_max_label"], "px": bbox["x_max_px"]},
                    },
                    "y": {
                        "scale": ax["y_scale"],
                        "unit": ax.get("y_axis_unit"),
                        "axis_label_raw": ax.get("y_axis_label_raw"),
                        "p1": {"value": ax["y_min_label"], "px": bbox["y_min_px"]},
                        "p2": {"value": ax["y_max_label"], "px": bbox["y_max_px"]},
                    },
                    "resolution": {
                        "x": _resolution(
                            ax["x_min_label"], bbox["x_min_px"],
                            ax["x_max_label"], bbox["x_max_px"], ax["x_scale"],
                        ),
                        "y": _resolution(
                            ax["y_min_label"], bbox["y_min_px"],
                            ax["y_max_label"], bbox["y_max_px"], ax["y_scale"],
                        ),
                        "note": "What one pixel is worth on each axis -- use it to size the "
                        "tolerance of a pixel-level assertion. Linear axes give units_per_px; "
                        "log axes give decades_per_px (1 px is a factor of 10**decades_per_px).",
                    },
                    "provenance": (
                        "Printed tick labels and their pixel positions, cross-read by two "
                        "independent vision models and then owner-reviewed against the image "
                        "(axis_pixel_candidates.json status=owner_reviewed). "
                        "Model-to-model pixel disagreement for these entries: "
                        + f"{ax.get('model_disagreement_px')} px."
                    ),
                    "notes": ax.get("notes") or [],
                },
                "drawn_frame_range": {
                    "x": reg.get("x_range"),
                    "y": reg.get("y_range"),
                    "note": (
                        "Outer edge of the drawn plot frame, NOT tick values. "
                        "Do not calibrate from this -- use axis_calibration. "
                        "Present only so you can tell framing margin from a data error."
                    ),
                },
                "series": series,
                "attribution": {
                    "doi": attr.get("doi"),
                    "doi_url": f"https://doi.org/{attr['doi']}" if attr.get("doi") else None,
                    "license_id": attr.get("license_id"),
                    "license_display": "CC BY",
                    "modified_from_original": attr.get("modified_from_original"),
                    "line": (
                        f"Figure from https://doi.org/{attr.get('doi')}, licensed CC BY "
                        f"(no version recorded -- see license_notice)."
                    ),
                },
                "why_selected": why,
            }
        )

    return {
        "schema": "real-chart-bench/starrydata3-e2e-fixtures",
        "schema_version": 2,
        "generated_at": "2026-09-09",
        "source_repo": "real-chart-bench (private)",
        "point_pixels_are_derived": (
            "px/py on every point are DERIVED by mapping the ground-truth data value "
            "through axis_calibration (linear, or log10 on log axes); they are not "
            "separately measured. The round trip pixel->value is therefore exact by "
            "construction, which is what an E2E oracle needs. Each figure was also "
            "checked on a rendered overlay so that the derived pixels do land on the "
            "markers actually drawn."
        ),
        "license_notice": (
            "Every figure comes from an open-access paper whose license_id in this "
            "repo's registry is the bare string \"cc-by\", taken from OpenAlex's "
            "`license` field, which carries NO version (CC BY 1.0/2.0/3.0/4.0 are all "
            "recorded identically). Attribute these as \"CC BY\" without a version "
            "unless you verify the specific paper's license version with the "
            "publisher. CC BY permits redistribution, including in a public "
            "repository, with attribution -- attribution.line and "
            "attribution.modified_from_original give what you need per figure."
        ),
        "excluded": [
            {"paper_id": p, "figure_id": f, "reason": r} for (p, f), r in EXCLUDED.items()
        ],
        "figures": figures,
    }


def write_overlays(bundle: dict, out_dir: Path) -> None:
    """Draw every fixture's derived pixels on its image, for eyeball checking.

    Not test input -- this is the evidence that the derived pixels sit on the
    markers actually drawn, which is the one thing the round-trip maths cannot
    prove on its own.
    """
    from PIL import Image, ImageDraw  # noqa: PLC0415

    overlays = out_dir / "overlays"
    overlays.mkdir(parents=True, exist_ok=True)
    for fig in bundle["figures"]:
        with Image.open(out_dir / fig["image"]["file"]) as im:
            canvas = im.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        cal = fig["axis_calibration"]
        for px in (cal["x"]["p1"]["px"], cal["x"]["p2"]["px"]):
            draw.line([(px, 0), (px, canvas.height)], fill=(0, 200, 0), width=1)
        for py in (cal["y"]["p1"]["px"], cal["y"]["p2"]["px"]):
            draw.line([(0, py), (canvas.width, py)], fill=(0, 200, 200), width=1)
        for s in fig["series"]:
            for p in s["points"]:
                x, y, r = p["px"], p["py"], 4
                draw.line([(x - r, y), (x + r, y)], fill=(255, 0, 0), width=1)
                draw.line([(x, y - r), (x, y + r)], fill=(255, 0, 0), width=1)
        canvas.save(overlays / f"{fig['paper_id']}_{fig['figure_id']}.png")


def main() -> None:
    out_dir = Path(sys.argv[1]).resolve()
    bundle = build(out_dir)
    (out_dir / "fixtures.json").write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
    write_overlays(bundle, out_dir)
    n_pts = sum(s["n_points"] for f in bundle["figures"] for s in f["series"])
    print(f"wrote {out_dir}/fixtures.json: {len(bundle['figures'])} figures, {n_pts} points")


if __name__ == "__main__":
    main()
