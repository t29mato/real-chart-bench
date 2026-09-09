"""Sweeps every verified pairing for GT-vs-image misalignment (2026-09-09).

Motivation (Starrydata3's question after receiving the E2E fixture bundle):
``status: "verified"`` means the *pairing* was cross-verified numerically --
curve count, value ranges and magnitude ordering against the printed axis
(design §7.19). It never checked that individual ground-truth points land on
the markers actually drawn; the pixel-level axis data needed for that check
did not exist until 2026-08-30, after most entries were verified. So a curve
that is distorted *within* the right range passes verification silently.
This script is the missing check, run over the whole registry.

Method, per figure: map every GT point to a pixel through the axis reading in
``axis_pixel_candidates.json`` (``PixelCalibration.to_pixel``), then measure

  d_true  median distance from those pixels to the nearest non-background
          pixel ("ink") in the image
  d_null  the same after 8 rigid shifts of 2% of the image diagonal -- how
          well a deliberately WRONG placement scores on this same image
  ratio   d_true / d_null

``ratio`` is the useful number: near 0 the alignment is real, near 1 the
points are no better placed than a wrong answer. The absolute d_true cannot
be read on its own -- a figure whose markers are large and hollow scores
d_true ~= the marker radius while being perfectly aligned, and a figure whose
ink is dense scores low no matter what. That is exactly what d_null
normalises away.

Known limits, all of which mean this ranks figures for human review rather
than deciding anything:
  - a flagged figure whose axis reading is still ``llm_candidate`` may have a
    wrong axis rather than wrong GT; the two are not separable here.
  - the check cannot run at all where there is no axis reading or no image.
  - do NOT fit a correction (y -> a*y + b) and read a cause off the fitted a.
    On dense figures many transforms tie at distance 0 and the search returns
    an arbitrary one: known-good figures "fit" a ~= 0.85 just as readily as
    the suspicious ones. That coefficient is an artifact, not evidence.

Per design §7.48 nothing here may be called a GT error: an automated check
can only raise ``llm_flagged``, and only a human promotes to
``human_confirmed``.

Run: python scripts/eval/sweep_gt_image_alignment.py [--json OUT]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.domain.curve import ScaleType  # noqa: E402
from real_chart_bench.domain.pixel_calibration import PixelCalibration  # noqa: E402

VP = REPO / "data" / "verified_pairs"
SHIFT_FRACTION = 0.02  # of the image diagonal, for the d_null baseline
FLAG_RATIO = 0.5  # above this, the placement is not convincingly better than wrong


def _calibration(axis: dict) -> PixelCalibration:
    bbox = axis["pixel_bbox_mean"]
    return PixelCalibration(
        pixel_bbox=(bbox["x_min_px"], bbox["y_max_px"], bbox["x_max_px"], bbox["y_min_px"]),
        x_range=(axis["x_min_label"], axis["x_max_label"]),
        y_range=(axis["y_min_label"], axis["y_max_label"]),
        x_scale=ScaleType(axis["x_scale"]),
        y_scale=ScaleType(axis["y_scale"]),
    )


def _ink_distance(path: Path):
    """Distance-to-nearest-ink map. Background is the modal grey level, so
    this works on inverted figures (light ink on a dark ground) too."""
    import numpy as np
    from PIL import Image
    from scipy import ndimage

    image = np.asarray(Image.open(path).convert("L")).astype(float)
    histogram, edges = np.histogram(image, bins=32, range=(0, 256))
    background = (edges[histogram.argmax()] + edges[histogram.argmax() + 1]) / 2
    return ndimage.distance_transform_edt(~(np.abs(image - background) > 60))


def _median_distance(distance, points, dx: float = 0.0, dy: float = 0.0) -> float:
    import numpy as np

    height, width = distance.shape
    xs = np.clip(np.round(points[:, 0] + dx).astype(int), 0, width - 1)
    ys = np.clip(np.round(points[:, 1] + dy).astype(int), 0, height - 1)
    return float(np.median(distance[ys, xs]))


def _axis_is_usable(axis: dict | None) -> bool:
    if axis is None or axis.get("status") == "excluded":
        return False
    labels = ("x_min_label", "x_max_label", "y_min_label", "y_max_label")
    return all(axis.get(label) is not None for label in labels)


def sweep() -> list[dict]:
    import numpy as np

    registry = json.loads((VP / "registry.json").read_text())
    axes = {
        (e["paper_id"], e["figure_id"]): e
        for e in json.loads((VP / "axis_pixel_candidates.json").read_text())
        if "_meta" not in e
    }
    ground_truth = json.loads((VP / "ground_truth.json").read_text())

    rows = []
    for entry in registry:
        if entry.get("status") != "verified":
            continue
        key = (entry["paper_id"], entry["figure_id"])
        axis = axes.get(key)
        curves = ground_truth.get(entry["figure_id"])
        image_path = REPO / entry["image_path"]
        if not _axis_is_usable(axis) or not curves or not image_path.exists():
            continue

        calibration = _calibration(axis)
        points = []
        for curve in curves:
            for x, y in zip(curve["x"], curve["y"], strict=True):
                try:
                    points.append(calibration.to_pixel(x, y))
                except ValueError:
                    pass  # non-positive value on a log axis: not plottable
        if len(points) < 4:
            continue
        points = np.array(points)

        distance = _ink_distance(image_path)
        height, width = distance.shape
        shift = SHIFT_FRACTION * (width**2 + height**2) ** 0.5
        d_true = _median_distance(distance, points)
        d_null = float(
            np.median(
                [
                    _median_distance(
                        distance, points, shift * math.cos(t), shift * math.sin(t)
                    )
                    for t in np.arange(8) * np.pi / 4
                ]
            )
        )
        rows.append(
            {
                "paper_id": key[0],
                "figure_id": key[1],
                "n_points": len(points),
                "axis_status": axis["status"],
                "d_true": round(d_true, 2),
                "d_null": round(d_null, 2),
                "ratio": round(d_true / d_null, 2) if d_null else None,
                "excluded_reason": (entry.get("excluded_reason") or "").split(":")[0],
                "gt_suspect_status": entry.get("gt_suspect_status"),
            }
        )
    return rows


def is_flagged(row: dict) -> bool:
    return row["ratio"] is None or row["ratio"] > FLAG_RATIO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--json", type=Path, help="also write the raw rows here")
    args = parser.parse_args()

    rows = sweep()
    flagged = sorted(
        (r for r in rows if is_flagged(r)), key=lambda r: -(r["ratio"] or 99)
    )
    print(f"measured {len(rows)} verified figures ({len(flagged)} flagged)\n")
    header = f"{'figure':>14} {'n':>5} {'axis':>14} {'d_true':>7} {'d_null':>7} {'ratio':>6}  note"
    print(header)
    for row in flagged:
        note = row["excluded_reason"] or row["gt_suspect_status"] or ""
        print(
            f"{row['paper_id'] + '/' + row['figure_id']:>14} {row['n_points']:5d} "
            f"{row['axis_status']:>14} {row['d_true']:7.2f} {row['d_null']:7.2f} "
            f"{str(row['ratio']):>6}  {note}"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
