"""Converts a starry-digitizer (https://github.com/t29mato/starry-digitizer,
also at /home/mato/repos/starry-digitizer on this owner's machine -- the
Starrydata project's own digitizer web UI) exported project into a
data/human_ceiling/annotations/*.json record (see FORMAT.md's "Using
starry-digitizer" section for the workflow this supports).

starry-digitizer's "Export Project" button downloads a .zip containing
project.json + the source image. project.json stores axis calibration (two
pixel<->value pairs per axis, log-scale flags, an optional graph-tilt
correction) and each dataset's clicked points as *pixel* coordinates, not
real-world values -- the app only converts pixel to value at render/copy
time. This module replicates that conversion
(starry-digitizer's src/domain/services/axisSetCalculator.ts
calculateXYValues(), mirrored field-for-field below) so a re-digitization
done in the tool can be turned into a schema-valid annotation without the
annotator doing unit conversion by hand, which would just be a second,
uncontrolled source of disagreement stacked on top of the digitization
itself.

This adapter never guesses annotation_source or any other identity field --
scripts/eval/import_human_ceiling_annotation.py always passes them
explicitly, the same discipline
adapter/human_ceiling_annotations.py's parser applies on the read side.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path
from typing import Any


class StarryDigitizerImportError(ValueError):
    """Raised when a starry-digitizer project cannot be converted -- e.g. an
    axis point was never placed on the image (still sitting at the tool's
    {xPx: -999, yPx: -999} placeholder), which would otherwise silently
    produce nonsense coordinates rather than a clear failure."""


def load_project_from_zip(zip_path: Path) -> dict[str, Any]:
    """Reads project.json out of a starry-digitizer "Export Project" .zip."""
    with zipfile.ZipFile(zip_path) as zf:
        try:
            raw = zf.read("project.json")
        except KeyError as exc:
            raise StarryDigitizerImportError(
                f"{zip_path}: no project.json in this zip -- is it a "
                "starry-digitizer 'Export Project' download?"
            ) from exc
    return json.loads(raw)


def _axis_is_calibrated(axis: dict[str, Any]) -> bool:
    coord = axis.get("coord", {})
    return coord.get("xPx", -999) >= 0 and coord.get("yPx", -999) >= 0


def _require_calibrated_axis_set(axis_set: dict[str, Any], *, dataset_name: str) -> None:
    axis_set_name = axis_set.get("name") or axis_set.get("id")
    for key in ("x1", "x2", "y1", "y2"):
        if not _axis_is_calibrated(axis_set[key]):
            raise StarryDigitizerImportError(
                f"dataset {dataset_name!r} uses axis set {axis_set_name!r} whose "
                f"{key} point was never placed on the image -- axis calibration "
                "is incomplete."
            )
    if axis_set["x1"]["value"] == axis_set["x2"]["value"]:
        raise StarryDigitizerImportError(
            f"axis set {axis_set_name!r}: x1 and x2 have the same value "
            f"({axis_set['x1']['value']!r})"
        )
    if axis_set["y1"]["value"] == axis_set["y2"]["value"]:
        raise StarryDigitizerImportError(
            f"axis set {axis_set_name!r}: y1 and y2 have the same value "
            f"({axis_set['y1']['value']!r})"
        )


def _pixel_to_value(axis_set: dict[str, Any], x_px: float, y_px: float) -> tuple[float, float]:
    """Mirrors AxisSetCalculator.calculateXYValues() in starry-digitizer
    (src/domain/services/axisSetCalculator.ts), minus its final
    significant-digit rounding (effectiveDigits) -- we keep full float
    precision instead, which only helps agreement measurement, never hurts
    it."""
    x1, x2, y1, y2 = axis_set["x1"], axis_set["x2"], axis_set["y1"], axis_set["y2"]
    xa, ya = x1["coord"]["xPx"], x1["coord"]["yPx"]
    xb, yb = x2["coord"]["xPx"], x2["coord"]["yPx"]
    a, b = x1["value"], x2["value"]
    xc, yc = y1["coord"]["xPx"], y1["coord"]["yPx"]
    xd, yd = y2["coord"]["xPx"], y2["coord"]["yPx"]
    c, d = y1["value"], y2["value"]

    xp, yq = x_px, y_px
    if axis_set.get("considerGraphTilt"):
        xab, yab = xb - xa, yb - ya
        xcd, ycd = xd - xc, yd - yc
        r = ((y_px - ya) * xcd - (x_px - xa) * ycd) / (yab * xcd - xab * ycd)
        s = ((y_px - yc) * xab - (x_px - xc) * yab) / (ycd * xab - xcd * yab)
        xp = xa + r * xab
        yq = yc + s * ycd

    if axis_set["xIsLogScale"]:
        x_value = 10 ** (
            ((xp - xa) / (xb - xa)) * (math.log10(b) - math.log10(a)) + math.log10(a)
        )
    else:
        x_value = ((xp - xa) / (xb - xa)) * (b - a) + a

    if axis_set["yIsLogScale"]:
        y_value = 10 ** (
            ((yq - yc) / (yd - yc)) * (math.log10(d) - math.log10(c)) + math.log10(c)
        )
    else:
        y_value = ((yq - yc) / (yd - yc)) * (d - c) + c

    return x_value, y_value


def convert_project_to_annotation(
    project: dict[str, Any],
    *,
    paper_id: str,
    figure_id: str,
    annotator_id: str,
    annotated_at: str,
    annotation_source: str = "human",
    tool: str = "starry-digitizer",
    notes: str | None = None,
) -> dict[str, Any]:
    """Turns a parsed starry-digitizer project (see load_project_from_zip)
    into a dict matching data/human_ceiling/schema.json: one curve per
    non-empty dataset, series_label taken from the dataset's name.

    A dataset with zero points is skipped rather than emitted as an empty
    curve -- the schema requires every curve to have at least one point.
    See data/human_ceiling/FORMAT.md on why a series the annotator could not
    reliably digitize belongs in ``notes`` (free text), not as an empty or
    fabricated curve: an omitted series and a disagreeing series mean
    different things to the scoring metric, so silently dropping one here
    with no record would be worse than not touching it at all.
    """
    axis_sets_by_id = {a["id"]: a for a in project.get("axisSets", [])}

    curves: list[dict[str, Any]] = []
    for dataset in project.get("datasets", []):
        points = dataset.get("points", [])
        if not points:
            continue

        dataset_name = dataset.get("name") or f"dataset-{dataset['id']}"
        axis_set = axis_sets_by_id.get(dataset["axisSetId"])
        if axis_set is None:
            raise StarryDigitizerImportError(
                f"dataset {dataset_name!r} references axisSetId="
                f"{dataset['axisSetId']!r}, which is not in project['axisSets']"
            )
        _require_calibrated_axis_set(axis_set, dataset_name=dataset_name)

        xs: list[float] = []
        ys: list[float] = []
        for point in points:
            x_value, y_value = _pixel_to_value(axis_set, point["xPx"], point["yPx"])
            xs.append(x_value)
            ys.append(y_value)

        curves.append({"series_label": dataset_name, "x": xs, "y": ys})

    if not curves:
        raise StarryDigitizerImportError(
            "no dataset in this project has any digitized points -- nothing to convert"
        )

    record: dict[str, Any] = {
        "paper_id": paper_id,
        "figure_id": figure_id,
        "annotation_source": annotation_source,
        "annotator_id": annotator_id,
        "annotated_at": annotated_at,
        "tool": tool,
        "curves": curves,
    }
    if notes:
        record["notes"] = notes
    return record
