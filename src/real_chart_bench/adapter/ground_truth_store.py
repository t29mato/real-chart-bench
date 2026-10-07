"""Loads the ground truth every scorer uses: Starrydata's digitization
(data/verified_pairs/ground_truth.json) plus supplementary digitizations
(data/verified_pairs/ground_truth_supplement/*.json).

Why a supplement exists: Starrydata records a paper's *own* data, so series
a figure draws from other works -- reference or comparison curves -- are
never digitized there. The benchmark task is to extract every series the
figure draws, so a model that reads such a curve correctly was being scored
as a false positive (owner decision 2026-10-01, design §7.65). Those series
are digitized for this benchmark and kept apart, with who/when/why, so the
Starrydata part stays exactly what anyone can download.

Every merged row carries `source`: "starrydata" or
"real-chart-bench-supplement". A row may also carry `y_axis`
("primary" / "secondary", design §7.84) -- which y axis the series is read
against on a figure that prints two. Absent means the left axis, which is
every row today.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from real_chart_bench.domain.curve import YAxis

SUPPLEMENT_SOURCE = "real-chart-bench-supplement"
_REQUIRED = ("paper_id", "figure_id", "digitized_by", "digitized_at", "tool", "reason")
# design §7.84: which y axis a curve is read against. Absent -- every row of
# the ground truth today -- means the left axis.
_Y_AXES = ("primary", "secondary")


class GroundTruthSupplementError(ValueError):
    pass


def _supplement_files(supplement_dir: Path) -> list[Path]:
    if not supplement_dir.is_dir():
        return []
    return sorted(supplement_dir.glob("*.json"))


def _validate(record: dict[str, Any], path: Path) -> None:
    for key in _REQUIRED:
        if not str(record.get(key) or "").strip():
            raise GroundTruthSupplementError(f"{path.name}: missing or empty '{key}'")
    curves = record.get("curves")
    if not curves:
        raise GroundTruthSupplementError(f"{path.name}: no curves")
    for i, curve in enumerate(curves):
        x, y = curve.get("x") or [], curve.get("y") or []
        if not x or len(x) != len(y):
            raise GroundTruthSupplementError(
                f"{path.name}: curve {i} needs equal-length, non-empty x and y"
            )
        if "y_axis" in curve and curve["y_axis"] not in _Y_AXES:
            raise GroundTruthSupplementError(
                f"{path.name}: curve {i} has y_axis {curve['y_axis']!r}, expected one of {_Y_AXES}"
            )


def load_ground_truth(base_path: Path, supplement_dir: Path) -> dict[str, list[dict[str, Any]]]:
    """figure_id -> curve rows (Starrydata's, then the supplement's)."""
    base = json.loads(base_path.read_text())
    merged = {
        figure_id: [{**row, "source": "starrydata"} for row in rows]
        for figure_id, rows in base.items()
    }
    for path in _supplement_files(supplement_dir):
        record = json.loads(path.read_text())
        _validate(record, path)
        figure_id = str(record["figure_id"])
        if figure_id not in merged:
            raise GroundTruthSupplementError(
                f"{path.name}: figure_id {figure_id} has no Starrydata ground truth to supplement"
            )
        for curve in record["curves"]:
            merged[figure_id].append(
                {
                    "x": curve["x"],
                    "y": curve["y"],
                    "prop_y": curve.get("series_label") or "",
                    **({"y_axis": curve["y_axis"]} if "y_axis" in curve else {}),
                    "source": SUPPLEMENT_SOURCE,
                    "digitized_by": record["digitized_by"],
                    "digitized_at": record["digitized_at"],
                }
            )
    return merged


def ground_truth_revision(supplement_dir: Path) -> str:
    """Suffix for dataset_version: "" with no supplement, otherwise
    "-gtsup{N}-{hash}" -- N supplemented figures, hash of their contents --
    so any change to the supplement yields a different dataset_version."""
    files = _supplement_files(supplement_dir)
    if not files:
        return ""
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return f"-gtsup{len(files)}-{digest.hexdigest()[:7]}"


def y_axis_of(row: dict[str, Any]) -> YAxis:
    """Which y axis a merged ground-truth row is read against (design §7.84).

    Absent or empty is the left axis: every row of the ground truth today says
    nothing, and a figure with one y axis never will.
    """
    return YAxis(row.get("y_axis") or YAxis.PRIMARY.value)
