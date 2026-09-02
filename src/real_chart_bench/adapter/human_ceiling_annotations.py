"""I/O adapter for data/human_ceiling/annotations/*.json (see
data/human_ceiling/FORMAT.md for the on-disk schema this parses).

annotation_source is validated strictly here, at the boundary where untrusted
file content enters the system: a missing or unrecognized value raises
rather than defaulting to "human". This is the adapter-level half of the
project rule that an LLM judgment must never be presented as a human one --
the domain-level half is domain.human_ceiling.require_human_ceiling(), which
guards what a *set* of already-validated sources may be labeled.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from real_chart_bench.domain.curve import Curve, ScaleType
from real_chart_bench.domain.human_ceiling import AnnotationSource
from real_chart_bench.usecase.compute_human_ceiling import FigureAnnotation

_REQUIRED_FIELDS = ("paper_id", "figure_id", "annotation_source", "annotator_id", "annotated_at")


def _require(raw: dict[str, Any], field: str, filename: str) -> Any:
    if field not in raw:
        raise ValueError(f"{filename}: missing required field {field!r}")
    return raw[field]


def _parse_source(raw: dict[str, Any], filename: str) -> AnnotationSource:
    value = _require(raw, "annotation_source", filename)
    try:
        return AnnotationSource(value)
    except ValueError as exc:
        allowed = sorted(s.value for s in AnnotationSource)
        raise ValueError(
            f"{filename}: unrecognized annotation_source {value!r}, must be one of {allowed}"
        ) from exc


def _parse_curve(raw_curve: dict[str, Any], filename: str, x_scale: ScaleType) -> Curve:
    x = raw_curve.get("x", [])
    y = raw_curve.get("y", [])
    if len(x) != len(y):
        raise ValueError(
            f"{filename}: curve has mismatched x/y lengths ({len(x)} vs {len(y)})"
        )
    if not x:
        raise ValueError(f"{filename}: curve has no points")
    return Curve(
        x_values=tuple(x),
        y_values=tuple(y),
        series_label=raw_curve.get("series_label", ""),
        x_scale=x_scale,
    )


def _parse_record(
    raw: dict[str, Any], filename: str, x_scale_by_figure_id: dict[str, ScaleType]
) -> FigureAnnotation:
    for field in _REQUIRED_FIELDS:
        _require(raw, field, filename)

    figure_id = f"{raw['paper_id']}-{raw['figure_id']}"
    source = _parse_source(raw, filename)

    raw_curves = _require(raw, "curves", filename)
    if not raw_curves:
        raise ValueError(f"{filename}: curves must be a non-empty list")

    x_scale = x_scale_by_figure_id.get(figure_id, ScaleType.LINEAR)
    curves = tuple(_parse_curve(c, filename, x_scale) for c in raw_curves)

    return FigureAnnotation(
        figure_id=figure_id,
        source=source,
        annotator_id=raw["annotator_id"],
        annotated_at=raw["annotated_at"],
        curves=curves,
    )


def load_annotation_files(
    directory: Path,
    *,
    x_scale_by_figure_id: dict[str, ScaleType] | None = None,
) -> list[FigureAnnotation]:
    """Parses every *.json file directly under ``directory`` into a
    FigureAnnotation. Returns [] if the directory does not exist yet or has
    no files -- "no annotations collected yet" is the expected v0 state, not
    an error (see usecase/compute_human_ceiling.py's PENDING_NO_ANNOTATIONS).

    Files are read in sorted-filename order so the result is deterministic
    across runs and filesystems.
    """
    if not directory.is_dir():
        return []

    x_scale_by_figure_id = x_scale_by_figure_id or {}
    annotations = []
    for path in sorted(directory.glob("*.json")):
        raw = json.loads(path.read_text())
        annotations.append(_parse_record(raw, path.name, x_scale_by_figure_id))
    return annotations
