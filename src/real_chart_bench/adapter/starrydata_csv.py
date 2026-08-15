"""Parses rows from Starrydata's distributed CSV files (design §7.9 confirmed
schema: ``SID, DOI, composition, sample_id, figure_id, figure_name, prop_x,
prop_y, unit_x, unit_y, x, y, created_at, updated_at, project_names,
comments`` for ``*_curves.csv.gz``).

``x``/``y`` are JSON-array-literal strings (e.g. ``"[299.86,324.87]"``);
translating that external representation into plain float tuples is this
adapter's job, not the domain's.
"""

from __future__ import annotations

import csv
import gzip
import json
import pathlib
from collections.abc import Iterator

from real_chart_bench.usecase.starrydata_ingestion import ParsedCurveRow


def parse_curve_row(row: dict[str, str]) -> ParsedCurveRow:
    try:
        x_values = tuple(float(v) for v in json.loads(row["x"]))
        y_values = tuple(float(v) for v in json.loads(row["y"]))
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ValueError(f"Malformed curve row (SID={row.get('SID')!r}): {exc}") from exc

    if len(x_values) != len(y_values):
        raise ValueError(
            f"curve row x/y length mismatch (SID={row.get('SID')!r}): "
            f"{len(x_values)} vs {len(y_values)}"
        )

    prop_y = row.get("prop_y", "")
    unit_y = row.get("unit_y", "")
    series_label = f"{prop_y} ({unit_y})" if unit_y else prop_y

    return ParsedCurveRow(
        sid=row["SID"],
        doi=row.get("DOI", ""),
        figure_id=row["figure_id"],
        figure_name=row.get("figure_name", ""),
        series_label=series_label,
        x_values=x_values,
        y_values=y_values,
    )


def iter_curve_rows(csv_gz_path: pathlib.Path) -> Iterator[dict[str, str]]:
    """Reads a Starrydata ``*_curves.csv.gz`` file, yielding raw row dicts
    (before parse_curve_row translates them). Kept separate from parsing so
    tests can exercise parse_curve_row without touching the filesystem."""
    with gzip.open(csv_gz_path, mode="rt", newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)
