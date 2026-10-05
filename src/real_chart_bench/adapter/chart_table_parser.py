"""Parses the text tables that chart-to-table models (DePlot, UniChart,
ChartGemma, Granite Vision, TinyChart -- design §7.68, §7.75) produce.

The rules are fixed and deliberately literal, so a score measures the model,
not the parser's generosity:
  - rows: DePlot's "<0x0A>" token or newlines (UniChart: " & ", passed in);
  - cells: "|" when the row has one, else CSV;
  - first column is x, every other column is a series named by the header
    row (the last row before the data whose x and values are all non-numeric),
    or "series_<n>";
  - a row whose x is not a number is dropped; `n_rows_dropped` counts the
    ones that carried numeric values (data whose x the model did not give as
    a number -- headers and titles are not counted); exact duplicate rows are
    dropped;
  - an empty or non-numeric cell leaves that point out of its series only;
  - nothing is interpolated or extrapolated.
A table with no numeric x row is unparseable (`parsed` False).
"""

from __future__ import annotations

import csv
import io
import math
import re
from dataclasses import dataclass

_SCI = re.compile(r"^([+-]?\d*\.?\d+)\s*[x×*]\s*10\^\(?([+-]?\d+)\)?$", re.I)
_POW = re.compile(r"^10\^\(?([+-]?\d+)\)?$")
_THOUSANDS = re.compile(r"^[+-]?\d{1,3}(,\d{3})+(\.\d+)?$")
_MD_RULE = re.compile(r"^\|?\s*:?-{2,}")


@dataclass(frozen=True)
class TableSeries:
    label: str
    x: tuple[float, ...]
    y: tuple[float, ...]


@dataclass(frozen=True)
class ParsedTable:
    series: tuple[TableSeries, ...]
    n_rows_dropped: int

    @property
    def parsed(self) -> bool:
        return bool(self.series)


def parse_number(text: str) -> float | None:
    s = (text or "").strip().replace("−", "-").replace("–", "-").rstrip("%").strip()
    if not s:
        return None
    if _THOUSANDS.match(s):
        s = s.replace(",", "")
    m = _SCI.match(s)
    if m:
        value = float(m.group(1)) * 10.0 ** int(m.group(2))
    else:
        m = _POW.match(s)
        if m:
            value = 10.0 ** int(m.group(1))
        else:
            try:
                value = float(s)
            except ValueError:
                return None
    return value if math.isfinite(value) else None


def _split_rows(text: str, row_separator: str | None) -> list[str]:
    text = text.replace("<0x0A>", "\n")
    if row_separator:
        text = text.replace(row_separator, "\n")
    return [r for r in (line.strip() for line in text.splitlines()) if r]


def _cells(row: str) -> list[str]:
    if "|" in row:
        parts = row.strip().strip("|").split("|")
        return [p.strip() for p in parts]
    return [c.strip() for c in next(csv.reader(io.StringIO(row)))]


def parse_chart_table(text: str | None, row_separator: str | None = None) -> ParsedTable:
    if not text:
        return ParsedTable(series=(), n_rows_dropped=0)
    header: list[str] | None = None
    data: list[tuple[float, list[str]]] = []
    seen: set[tuple[str, ...]] = set()
    dropped = 0
    for row in _split_rows(text, row_separator):
        if _MD_RULE.match(row):
            continue
        cells = _cells(row)
        if len(cells) < 2:
            continue  # a title line or stray text: no values to lose
        x = parse_number(cells[0])
        if x is None:
            if any(parse_number(c) is not None for c in cells[1:]):
                dropped += 1
            elif not data:
                header = cells
            continue
        key = tuple(cells)
        if key in seen:
            continue
        seen.add(key)
        data.append((x, cells[1:]))

    if not data:
        return ParsedTable(series=(), n_rows_dropped=dropped)
    width = max(len(values) for _, values in data)
    series = []
    for j in range(width):
        points = [
            (x, y)
            for x, values in data
            if j < len(values) and (y := parse_number(values[j])) is not None
        ]
        if not points:
            continue
        label = header[j + 1] if header and j + 1 < len(header) and header[j + 1] else ""
        series.append(
            TableSeries(
                label=label or f"series_{j}",
                x=tuple(p[0] for p in points),
                y=tuple(p[1] for p in points),
            )
        )
    return ParsedTable(series=tuple(series), n_rows_dropped=dropped)
