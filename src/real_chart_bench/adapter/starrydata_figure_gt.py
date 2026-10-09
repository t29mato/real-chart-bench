"""Starrydata's hand-digitized curves grouped per paper and figure, with the
axis quantities and units each figure was digitized in -- what the pairing
step projects onto a frame (docs/design/pairing-automation.md §12).

The committed manifest (data/manifest/v0/curves.json) keeps only counts, so
the values are read from Starrydata's own distributed file
(``ThermoelectricMaterials_curves.csv.gz``, local cache; no network).
``curve_id`` is the manifest's: ``<paper>-<figure>-<index of the row among
the paper's rows>`` (usecase/build_ground_truth_manifest.py).
"""

from __future__ import annotations

import json
import pathlib
from collections import Counter, defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

from real_chart_bench.adapter.starrydata_csv import iter_curve_rows


@dataclass(frozen=True)
class CurveGt:
    curve_id: str
    composition: str
    xs: tuple[float, ...]
    ys: tuple[float, ...]


@dataclass(frozen=True)
class FigureGt:
    paper_id: str
    figure_id: str
    figure_name: str
    prop_x: str
    unit_x: str
    prop_y: str
    unit_y: str
    curves: tuple[CurveGt, ...]  # the figure's dominant (prop_x, unit_x, prop_y, unit_y)
    n_other_axis_curves: int  # digitized in another quantity (twin axis, second panel)


def _axes(row: dict[str, str]) -> tuple[str, str, str, str]:
    return (row["prop_x"], row["unit_x"], row["prop_y"], row["unit_y"])


def load_figure_gt(
    csv_gz_path: pathlib.Path, paper_ids: Iterable[str]
) -> dict[str, dict[str, FigureGt]]:
    wanted = set(paper_ids)
    per_paper_index: dict[str, int] = defaultdict(int)
    rows_by_figure: dict[tuple[str, str], list[tuple[int, dict[str, str], list, list]]] = (
        defaultdict(list)
    )
    for row in iter_curve_rows(csv_gz_path):
        sid = row["SID"]
        if sid not in wanted:
            continue
        try:
            xs = [float(v) for v in json.loads(row["x"])]
            ys = [float(v) for v in json.loads(row["y"])]
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
        if len(xs) != len(ys):
            continue
        index = per_paper_index[sid]
        per_paper_index[sid] += 1
        if xs:
            rows_by_figure[(sid, row["figure_id"])].append((index, row, xs, ys))

    out: dict[str, dict[str, FigureGt]] = defaultdict(dict)
    for (sid, fid), rows in rows_by_figure.items():
        counts = Counter(_axes(r) for _, r, _, _ in rows)
        key = max(counts, key=lambda k: (counts[k], k))
        kept = [(i, r, xs, ys) for i, r, xs, ys in rows if _axes(r) == key]
        out[sid][fid] = FigureGt(
            paper_id=sid,
            figure_id=fid,
            figure_name=rows[0][1].get("figure_name", ""),
            prop_x=key[0],
            unit_x=key[1],
            prop_y=key[2],
            unit_y=key[3],
            curves=tuple(
                CurveGt(f"{sid}-{fid}-{i}", (r.get("composition") or "").strip(),
                        tuple(xs), tuple(ys))
                for i, r, xs, ys in kept
            ),
            n_other_axis_curves=len(rows) - len(kept),
        )
    return dict(out)
