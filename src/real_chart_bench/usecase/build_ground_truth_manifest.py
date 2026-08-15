"""Builds FigureRecord + GroundTruthCurve entries for one already
license-classified paper (design §1.4, §7.9 Phase 3).

Ground truth (Starrydata XY values) is CC BY 4.0-redistributable regardless
of the source paper's figure-image license (design §7.1) — but the paper's
*figure image* still needs paper_license == REDISTRIBUTABLE before we'd
include it in the public dataset, which is why this function requires it up
front rather than silently accepting any paper.

Grouping key is ``figure_id`` (Starrydata's own composite-figure key), which
already disambiguates panels/curves without needing figure_name parsing —
see design §7.9 point 5. Pairing an extracted PDF image to a given
figure_id is a *separate*, still-open problem (§7.10) and is not attempted
here; FigureRecord.image_uri stays unset until that's solved.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from real_chart_bench.domain.collection_records import (
    FigureRecord,
    GroundTruthCurve,
    PaperRecord,
)
from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.dataset_split import assign_split
from real_chart_bench.domain.licensing import LicenseStatus
from real_chart_bench.usecase.starrydata_ingestion import ParsedCurveRow


def build_ground_truth_for_paper(
    paper: PaperRecord,
    curve_rows: Sequence[ParsedCurveRow],
    *,
    held_out_ratio: float,
) -> tuple[tuple[FigureRecord, ...], tuple[GroundTruthCurve, ...]]:
    if paper.license_status is not LicenseStatus.REDISTRIBUTABLE:
        raise ValueError(
            f"paper {paper.paper_id} is not REDISTRIBUTABLE "
            f"(status={paper.license_status}); refusing to include its figures"
        )

    if not curve_rows:
        return (), ()

    split = assign_split(paper.paper_id, held_out_ratio=held_out_ratio)

    rows_by_figure: dict[str, list[ParsedCurveRow]] = defaultdict(list)
    for row in curve_rows:
        rows_by_figure[row.figure_id].append(row)

    figures = tuple(
        FigureRecord(
            figure_id=figure_id,
            paper_id=paper.paper_id,
            figure_reference=rows[0].figure_name,
            split=split,
        )
        for figure_id, rows in rows_by_figure.items()
    )

    curves = tuple(
        GroundTruthCurve(
            curve_id=f"{paper.paper_id}-{row.figure_id}-{index}",
            figure_id=row.figure_id,
            x_values=row.x_values,
            y_values=row.y_values,
            x_scale=ScaleType.LINEAR,
            series_label=row.series_label,
        )
        for index, row in enumerate(curve_rows)
    )

    return figures, curves
