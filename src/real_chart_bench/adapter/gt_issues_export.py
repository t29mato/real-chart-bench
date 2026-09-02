"""Renders usecase/gt_issues.py's GtIssueRow list to the two on-disk
export formats (data/gt_issues/gt_issues.csv, data/gt_issues/gt_issues.json)
(design §7.48, 戦略メモ「柱G」).

CSV escaping is delegated entirely to the stdlib ``csv`` module (quoting
commas/quotes/embedded newlines correctly) rather than hand-joining strings
-- the real evidence strings are long prose and definitely contain commas.

This module does no filesystem I/O itself (that's scripts/export/
gt_issues.py's job) -- same split as adapter/verified_pairing_registry.py's
serialize_entry, which shapes dicts without touching disk.
"""

from __future__ import annotations

import csv
import io
from collections.abc import Sequence
from dataclasses import asdict
from typing import Any

from real_chart_bench.usecase.gt_issues import GtIssueRow, GtSuspectReviewSummary

CSV_HEADER = (
    "paper_id",
    "figure_id",
    "doi",
    "properties",
    "gt_suspect_status",
    "verified_at",
    "axis_range_mismatch",
    "point_count_mismatch",
    "y_value_offset_magnitude",
    "missing_series",
    "evidence",
)


def _format_properties(row: GtIssueRow) -> str:
    return " | ".join(
        f"{p.prop_x} [{p.unit_x}] vs {p.prop_y} [{p.unit_y}]" for p in row.properties
    )


def _rejection_evidence_columns(row: GtIssueRow) -> dict[str, Any]:
    if row.rejection_evidence is None:
        return {
            "axis_range_mismatch": "",
            "point_count_mismatch": "",
            "y_value_offset_magnitude": "",
            "missing_series": "",
        }
    ev = asdict(row.rejection_evidence)
    return {k: ("" if v is None else v) for k, v in ev.items()}


def to_csv_text(rows: Sequence[GtIssueRow]) -> str:
    """Header-only text (no data rows) when ``rows`` is empty -- a valid,
    well-formed empty export, never a placeholder row (task hard
    requirement 4)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(CSV_HEADER)
    for row in rows:
        ev_cols = _rejection_evidence_columns(row)
        writer.writerow(
            [
                row.paper_id,
                row.figure_id,
                row.doi or "",
                _format_properties(row),
                row.gt_suspect_status.value,
                row.verified_at,
                ev_cols["axis_range_mismatch"],
                ev_cols["point_count_mismatch"],
                ev_cols["y_value_offset_magnitude"],
                ev_cols["missing_series"],
                row.evidence,
            ]
        )
    return buf.getvalue()


def to_json_export(
    rows: Sequence[GtIssueRow], summary: GtSuspectReviewSummary
) -> dict[str, Any]:
    """``rows=[]`` renders ``"issues": []`` plus a ``summary`` block that
    still reports the true total/awaiting counts -- a valid, well-formed
    empty export, never a placeholder row (task hard requirement 4)."""
    return {
        "license": "CC BY 4.0",
        "source": "real-chart-bench (data/verified_pairs/registry.json)",
        "note": (
            "One-way export of human-confirmed Starrydata ground-truth issues. "
            "Only entries a human has reviewed and confirmed as wrong are listed "
            "here -- see summary.awaiting_human_review for suspicions an LLM "
            "raised that no human has adjudicated yet. Nothing in this repository "
            "writes back to starrydata2.org; see data/gt_issues/README.md."
        ),
        "summary": asdict(summary),
        "issues": [
            {
                "paper_id": row.paper_id,
                "figure_id": row.figure_id,
                "doi": row.doi,
                "properties": [asdict(p) for p in row.properties],
                "gt_suspect_status": row.gt_suspect_status.value,
                "verified_at": row.verified_at,
                "evidence": row.evidence,
                "rejection_evidence": (
                    asdict(row.rejection_evidence) if row.rejection_evidence is not None else None
                ),
            }
            for row in rows
        ],
    }
