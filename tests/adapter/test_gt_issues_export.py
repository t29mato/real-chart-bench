"""TDD for the CSV/JSON rendering of confirmed GT issues (design §7.48,
戦略メモ「柱G」, scripts/export/gt_issues.py).

CSV escaping matters concretely here: real evidence strings are long prose
and definitely contain commas ("... x-range 320-751K matches the chart's
300-800K axis, but y-range ... does not match ..."), and could plausibly
contain quotes or embedded newlines. A naive ",".join()-style writer would
silently corrupt the export; only Python's csv module (or an equivalent
quoting-aware writer) round-trips such text correctly.
"""

from __future__ import annotations

import csv
import io

from real_chart_bench.adapter.gt_issues_export import to_csv_text, to_json_export
from real_chart_bench.domain.verified_pairing import GtSuspectStatus, RejectionEvidence
from real_chart_bench.usecase.gt_issues import (
    GtIssueProperty,
    GtIssueRow,
    GtSuspectReviewSummary,
)


def _row(**overrides):
    defaults = dict(
        paper_id="42",
        figure_id="99",
        doi="10.1234/example",
        properties=(
            GtIssueProperty(prop_x="Temperature", unit_x="K", prop_y="Seebeck", unit_y="uV/K"),
        ),
        gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        evidence="plain evidence",
        rejection_evidence=RejectionEvidence(y_value_offset_magnitude=100.0, missing_series=True),
        verified_at="2026-08-30",
    )
    defaults.update(overrides)
    return GtIssueRow(**defaults)


# --- CSV escaping ------------------------------------------------------------


def test_csv_evidence_with_commas_round_trips():
    row = _row(evidence="off by roughly two orders of magnitude, checked all 6 images")
    text = to_csv_text([row])

    reader = list(csv.reader(io.StringIO(text)))
    header, data_row = reader[0], reader[1]

    assert data_row[header.index("evidence")] == (
        "off by roughly two orders of magnitude, checked all 6 images"
    )


def test_csv_evidence_with_quotes_round_trips():
    row = _row(evidence='the axis is printed as "log sigma [S x cm^-1]"')
    text = to_csv_text([row])

    reader = list(csv.reader(io.StringIO(text)))
    header, data_row = reader[0], reader[1]

    assert data_row[header.index("evidence")] == 'the axis is printed as "log sigma [S x cm^-1]"'


def test_csv_evidence_with_embedded_newline_round_trips():
    row = _row(evidence="line one\nline two")
    text = to_csv_text([row])

    reader = list(csv.reader(io.StringIO(text)))
    header, data_row = reader[0], reader[1]

    assert data_row[header.index("evidence")] == "line one\nline two"


def test_csv_has_exactly_one_data_row_per_issue():
    rows = [_row(paper_id="1", figure_id="10"), _row(paper_id="2", figure_id="20")]
    text = to_csv_text(rows)

    reader = list(csv.reader(io.StringIO(text)))
    assert len(reader) == 3  # header + 2 data rows


def test_csv_empty_rows_yields_header_only():
    text = to_csv_text([])
    reader = list(csv.reader(io.StringIO(text)))
    assert len(reader) == 1


def test_csv_includes_starrydata_identifiers_and_evidence_columns():
    row = _row()
    text = to_csv_text([row])
    header = next(csv.reader(io.StringIO(text)))

    for expected_col in (
        "paper_id",
        "figure_id",
        "doi",
        "properties",
        "gt_suspect_status",
        "evidence",
        "verified_at",
    ):
        assert expected_col in header


def test_csv_includes_structured_rejection_evidence_fields():
    row = _row()
    text = to_csv_text([row])
    header = next(csv.reader(io.StringIO(text)))

    for expected_col in (
        "axis_range_mismatch",
        "point_count_mismatch",
        "y_value_offset_magnitude",
        "missing_series",
    ):
        assert expected_col in header

    reader = csv.DictReader(io.StringIO(text))
    data_row = next(reader)
    assert data_row["y_value_offset_magnitude"] == "100.0"
    assert data_row["missing_series"] == "True"


def test_csv_handles_none_rejection_evidence():
    row = _row(rejection_evidence=None)
    text = to_csv_text([row])
    reader = csv.DictReader(io.StringIO(text))
    data_row = next(reader)
    assert data_row["axis_range_mismatch"] == ""
    assert data_row["y_value_offset_magnitude"] == ""


def test_csv_joins_multiple_properties_readably():
    row = _row(
        properties=(
            GtIssueProperty(prop_x="T", unit_x="K", prop_y="Seebeck", unit_y="uV/K"),
            GtIssueProperty(prop_x="T", unit_x="K", prop_y="Conductivity", unit_y="S/m"),
        )
    )
    text = to_csv_text([row])
    reader = csv.DictReader(io.StringIO(text))
    data_row = next(reader)
    assert "Seebeck" in data_row["properties"]
    assert "Conductivity" in data_row["properties"]


# --- JSON export --------------------------------------------------------------


def test_json_export_empty_issues_is_well_formed_with_zero_count():
    summary = GtSuspectReviewSummary(
        total_gt_suspect=3, human_confirmed=0, human_rejected=0, awaiting_human_review=3
    )
    payload = to_json_export([], summary)

    assert payload["issues"] == []
    assert payload["summary"]["human_confirmed"] == 0
    assert payload["summary"]["total_gt_suspect"] == 3
    assert payload["summary"]["awaiting_human_review"] == 3
    assert payload["license"] == "CC BY 4.0"


def test_json_export_never_emits_a_placeholder_row_when_empty():
    summary = GtSuspectReviewSummary(
        total_gt_suspect=3, human_confirmed=0, human_rejected=0, awaiting_human_review=3
    )
    payload = to_json_export([], summary)

    assert len(payload["issues"]) == 0


def test_json_export_row_shape():
    row = _row()
    summary = GtSuspectReviewSummary(
        total_gt_suspect=1, human_confirmed=1, human_rejected=0, awaiting_human_review=0
    )
    payload = to_json_export([row], summary)

    issue = payload["issues"][0]
    assert issue["paper_id"] == "42"
    assert issue["figure_id"] == "99"
    assert issue["doi"] == "10.1234/example"
    assert issue["gt_suspect_status"] == "human_confirmed"
    assert issue["evidence"] == "plain evidence"
    assert issue["rejection_evidence"] == {
        "axis_range_mismatch": None,
        "point_count_mismatch": None,
        "y_value_offset_magnitude": 100.0,
        "missing_series": True,
    }
    assert issue["properties"] == [
        {"prop_x": "Temperature", "unit_x": "K", "prop_y": "Seebeck", "unit_y": "uV/K"}
    ]


def test_json_export_rejection_evidence_none_serializes_to_null():
    row = _row(rejection_evidence=None)
    summary = GtSuspectReviewSummary(
        total_gt_suspect=1, human_confirmed=1, human_rejected=0, awaiting_human_review=0
    )
    payload = to_json_export([row], summary)

    assert payload["issues"][0]["rejection_evidence"] is None
