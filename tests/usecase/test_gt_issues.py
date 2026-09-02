"""TDD for scripts/export/gt_issues.py's selection/summarisation logic
(design §7.48, 戦略メモ「柱G」).

The single most important property under test: only entries whose
``is_confirmed_gt_error`` is True (i.e. ``rejection_category=gt_suspect`` AND
``gt_suspect_status=human_confirmed``) may ever appear in
``select_confirmed_gt_issues``'s output. An ``llm_flagged`` entry is a
*suspicion*, not a confirmed ground-truth error, and exporting it as one
would misrepresent an unverified machine judgment as human-confirmed to the
upstream Starrydata maintainers -- the exact failure this schema exists to
prevent (see domain/verified_pairing.py).
"""

from __future__ import annotations

from real_chart_bench.domain.verified_pairing import (
    GtSuspectStatus,
    RejectionCategory,
    RejectionEvidence,
    VerificationStatus,
    VerifiedPairing,
)
from real_chart_bench.usecase.gt_issues import (
    GtIssueProperty,
    select_confirmed_gt_issues,
    summarize_gt_suspect_review,
)


def _pairing(
    paper_id="1",
    figure_id="10",
    *,
    status=VerificationStatus.REJECTED,
    rejection_category=None,
    gt_suspect_status=None,
    rejection_evidence=None,
    evidence="test evidence",
    verified_at="2026-08-30",
):
    return VerifiedPairing(
        paper_id=paper_id,
        figure_id=figure_id,
        image_path="img.jpg" if status is VerificationStatus.VERIFIED else None,
        panel_label=None,
        x_range=(0.0, 1.0) if status is VerificationStatus.VERIFIED else None,
        y_range=(0.0, 1.0) if status is VerificationStatus.VERIFIED else None,
        status=status,
        verified_at=verified_at,
        evidence=evidence,
        rejection_category=rejection_category,
        gt_suspect_status=gt_suspect_status,
        rejection_evidence=rejection_evidence,
    )


# --- select_confirmed_gt_issues: the human_confirmed-only guarantee --------


def test_llm_flagged_entry_is_excluded():
    registry = [
        _pairing(
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.LLM_FLAGGED,
        )
    ]

    assert select_confirmed_gt_issues(registry) == []


def test_human_rejected_entry_is_excluded():
    registry = [
        _pairing(
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_REJECTED,
        )
    ]

    assert select_confirmed_gt_issues(registry) == []


def test_pairing_category_rejection_is_excluded():
    registry = [_pairing(rejection_category=RejectionCategory.PAIRING)]

    assert select_confirmed_gt_issues(registry) == []


def test_plain_rejected_entry_with_no_category_is_excluded():
    registry = [_pairing()]

    assert select_confirmed_gt_issues(registry) == []


def test_human_confirmed_entry_is_included():
    registry = [
        _pairing(
            paper_id="42",
            figure_id="99",
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        )
    ]

    rows = select_confirmed_gt_issues(registry)

    assert len(rows) == 1
    assert rows[0].paper_id == "42"
    assert rows[0].figure_id == "99"
    assert rows[0].gt_suspect_status is GtSuspectStatus.HUMAN_CONFIRMED


def test_verified_entry_with_human_confirmed_gt_suspect_flag_is_included():
    """design §7.48: a VERIFIED entry (pairing correct) can still carry a
    confirmed gt_suspect flag -- these ARE meant to be exported."""
    registry = [
        _pairing(
            status=VerificationStatus.VERIFIED,
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        )
    ]

    rows = select_confirmed_gt_issues(registry)

    assert len(rows) == 1


def test_empty_registry_yields_empty_list():
    assert select_confirmed_gt_issues([]) == []


def test_mixed_registry_only_confirmed_entries_pass():
    registry = [
        _pairing(paper_id="1", figure_id="10"),
        _pairing(
            paper_id="2",
            figure_id="20",
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.LLM_FLAGGED,
        ),
        _pairing(
            paper_id="3",
            figure_id="30",
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_REJECTED,
        ),
        _pairing(
            paper_id="4",
            figure_id="40",
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        ),
        _pairing(paper_id="5", figure_id="50", rejection_category=RejectionCategory.IMAGE),
    ]

    rows = select_confirmed_gt_issues(registry)

    assert [r.paper_id for r in rows] == ["4"]


# --- carries evidence --------------------------------------------------------


def test_carries_free_text_evidence_and_structured_rejection_evidence():
    ev = RejectionEvidence(y_value_offset_magnitude=100.0, missing_series=True)
    registry = [
        _pairing(
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
            evidence="the y-values are off by two orders of magnitude",
            rejection_evidence=ev,
        )
    ]

    rows = select_confirmed_gt_issues(registry)

    assert rows[0].evidence == "the y-values are off by two orders of magnitude"
    assert rows[0].rejection_evidence == ev


def test_rejection_evidence_none_is_handled_without_crashing():
    registry = [
        _pairing(
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
            rejection_evidence=None,
        )
    ]

    rows = select_confirmed_gt_issues(registry)

    assert rows[0].rejection_evidence is None


# --- Starrydata identifiers: property (prop_x/prop_y) + units --------------


def test_properties_pulled_from_ground_truth_json_and_deduped():
    registry = [
        _pairing(
            paper_id="1",
            figure_id="10",
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        )
    ]
    ground_truth = {
        "10": [
            {"prop_x": "Temperature", "unit_x": "K", "prop_y": "Seebeck", "unit_y": "uV/K"},
            {"prop_x": "Temperature", "unit_x": "K", "prop_y": "Seebeck", "unit_y": "uV/K"},
            {
                "prop_x": "Temperature",
                "unit_x": "K",
                "prop_y": "Electrical conductivity",
                "unit_y": "S/m",
            },
        ]
    }

    rows = select_confirmed_gt_issues(registry, ground_truth=ground_truth)

    assert rows[0].properties == (
        GtIssueProperty(prop_x="Temperature", unit_x="K", prop_y="Seebeck", unit_y="uV/K"),
        GtIssueProperty(
            prop_x="Temperature",
            unit_x="K",
            prop_y="Electrical conductivity",
            unit_y="S/m",
        ),
    )


def test_figure_not_in_ground_truth_json_yields_empty_properties():
    """A REJECTED entry's curves are never written into ground_truth.json
    (that file only holds VERIFIED/includable pairings' curves) -- the
    lookup must degrade gracefully, not crash, for exactly this case."""
    registry = [
        _pairing(
            paper_id="1",
            figure_id="not-in-gt",
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        )
    ]

    rows = select_confirmed_gt_issues(registry, ground_truth={})

    assert rows[0].properties == ()


def test_no_ground_truth_argument_defaults_to_empty_properties():
    registry = [
        _pairing(
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        )
    ]

    rows = select_confirmed_gt_issues(registry)

    assert rows[0].properties == ()


def test_rows_sorted_by_paper_id_then_figure_id():
    registry = [
        _pairing(
            paper_id="9",
            figure_id="1",
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        ),
        _pairing(
            paper_id="2",
            figure_id="5",
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        ),
        _pairing(
            paper_id="2",
            figure_id="1",
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        ),
    ]

    rows = select_confirmed_gt_issues(registry)

    assert [(r.paper_id, r.figure_id) for r in rows] == [("2", "1"), ("2", "5"), ("9", "1")]


# --- doi enrichment (optional) ----------------------------------------------


def test_doi_looked_up_from_papers_when_provided():
    registry = [
        _pairing(
            paper_id="1",
            figure_id="10",
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        )
    ]
    papers_by_id = {"1": {"paper_id": "1", "doi": "10.1234/example"}}

    rows = select_confirmed_gt_issues(registry, papers_by_id=papers_by_id)

    assert rows[0].doi == "10.1234/example"


def test_doi_is_none_when_paper_not_found():
    registry = [
        _pairing(
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        )
    ]

    rows = select_confirmed_gt_issues(registry, papers_by_id={})

    assert rows[0].doi is None


# --- summarize_gt_suspect_review --------------------------------------------


def test_summary_all_zero_on_empty_registry():
    summary = summarize_gt_suspect_review([])
    assert summary.total_gt_suspect == 0
    assert summary.human_confirmed == 0
    assert summary.human_rejected == 0
    assert summary.awaiting_human_review == 0


def test_summary_counts_current_real_shape_all_llm_flagged():
    """Mirrors the real current data: 3 gt_suspect entries, all
    llm_flagged, 0 human_confirmed -- the exact case this script must
    handle loudly and correctly (task hard requirement 4)."""
    registry = [
        _pairing(
            paper_id=str(i),
            figure_id=str(i),
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.LLM_FLAGGED,
        )
        for i in range(3)
    ]

    summary = summarize_gt_suspect_review(registry)

    assert summary.total_gt_suspect == 3
    assert summary.human_confirmed == 0
    assert summary.human_rejected == 0
    assert summary.awaiting_human_review == 3


def test_summary_mixed_statuses():
    registry = [
        _pairing(
            paper_id="1",
            figure_id="1",
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.LLM_FLAGGED,
        ),
        _pairing(
            paper_id="2",
            figure_id="2",
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        ),
        _pairing(
            paper_id="3",
            figure_id="3",
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_REJECTED,
        ),
        _pairing(paper_id="4", figure_id="4", rejection_category=RejectionCategory.PAIRING),
        _pairing(paper_id="5", figure_id="5"),
    ]

    summary = summarize_gt_suspect_review(registry)

    assert summary.total_gt_suspect == 3
    assert summary.human_confirmed == 1
    assert summary.human_rejected == 1
    assert summary.awaiting_human_review == 1
