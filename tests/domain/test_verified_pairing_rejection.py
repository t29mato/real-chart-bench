"""TDD for design §7.4x (戦略メモ「柱G」): rejection_category / GtSuspectStatus
on VerifiedPairing.

See src/real_chart_bench/domain/verified_pairing.py for the invariants these
tests pin down, and its module docstring / RejectionCategory docstring for
why "rejection_category required on every REJECTED entry" is enforced as a
queryable property (needs_rejection_classification) rather than an
unconditional constructor-time raise.
"""

import pytest

from real_chart_bench.domain.verified_pairing import (
    GtSuspectStatus,
    RejectionCategory,
    RejectionEvidence,
    VerificationStatus,
    VerifiedPairing,
)


def _base_kwargs(**overrides):
    kwargs = dict(
        paper_id="1",
        figure_id="1",
        image_path="a.jpg",
        panel_label=None,
        x_range=None,
        y_range=None,
        status=VerificationStatus.REJECTED,
        verified_at="2026-09-02",
        evidence="x",
    )
    kwargs.update(overrides)
    return kwargs


# --- RejectionCategory / GtSuspectStatus enums -----------------------------


def test_rejection_category_has_the_three_values():
    assert {c.value for c in RejectionCategory} == {"pairing", "image", "gt_suspect"}


def test_gt_suspect_status_has_the_three_lifecycle_values():
    assert {s.value for s in GtSuspectStatus} == {
        "llm_flagged",
        "human_confirmed",
        "human_rejected",
    }


def test_unknown_rejection_category_value_raises():
    with pytest.raises(ValueError):
        RejectionCategory("not-a-real-category")


def test_unknown_gt_suspect_status_value_raises():
    with pytest.raises(ValueError):
        GtSuspectStatus("not-a-real-status")


# --- llm_flagged must never read as a confirmed GT error --------------------


def test_llm_flagged_is_not_a_confirmed_gt_error():
    assert GtSuspectStatus.LLM_FLAGGED.is_confirmed_gt_error is False


def test_human_rejected_is_not_a_confirmed_gt_error():
    assert GtSuspectStatus.HUMAN_REJECTED.is_confirmed_gt_error is False


def test_human_confirmed_is_a_confirmed_gt_error():
    assert GtSuspectStatus.HUMAN_CONFIRMED.is_confirmed_gt_error is True


def test_pairing_reports_confirmed_gt_error_only_for_human_confirmed():
    pairing = VerifiedPairing(
        **_base_kwargs(
            status=VerificationStatus.VERIFIED,
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_CONFIRMED,
        )
    )
    assert pairing.is_confirmed_gt_error is True


def test_pairing_does_not_report_confirmed_gt_error_when_only_llm_flagged():
    """The critical owner rule: an llm_flagged-only entry must NEVER read as
    a confirmed GT error, because VLM readings are themselves error-prone."""
    pairing = VerifiedPairing(
        **_base_kwargs(
            status=VerificationStatus.VERIFIED,
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.LLM_FLAGGED,
        )
    )
    assert pairing.is_confirmed_gt_error is False


def test_pairing_with_no_gt_suspect_flag_at_all_is_not_a_confirmed_gt_error():
    pairing = VerifiedPairing(**_base_kwargs(status=VerificationStatus.VERIFIED))
    assert pairing.is_confirmed_gt_error is False


# --- rejection_category <-> status interaction ------------------------------


def test_rejected_entry_may_carry_a_pairing_category():
    pairing = VerifiedPairing(
        **_base_kwargs(rejection_category=RejectionCategory.PAIRING)
    )
    assert pairing.rejection_category is RejectionCategory.PAIRING
    assert pairing.needs_rejection_classification is False


def test_rejected_entry_may_carry_an_image_category():
    pairing = VerifiedPairing(**_base_kwargs(rejection_category=RejectionCategory.IMAGE))
    assert pairing.rejection_category is RejectionCategory.IMAGE


def test_rejected_entry_missing_category_needs_classification_but_does_not_raise():
    """Boundary case: a REJECTED entry with no rejection_category. This is
    the real state of some already-rejected registry entries whose evidence
    genuinely didn't support a single category (design §7.4x) -- guessing
    one would be worse than leaving it explicitly pending, so construction
    succeeds and the gap is exposed via needs_rejection_classification."""
    pairing = VerifiedPairing(**_base_kwargs(rejection_category=None))
    assert pairing.needs_rejection_classification is True


def test_verified_entry_without_rejection_category_does_not_need_classification():
    pairing = VerifiedPairing(**_base_kwargs(status=VerificationStatus.VERIFIED))
    assert pairing.needs_rejection_classification is False


def test_verified_entry_may_carry_gt_suspect_category():
    """A pairing can be correctly matched (VERIFIED) and still have a
    suspect ground truth -- the two are orthogonal (design §7.4x)."""
    pairing = VerifiedPairing(
        **_base_kwargs(
            status=VerificationStatus.VERIFIED,
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.LLM_FLAGGED,
        )
    )
    assert pairing.status is VerificationStatus.VERIFIED
    assert pairing.rejection_category is RejectionCategory.GT_SUSPECT


@pytest.mark.parametrize("category", [RejectionCategory.PAIRING, RejectionCategory.IMAGE])
def test_verified_entry_cannot_carry_a_pairing_or_image_category(category):
    """A VERIFIED entry cannot claim 'wrong figure matched' or 'image
    unreadable' -- either of those would mean it should be REJECTED, not
    VERIFIED. Only GT_SUSPECT is orthogonal to pairing correctness."""
    with pytest.raises(ValueError):
        VerifiedPairing(
            **_base_kwargs(status=VerificationStatus.VERIFIED, rejection_category=category)
        )


# --- gt_suspect_status <-> rejection_category interaction -------------------


def test_gt_suspect_category_without_status_raises():
    with pytest.raises(ValueError):
        VerifiedPairing(
            **_base_kwargs(
                rejection_category=RejectionCategory.GT_SUSPECT,
                gt_suspect_status=None,
            )
        )


@pytest.mark.parametrize("category", [RejectionCategory.PAIRING, RejectionCategory.IMAGE, None])
def test_gt_suspect_status_on_a_non_gt_suspect_category_raises(category):
    with pytest.raises(ValueError):
        VerifiedPairing(
            **_base_kwargs(
                rejection_category=category,
                gt_suspect_status=GtSuspectStatus.LLM_FLAGGED,
            )
        )


def test_gt_suspect_category_with_status_is_valid():
    pairing = VerifiedPairing(
        **_base_kwargs(
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.HUMAN_REJECTED,
        )
    )
    assert pairing.gt_suspect_status is GtSuspectStatus.HUMAN_REJECTED


# --- rejection_evidence ------------------------------------------------------


def test_rejection_evidence_defaults_to_none():
    pairing = VerifiedPairing(**_base_kwargs())
    assert pairing.rejection_evidence is None


def test_rejection_evidence_holds_structured_fields():
    evidence = RejectionEvidence(
        axis_range_mismatch=False,
        point_count_mismatch=None,
        y_value_offset_magnitude=100.0,
        missing_series=False,
    )
    pairing = VerifiedPairing(
        **_base_kwargs(
            rejection_category=RejectionCategory.GT_SUSPECT,
            gt_suspect_status=GtSuspectStatus.LLM_FLAGGED,
            rejection_evidence=evidence,
        )
    )
    assert pairing.rejection_evidence.y_value_offset_magnitude == 100.0
    assert pairing.rejection_evidence.axis_range_mismatch is False


def test_free_text_evidence_field_is_kept_alongside_rejection_evidence():
    pairing = VerifiedPairing(
        **_base_kwargs(
            evidence="original free-text evidence, unchanged",
            rejection_category=RejectionCategory.IMAGE,
            rejection_evidence=RejectionEvidence(missing_series=True),
        )
    )
    assert pairing.evidence == "original free-text evidence, unchanged"
    assert pairing.rejection_evidence.missing_series is True
