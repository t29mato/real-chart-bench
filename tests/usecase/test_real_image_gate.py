from real_chart_bench.domain.verified_pairing import VerificationStatus, VerifiedPairing
from real_chart_bench.usecase.real_image_gate import is_verified, select_verified_pairings


def _pairing(paper_id, figure_id, status, *, excluded_reason=None):
    return VerifiedPairing(
        paper_id=paper_id,
        figure_id=figure_id,
        image_path="img.jpg" if status is VerificationStatus.VERIFIED else None,
        panel_label="a",
        x_range=(0.0, 1.0) if status is VerificationStatus.VERIFIED else None,
        y_range=(0.0, 1.0) if status is VerificationStatus.VERIFIED else None,
        status=status,
        verified_at="2026-08-16",
        evidence="test",
        excluded_reason=excluded_reason,
    )


def test_select_verified_pairings_excludes_rejected():
    registry = [
        _pairing("1", "10", VerificationStatus.VERIFIED),
        _pairing("2", "20", VerificationStatus.REJECTED),
    ]

    selected = select_verified_pairings(registry)

    assert [p.paper_id for p in selected] == ["1"]


def test_select_verified_pairings_empty_registry_is_empty():
    assert select_verified_pairings([]) == []


def test_select_verified_pairings_all_rejected_is_empty():
    registry = [_pairing("1", "10", VerificationStatus.REJECTED)]
    assert select_verified_pairings(registry) == []


def test_is_verified_true_only_for_verified_status():
    registry = [
        _pairing("1", "10", VerificationStatus.VERIFIED),
        _pairing("2", "20", VerificationStatus.REJECTED),
    ]

    assert is_verified(registry, paper_id="1", figure_id="10") is True
    assert is_verified(registry, paper_id="2", figure_id="20") is False


def test_is_verified_false_for_unknown_pairing():
    registry = [_pairing("1", "10", VerificationStatus.VERIFIED)]

    assert is_verified(registry, paper_id="99", figure_id="99") is False


def test_select_verified_pairings_excludes_entries_with_an_excluded_reason():
    """HQ decision 2026-08-19: a numerically-verified pairing the current
    harness can't score correctly (e.g. log-y axis) must not enter the
    real-image evaluation suite, even though its pairing status is VERIFIED
    -- see domain/verified_pairing.py's excluded_reason field."""
    registry = [
        _pairing("1", "10", VerificationStatus.VERIFIED),
        _pairing("2", "20", VerificationStatus.VERIFIED, excluded_reason="log-y axis"),
    ]

    selected = select_verified_pairings(registry)

    assert [p.paper_id for p in selected] == ["1"]


def test_is_verified_stays_true_for_a_verified_but_excluded_pairing():
    """is_verified answers 'is this pairing correct', independent of whether
    it's currently includable in the scoring suite -- excluded_reason is not
    the same claim as REJECTED."""
    registry = [_pairing("1", "10", VerificationStatus.VERIFIED, excluded_reason="log-y axis")]

    assert is_verified(registry, paper_id="1", figure_id="10") is True
