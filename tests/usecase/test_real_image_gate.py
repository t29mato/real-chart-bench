from real_chart_bench.domain.verified_pairing import VerificationStatus, VerifiedPairing
from real_chart_bench.usecase.real_image_gate import is_verified, select_verified_pairings


def _pairing(paper_id, figure_id, status):
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
