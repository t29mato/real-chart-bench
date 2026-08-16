from real_chart_bench.domain.verified_pairing import VerificationStatus, VerifiedPairing


def test_verified_pairing_holds_evidence_fields():
    pairing = VerifiedPairing(
        paper_id="18759",
        figure_id="12217",
        image_path="p04_embedded_4.jpg",
        panel_label="a",
        x_range=(200.0, 500.0),
        y_range=(25000.0, 135000.0),
        status=VerificationStatus.VERIFIED,
        verified_at="2026-08-16",
        evidence="axis ranges and series count cross-checked against raw Starrydata values",
    )
    assert pairing.status is VerificationStatus.VERIFIED
    assert pairing.evidence


def test_rejected_pairing_may_omit_image_and_calibration():
    pairing = VerifiedPairing(
        paper_id="47139",
        figure_id="48697",
        image_path="p05_embedded_7.jpg",
        panel_label="b",
        x_range=None,
        y_range=None,
        status=VerificationStatus.REJECTED,
        verified_at="2026-08-16",
        evidence="y-values order of magnitude inconsistent with the chart's printed axis range",
    )
    assert pairing.status is VerificationStatus.REJECTED
    assert pairing.x_range is None
