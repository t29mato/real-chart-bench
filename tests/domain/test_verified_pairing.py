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


def test_excluded_reason_defaults_to_none():
    pairing = VerifiedPairing(
        paper_id="18759",
        figure_id="12217",
        image_path="p04_embedded_4.jpg",
        panel_label="a",
        x_range=(200.0, 500.0),
        y_range=(25000.0, 135000.0),
        status=VerificationStatus.VERIFIED,
        verified_at="2026-08-16",
        evidence="ok",
    )
    assert pairing.excluded_reason is None


def test_verified_pairing_may_carry_an_excluded_reason():
    """A numerically-verified pairing whose source chart the current harness
    cannot correctly score (e.g. a log-y axis, HQ decision 2026-08-19: exclude
    from the real-image suite until y_scale support exists) is still a
    correct, trustworthy pairing -- it's just not includable in scoring yet.
    excluded_reason keeps that distinction explicit rather than overloading
    REJECTED (which means the pairing itself is wrong)."""
    pairing = VerifiedPairing(
        paper_id="47534",
        figure_id="49581",
        image_path="p03_embedded_2.jpg",
        panel_label=None,
        x_range=(873.15, 1173.15),
        y_range=(10.0, 1000.0),
        status=VerificationStatus.VERIFIED,
        verified_at="2026-08-19",
        evidence="numerically verified pairing, but the chart's y-axis is log-scale",
        excluded_reason="log-y axis chart; ExtractionTask has no y_scale support (see design 7.22)",
    )
    assert pairing.status is VerificationStatus.VERIFIED
    assert pairing.excluded_reason is not None
