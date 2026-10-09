"""Calibration/projection recall fixes found by diagnosing the 47 verified
figures the automatic pairing missed (docs/design/pairing-automation.md
section 12.7; docs/experiments/2026-10-10-pairing-calibration-recall.md)."""

from __future__ import annotations

from real_chart_bench.domain.tick_calibration import (
    Transform,
    candidate_transforms,
    split_merged_labels,
)


def test_twelve_three_digit_labels_split_into_twelve_not_three_huge_numbers():
    # "220230240250" / "260270280290" / "300310320330" is also an arithmetic
    # progression (step 40040040040); the real run is the one with most labels
    word = "220230240250260270280290300310320330"
    assert split_merged_labels(word) == [
        "220", "230", "240", "250", "260", "270", "280", "290", "300", "310", "320", "330",
    ]


def test_a_longer_decomposition_is_preferred_only_when_it_is_still_one_progression():
    assert split_merged_labels("708090100110") == ["70", "80", "90", "100", "110"]
    assert split_merged_labels("-20-1001020") == ["-20", "-10", "0", "10", "20"]


def test_carrier_concentrations_printed_in_1e21_cm3_are_reachable_from_si():
    # stored 2.0e26 m^-3, printed 2.0 (x 10^20 cm^-3): a factor of 1e-26
    names = {t.name for t in candidate_transforms("Carrier concentration", "m^(-3)")}
    assert Transform("scale", -26).name in names
    assert Transform("scale", -27).name in names


def test_scale_range_is_symmetric_and_bounded():
    scales = [t.param for t in candidate_transforms("X", "1") if t.kind == "scale"]
    assert min(scales) == -30 and max(scales) == 30
    assert len(scales) == 61
