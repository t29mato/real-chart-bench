import pytest

from real_chart_bench.usecase.pairing_threshold_validation import (
    CORRECT,
    SIBLING,
    UNLABELLED,
    WRONG,
    AMBIGUOUS,
    LabelledAssignment,
    adopt_stats,
    clopper_pearson_upper,
    label_assignment,
    normalise_reference,
    sweep_adoption_rules,
)

CROP_A = "data/verified_pairs/crops/1/a.png"
CROP_B = "data/verified_pairs/crops/1/b.png"
FULL = "data/verified_pairs/images/1/p01_embedded_0.jpg"


def _label(figure, image, verified=None, rejected=None, owners=None, **kw):
    return label_assignment(
        ("1", figure),
        image,
        verified=verified or {},
        rejected=rejected or {},
        owners=owners or {},
        **kw,
    )


def test_assignment_to_the_verified_image_is_correct():
    assert _label("10", CROP_A, verified={("1", "10"): CROP_A}) == CORRECT


def test_assignment_to_another_image_of_the_same_kind_is_wrong():
    assert _label("10", CROP_B, verified={("1", "10"): CROP_A}) == WRONG


def test_crop_versus_full_image_cannot_be_judged_and_is_ambiguous():
    assert _label("10", FULL, verified={("1", "10"): CROP_A}) == AMBIGUOUS
    assert _label("10", CROP_A, verified={("1", "10"): FULL}) == AMBIGUOUS


def test_assignment_to_the_image_a_rejected_entry_recorded_is_wrong():
    assert _label("11", CROP_A, rejected={("1", "11"): frozenset({CROP_A})}) == WRONG


def test_rejected_figure_on_some_other_image_is_unlabelled():
    assert _label("11", CROP_B, rejected={("1", "11"): frozenset({CROP_A})}) == UNLABELLED


def test_unverified_figure_taking_an_image_owned_by_another_figure_is_wrong():
    assert _label("12", CROP_A, owners={CROP_A: frozenset({"10"})}) == WRONG


def test_unverified_figure_on_an_image_nobody_owns_is_unlabelled():
    assert _label("12", CROP_B, owners={CROP_A: frozenset({"10"})}) == UNLABELLED


def test_unverified_figure_on_an_image_it_co_owns_is_not_wrong():
    # (several verified figures can share one image; an unverified id listed there is not a steal)
    assert _label("10", CROP_A, owners={CROP_A: frozenset({"10", "13"})}) == UNLABELLED


def _a(outcome, s=0.9, m=0.5, c=0.5):
    return LabelledAssignment(outcome=outcome, S=s, M=m, contrast=c)


def test_adopt_stats_counts_only_assignments_that_pass_all_three_thresholds():
    items = [_a(CORRECT), _a(CORRECT, s=0.7), _a(WRONG, m=0.1), _a(CORRECT, c=0.1)]
    st = adopt_stats(items, s_min=0.8, m_min=0.3, c_min=0.35, n_verified=4)
    assert (st.adopted, st.correct, st.wrong) == (1, 1, 0)
    assert st.recall == 0.25


def test_adopt_stats_boundary_values_are_inclusive():
    st = adopt_stats([_a(CORRECT, s=0.8, m=0.3, c=0.35)], 0.8, 0.3, 0.35, n_verified=1)
    assert st.adopted == 1


def test_precision_ignores_unlabelled_and_pessimistic_precision_counts_ambiguous_as_wrong():
    items = [_a(CORRECT), _a(CORRECT), _a(AMBIGUOUS), _a(UNLABELLED)]
    st = adopt_stats(items, 0.0, 0.0, 0.0, n_verified=4)
    assert st.precision == 1.0
    assert st.precision_pessimistic == pytest.approx(2 / 3)
    assert st.unlabelled == 1 and st.ambiguous == 1


def test_precision_is_none_when_nothing_labelled_is_adopted():
    st = adopt_stats([_a(UNLABELLED)], 0.0, 0.0, 0.0, n_verified=3)
    assert st.precision is None and st.precision_pessimistic is None


def test_empty_input_adopts_nothing():
    st = adopt_stats([], 0.8, 0.3, 0.35, n_verified=0)
    assert st.adopted == 0 and st.recall == 0.0


def test_clopper_pearson_upper_bound_with_no_errors_is_the_rule_of_three():
    assert clopper_pearson_upper(0, 91) == pytest.approx(0.032, abs=0.002)
    assert clopper_pearson_upper(0, 299) < 0.01 + 1e-3


def test_clopper_pearson_upper_bound_grows_with_errors_and_is_one_when_all_wrong():
    assert clopper_pearson_upper(2, 50) > clopper_pearson_upper(1, 50)
    assert clopper_pearson_upper(5, 5) == 1.0
    assert clopper_pearson_upper(0, 0) == 1.0


def test_sweep_picks_the_rule_with_most_correct_among_those_meeting_the_target():
    items = [_a(CORRECT, s=0.95), _a(CORRECT, s=0.85), _a(WRONG, s=0.75), _a(CORRECT, s=0.7)]
    rows = sweep_adoption_rules(
        items, s_grid=[0.7, 0.8, 0.9], m_grid=[0.0], c_grid=[0.0], n_verified=4
    )
    ok = [r for r in rows if r.stats.precision_pessimistic is not None
          and r.stats.precision_pessimistic >= 0.99]
    best = max(ok, key=lambda r: r.stats.correct)
    assert best.s_min == 0.8 and best.stats.correct == 2
    assert len(rows) == 3


def test_unverified_figure_with_the_owners_reference_is_a_sibling_digitization():
    kw = dict(owners={CROP_A: frozenset({"10"})}, reference="8(a)",
              owner_references={CROP_A: frozenset({"8a"})})
    assert _label("12", CROP_A, **kw) == SIBLING


def test_a_different_reference_on_an_owned_image_stays_wrong():
    kw = dict(owners={CROP_A: frozenset({"10"})}, reference="5f",
              owner_references={CROP_A: frozenset({"4f"})})
    assert _label("12", CROP_A, **kw) == WRONG


def test_sibling_counts_with_the_ambiguous_in_the_pessimistic_precision_only():
    items = [_a(CORRECT), _a(SIBLING)]
    st = adopt_stats(items, 0.0, 0.0, 0.0, n_verified=2)
    assert st.precision == 1.0 and st.precision_pessimistic == 0.5 and st.ambiguous == 1


def test_normalise_reference_ignores_case_spacing_and_punctuation():
    assert normalise_reference("Fig. 4(F)") == normalise_reference("fig4f")


def test_normalise_reference_drops_trailing_quantity_words():
    assert normalise_reference("8a sigma") == "8a"
    assert normalise_reference("5a Converted") == "5a"
    assert normalise_reference("FIGURE4(ZT)") == "4zt"
    assert normalise_reference("") == ""
