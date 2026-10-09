"""Per-paper Hungarian assignment of plot frames to Starrydata figures
(docs/design/pairing-automation.md §12)."""

from __future__ import annotations

import pytest

from real_chart_bench.domain.pairing_assignment import (
    HIGH_MARGIN,
    HIGH_MIN_CONTRAST,
    HIGH_MIN_HIT,
    MIN_CONTRAST,
    MIN_HIT,
    Lane,
    PairScore,
    assign_frames_to_figures,
)


def score(frame: str, figure: str, hit: float, null: float = 0.1, inside: float = 1.0):
    return PairScore(frame_id=frame, figure_id=figure, hit=hit, null=null, inside=inside)


class TestPairScore:
    def test_contrast_is_hit_minus_null(self):
        assert score("f", "g", 0.9, 0.2).contrast == pytest.approx(0.7)

    def test_ink_everywhere_is_not_eligible_even_with_high_hit(self):
        assert not score("f", "g", 0.95, 0.9).eligible

    def test_low_hit_is_not_eligible(self):
        assert not score("f", "g", MIN_HIT - 0.01, 0.0).eligible

    def test_exactly_at_both_floors_is_eligible(self):
        assert score("f", "g", MIN_HIT, MIN_HIT - MIN_CONTRAST).eligible


class TestAssign:
    def test_empty_inputs_give_nothing(self):
        assert assign_frames_to_figures([], [], []) == []

    def test_single_eligible_pair_is_assigned_with_full_margin(self):
        out = assign_frames_to_figures(["f"], ["g"], [score("f", "g", 0.9)])
        assert len(out) == 1
        assert (out[0].frame_id, out[0].figure_id) == ("f", "g")
        assert out[0].margin == 1.0

    def test_ineligible_pair_is_left_unassigned(self):
        assert assign_frames_to_figures(["f"], ["g"], [score("f", "g", 0.2, 0.0)]) == []

    def test_one_figure_goes_to_at_most_one_frame(self):
        out = assign_frames_to_figures(
            ["f1", "f2"], ["g"], [score("f1", "g", 0.9), score("f2", "g", 0.7)]
        )
        assert [(a.frame_id, a.figure_id) for a in out] == [("f1", "g")]

    def test_one_frame_goes_to_at_most_one_figure(self):
        out = assign_frames_to_figures(
            ["f"], ["g1", "g2"], [score("f", "g1", 0.7), score("f", "g2", 0.95)]
        )
        assert [(a.frame_id, a.figure_id) for a in out] == [("f", "g2")]

    def test_global_optimum_beats_greedy(self):
        # greedy takes (f1,g1)=0.9 and leaves f2 with nothing; the optimum pairs
        # f1-g2 and f2-g1 (0.85 + 0.85) and loses no figure
        scores = [
            score("f1", "g1", 0.90), score("f1", "g2", 0.85),
            score("f2", "g1", 0.85),
        ]
        out = assign_frames_to_figures(["f1", "f2"], ["g1", "g2"], scores)
        assert {(a.frame_id, a.figure_id) for a in out} == {("f1", "g2"), ("f2", "g1")}

    def test_margin_is_gap_to_best_rival_frame_for_the_same_figure(self):
        scores = [score("f1", "g", 0.95), score("f2", "g", 0.90)]
        out = assign_frames_to_figures(["f1", "f2"], ["g"], scores)
        assert out[0].margin == pytest.approx(0.05)

    def test_margin_is_gap_to_best_rival_figure_for_the_same_frame(self):
        scores = [score("f", "g1", 0.95), score("f", "g2", 0.80)]
        out = assign_frames_to_figures(["f"], ["g1", "g2"], scores)
        assert out[0].margin == pytest.approx(0.15)

    def test_ineligible_rivals_do_not_shrink_the_margin(self):
        scores = [score("f1", "g1", 0.9), score("f2", "g1", 0.2, 0.0)]
        out = assign_frames_to_figures(["f1", "f2"], ["g1"], scores)
        assert out[0].margin == 1.0

    def test_scores_for_unknown_ids_are_ignored(self):
        out = assign_frames_to_figures(["f"], ["g"], [score("zz", "g", 0.9)])
        assert out == []


class TestLane:
    def test_clear_winner_on_clean_ink_is_high(self):
        out = assign_frames_to_figures(["f"], ["g"], [score("f", "g", 0.9, 0.1)])
        assert out[0].lane is Lane.HIGH

    def test_hit_just_under_high_threshold_goes_to_review(self):
        out = assign_frames_to_figures(["f"], ["g"], [score("f", "g", HIGH_MIN_HIT - 0.01, 0.0)])
        assert out[0].lane is Lane.REVIEW

    def test_narrow_margin_goes_to_review(self):
        scores = [score("f1", "g", 0.95), score("f2", "g", 0.95 - HIGH_MARGIN + 0.05)]
        out = assign_frames_to_figures(["f1", "f2"], ["g"], scores)
        assert out[0].lane is Lane.REVIEW

    def test_weak_contrast_goes_to_review_even_at_high_hit(self):
        hit = 0.95
        out = assign_frames_to_figures(
            ["f"], ["g"], [score("f", "g", hit, hit - HIGH_MIN_CONTRAST + 0.05)]
        )
        assert out[0].lane is Lane.REVIEW

    def test_boundary_values_are_high(self):
        out = assign_frames_to_figures(
            ["f"], ["g"], [score("f", "g", HIGH_MIN_HIT, HIGH_MIN_HIT - HIGH_MIN_CONTRAST)]
        )
        assert out[0].lane is Lane.HIGH
