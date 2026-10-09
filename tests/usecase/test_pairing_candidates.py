from __future__ import annotations

from dataclasses import dataclass

from real_chart_bench.domain.pairing_assignment import PairScore
from real_chart_bench.usecase.pairing_candidates import (
    FigureInfo,
    FrameInfo,
    decide_paper,
    prefer_embedded_frames,
)


@dataclass
class Scored:
    pair: PairScore
    x_transform: str = "x1e0"
    y_transform: str = "x1e0"


def frame(fid):
    return FrameInfo(frame_id=fid, image=f"{fid}.png", bbox=(0, 0, 10, 10), y_side="left")


def fig(gid, split="public"):
    return FigureInfo(figure_id=gid, figure_reference=gid, split=split, curve_ids=(f"{gid}-0",))


def sc(f, g, hit, null=0.05):
    return Scored(PairScore(f, g, hit, null, 1.0))


def decisions(records):
    return {r["figure_id"]: r["decision"] for r in records}


def test_figure_with_no_calibrated_frame_is_unassigned_with_reason():
    out = decide_paper("p", [], [fig("g")], {}, rule="rule@x")
    assert out[0]["decision"] == "unassigned"
    assert out[0]["reason"] == "no_calibrated_frame"


def test_figure_with_only_ineligible_scores_is_unassigned_no_eligible_frame():
    out = decide_paper("p", [frame("f")], [fig("g")], {("f", "g"): sc("f", "g", 0.1)}, rule="r")
    assert out[0]["reason"] == "no_eligible_frame"


def test_figure_with_no_projection_at_all_is_no_eligible_frame():
    out = decide_paper("p", [frame("f")], [fig("g")], {}, rule="r")
    assert out[0]["reason"] == "no_eligible_frame"


def test_figure_that_loses_its_frame_is_lost_assignment():
    scored = {("f", "g1"): sc("f", "g1", 0.95), ("f", "g2"): sc("f", "g2", 0.55)}
    out = decide_paper("p", [frame("f")], [fig("g1"), fig("g2")], scored, rule="r")
    assert decisions(out) == {"g1": "proposed_high", "g2": "unassigned"}
    assert [r for r in out if r["figure_id"] == "g2"][0]["reason"] == "lost_assignment"


def test_assigned_record_carries_audit_fields():
    scored = {("f", "g"): sc("f", "g", 0.9)}
    out = decide_paper("p", [frame("f")], [fig("g")], scored, rule="rule@abc")
    r = out[0]
    assert r["candidate_id"] == "p-g"
    assert r["decided_by"] == "rule@abc"
    assert r["image"] == "f.png" and r["S"] == 0.9 and r["M"] == 1.0
    assert r["gt_curve_ids"] == ["g-0"]
    assert r["competitors"] == []


def test_competitors_list_other_eligible_frames_for_the_figure():
    scored = {("f1", "g"): sc("f1", "g", 0.9), ("f2", "g"): sc("f2", "g", 0.85)}
    out = decide_paper("p", [frame("f1"), frame("f2")], [fig("g")], scored, rule="r")
    assert out[0]["decision"] == "proposed_review"
    assert out[0]["competitors"] == [{"frame": "f2", "S": 0.85}]


def test_paper_rank_orders_assigned_figures_by_score():
    scored = {("f1", "g1"): sc("f1", "g1", 0.7), ("f2", "g2"): sc("f2", "g2", 0.95)}
    out = decide_paper("p", [frame("f1"), frame("f2")], [fig("g1"), fig("g2")], scored, rule="r")
    ranks = {r["figure_id"]: r["paper_rank"] for r in out}
    assert ranks == {"g2": 1, "g1": 2}


def pframe(fid, image):
    return FrameInfo(frame_id=fid, image=image, bbox=(0, 0, 10, 10), y_side="left")


EMB = "p/p03_embedded_4.png"
REN = "p/p03_page_render_2.png"


class TestPreferEmbedded:
    def test_page_render_duplicate_of_embedded_frame_is_dropped(self):
        frames = [pframe("e#0", EMB), pframe("r#0", REN)]
        scored = {("e#0", "g"): sc("e#0", "g", 0.9), ("r#0", "g"): sc("r#0", "g", 0.9)}
        assert [f.frame_id for f in prefer_embedded_frames(frames, scored)] == ["e#0"]

    def test_page_render_on_another_page_is_kept(self):
        frames = [pframe("e#0", EMB), pframe("r#0", "p/p04_page_render_3.png")]
        scored = {("e#0", "g"): sc("e#0", "g", 0.9), ("r#0", "g"): sc("r#0", "g", 0.9)}
        assert len(prefer_embedded_frames(frames, scored)) == 2

    def test_page_render_showing_a_different_figure_is_kept(self):
        frames = [pframe("e#0", EMB), pframe("r#0", REN)]
        scored = {("e#0", "g1"): sc("e#0", "g1", 0.9), ("r#0", "g2"): sc("r#0", "g2", 0.9)}
        assert len(prefer_embedded_frames(frames, scored)) == 2

    def test_ineligible_overlap_does_not_count_as_duplicate(self):
        frames = [pframe("e#0", EMB), pframe("r#0", REN)]
        scored = {("e#0", "g"): sc("e#0", "g", 0.9), ("r#0", "g"): sc("r#0", "g", 0.1)}
        assert len(prefer_embedded_frames(frames, scored)) == 2

    def test_decide_paper_does_not_dilute_margin_with_the_duplicate(self):
        frames = [pframe("e#0", EMB), pframe("r#0", REN)]
        scored = {("e#0", "g"): sc("e#0", "g", 0.9), ("r#0", "g"): sc("r#0", "g", 0.9)}
        out = decide_paper("p", frames, [fig("g")], scored, rule="r")
        assert out[0]["decision"] == "proposed_high"
        assert out[0]["image"].endswith("embedded_4.png")
