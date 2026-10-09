"""Sibling groups in decide_paper (design pairing-automation.md 12.9)."""

from __future__ import annotations

from dataclasses import dataclass

from real_chart_bench.domain.pairing_assignment import PairScore
from real_chart_bench.usecase.pairing_candidates import FigureInfo, FrameInfo, decide_paper


@dataclass
class Scored:
    pair: PairScore
    x_transform: str = "x1e0"
    y_transform: str = "x1e0"
    matches_axis_scale: bool = False


def frame(fid):
    return FrameInfo(frame_id=fid, image=f"{fid}.png", bbox=(0, 0, 10, 10), y_side="left")


def fig(gid):
    return FigureInfo(figure_id=gid, figure_reference=gid, split="public", curve_ids=(f"{gid}-0",))


def sc(f, g, hit, matches=False):
    return Scored(PairScore(f, g, hit, 0.05, 1.0), matches_axis_scale=matches)


def decisions(records):
    return {r["figure_id"]: r["decision"] for r in records}


def run(frames, figs, scored, groups=None):
    kw = {} if groups is None else {"sibling_groups": groups}
    return decide_paper("p", frames, figs, scored, rule="r", **kw)


def test_group_is_proposed_once_for_the_member_that_projects_and_the_other_is_sibling_of():
    out = run([frame("f")], [fig("g1"), fig("g2")], {("f", "g2"): sc("f", "g2", 0.95)},
              [("g1", "g2")])
    by = {r["figure_id"]: r for r in out}
    assert by["g2"]["decision"] == "proposed_high"
    assert by["g1"]["decision"] == "sibling_of"
    assert by["g1"]["sibling_of"] == "p-g2"
    assert by["g1"]["image"] == "f.png"
    assert by["g2"]["siblings"] == ["p-g1"]


def test_the_better_projecting_member_is_primary():
    scored = {("f", "g1"): sc("f", "g1", 0.80), ("f", "g2"): sc("f", "g2", 0.92)}
    out = run([frame("f")], [fig("g1"), fig("g2")], scored, [("g1", "g2")])
    assert decisions(out) == {"g1": "sibling_of", "g2": "proposed_high"}


def test_a_tie_goes_to_the_member_matching_the_printed_axis_scale_then_the_lower_id():
    tie = {("f", "g1"): sc("f", "g1", 0.9), ("f", "g2"): sc("f", "g2", 0.9)}
    assert decisions(run([frame("f")], [fig("g1"), fig("g2")], tie, [("g1", "g2")]))[
        "g1"] == "proposed_high"
    tie2 = {("f", "g1"): sc("f", "g1", 0.9), ("f", "g2"): sc("f", "g2", 0.9, matches=True)}
    assert decisions(run([frame("f")], [fig("g1"), fig("g2")], tie2, [("g1", "g2")]))[
        "g2"] == "proposed_high"


def test_siblings_do_not_compete_with_each_other_for_the_margin():
    scored = {("f", "g1"): sc("f", "g1", 0.95), ("f", "g2"): sc("f", "g2", 0.94)}
    grouped = run([frame("f")], [fig("g1"), fig("g2")], scored, [("g1", "g2")])
    plain = run([frame("f")], [fig("g1"), fig("g2")], scored)
    assert next(r for r in grouped if r["figure_id"] == "g1")["M"] == 1.0
    assert next(r for r in plain if r["figure_id"] == "g1")["M"] < 0.3


def test_a_group_with_no_eligible_frame_keeps_each_members_own_reason():
    out = run([frame("f")], [fig("g1"), fig("g2")], {}, [("g1", "g2")])
    assert [(r["decision"], r["reason"]) for r in out] == [("unassigned", "no_eligible_frame")] * 2


def test_a_group_takes_one_frame_so_another_figure_can_have_the_other():
    scored = {("f1", "g1"): sc("f1", "g1", 0.95), ("f1", "g2"): sc("f1", "g2", 0.9),
              ("f2", "g3"): sc("f2", "g3", 0.9)}
    out = run([frame("f1"), frame("f2")], [fig("g1"), fig("g2"), fig("g3")], scored,
              [("g1", "g2")])
    assert decisions(out) == {"g1": "proposed_high", "g2": "sibling_of", "g3": "proposed_high"}


def test_no_groups_is_the_old_behaviour():
    scored = {("f", "g1"): sc("f", "g1", 0.95)}
    assert run([frame("f")], [fig("g1"), fig("g2")], scored) == run(
        [frame("f")], [fig("g1"), fig("g2")], scored, [])
