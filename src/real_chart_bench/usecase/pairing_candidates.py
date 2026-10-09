"""One paper's pairing candidate records (docs/design/pairing-automation.md
§8, §12): every Starrydata figure gets exactly one record, assigned or not,
so the audit trail shows what the rule did and why.

Nothing here is adopted: `proposed_high` and `proposed_review` both go to a
human (design §12.2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from real_chart_bench.domain.pairing_assignment import (
    Lane,
    PairScore,
    assign_frames_to_figures,
)


class ScoredPair(Protocol):
    pair: PairScore
    x_transform: str
    y_transform: str


@dataclass(frozen=True)
class FrameInfo:
    frame_id: str
    image: str  # path relative to data/raw/images
    bbox: tuple[float, float, float, float]
    y_side: str


@dataclass(frozen=True)
class FigureInfo:
    figure_id: str
    figure_reference: str
    split: str
    curve_ids: tuple[str, ...]


def _r(v: float) -> float:
    return round(float(v), 4)


def _page_of(image: str) -> str:
    return image.rsplit("/", 1)[-1].split("_", 1)[0]


def prefer_embedded_frames(
    frames: Sequence[FrameInfo], scored: Mapping[tuple[str, str], ScoredPair]
) -> list[FrameInfo]:
    """Drop page-render frames that show the same figure as an embedded
    image's frame on the same page (design pairing-automation.md §12.5)."""
    eligible: dict[str, set[str]] = {}
    for (f, g), s in scored.items():
        if s.pair.eligible:
            eligible.setdefault(f, set()).add(g)
    embedded = [f for f in frames if "_embedded_" in f.image]
    kept = []
    for fr in frames:
        if "_page_render_" in fr.image and any(
            _page_of(e.image) == _page_of(fr.image)
            and eligible.get(e.frame_id, set()) & eligible.get(fr.frame_id, set())
            for e in embedded
        ):
            continue
        kept.append(fr)
    return kept


def _idkey(figure_id: str) -> tuple[int, str]:
    """Numeric order for numeric ids (shorter first), plain order otherwise."""
    return (len(figure_id), figure_id)


def _group_scores(
    scored: Mapping[tuple[str, str], ScoredPair], groups: Sequence[Sequence[str]]
) -> tuple[dict[tuple[str, str], ScoredPair], dict[str, str], dict[tuple[str, str], str]]:
    """Collapse each sibling group to one figure for the assignment (design
    12.9). Returns the scores keyed (frame, representative), member -> its
    representative (the lowest id) and (frame, representative) -> the member
    that projects best on that frame: highest S, then the member whose
    projection matches the printed axis scale, then the lower id."""
    rep_of: dict[str, str] = {}
    for g in groups:
        rep = min(g, key=_idkey)
        for m in g:
            rep_of[m] = rep
    chosen: dict[tuple[str, str], str] = {}
    out: dict[tuple[str, str], ScoredPair] = {}
    for (f, g), s in scored.items():
        rep = rep_of.get(g, g)
        key = (f, rep)

        def rank(m: str, sp: ScoredPair) -> tuple:
            return (-sp.pair.hit, not getattr(sp, "matches_axis_scale", False), _idkey(m))

        if key not in chosen or rank(g, s) < rank(chosen[key], scored[(f, chosen[key])]):
            chosen[key] = g
    for (f, rep), m in chosen.items():
        s = scored[(f, m)]
        out[(f, rep)] = _Regrouped(replace(s.pair, figure_id=rep), s)
    return out, rep_of, chosen


class _Regrouped:
    """A member's score presented under its group's id."""

    def __init__(self, pair: PairScore, inner: ScoredPair) -> None:
        self.pair = pair
        self.inner = inner
        self.x_transform = inner.x_transform
        self.y_transform = inner.y_transform


def decide_paper(
    paper_id: str,
    frames: Sequence[FrameInfo],
    figures: Sequence[FigureInfo],
    scored: Mapping[tuple[str, str], ScoredPair],
    *,
    rule: str,
    sibling_groups: Sequence[Sequence[str]] = (),
) -> list[dict]:
    """sibling_groups (design 12.9): ids of figures that are one plot digitized
    twice. A group takes one frame; the member that projects best is proposed
    and the others get decision `sibling_of` (no pairing of their own)."""
    frames = prefer_embedded_frames(frames, scored)
    kept_ids = {f.frame_id for f in frames}
    scored = {k: v for k, v in scored.items() if k[0] in kept_ids}
    raw_scored = scored
    group_of: dict[str, tuple[str, ...]] = {}
    chosen: dict[tuple[str, str], str] = {}
    if sibling_groups:
        scored, rep_of, chosen = _group_scores(scored, sibling_groups)
        for g in sibling_groups:
            for m in g:
                group_of[m] = tuple(g)
        figure_ids = sorted({rep_of.get(g.figure_id, g.figure_id) for g in figures},
                            key=lambda x: [g.figure_id for g in figures].index(x))
    else:
        rep_of = {}
        figure_ids = [g.figure_id for g in figures]
    pairs = [s.pair for s in scored.values()]
    assignments = assign_frames_to_figures([f.frame_id for f in frames], figure_ids, pairs)
    by_figure = {a.figure_id: a for a in assignments}
    frame_by_id = {f.frame_id: f for f in frames}
    ranked = sorted(assignments, key=lambda a: -a.score.hit)
    rank = {a.figure_id: i + 1 for i, a in enumerate(ranked)}
    cand = {f.figure_id: f"{paper_id}-{f.figure_id}" for f in figures}

    records: list[dict] = []
    for fig in figures:
        base = {
            "candidate_id": cand[fig.figure_id],
            "paper_id": paper_id,
            "figure_id": fig.figure_id,
            "figure_reference": fig.figure_reference,
            "split": fig.split,
            "gt_curve_ids": list(fig.curve_ids),
            "decided_by": rule,
        }
        rep = rep_of.get(fig.figure_id, fig.figure_id)
        a = by_figure.get(rep)
        primary = chosen.get((a.frame_id, rep), fig.figure_id) if a is not None else None
        if a is not None and primary != fig.figure_id:
            fr = frame_by_id[a.frame_id]
            records.append({
                **base,
                "decision": "sibling_of",
                "sibling_of": cand[primary],
                "image": fr.image,
                "frame_id": fr.frame_id,
            })
            continue
        if a is None:
            eligible = [s for (f, g), s in raw_scored.items()
                        if g == fig.figure_id and s.pair.eligible]
            if not frames:
                reason = "no_calibrated_frame"
            elif not eligible:
                reason = "no_eligible_frame"
            else:
                reason = "lost_assignment"
            records.append({**base, "decision": "unassigned", "reason": reason})
            continue
        s = scored[(a.frame_id, rep)]
        fr = frame_by_id[a.frame_id]
        competitors = sorted(
            (
                {"frame": f, "S": _r(o.pair.hit)}
                for (f, g), o in scored.items()
                if g == rep and f != a.frame_id and o.pair.eligible
            ),
            key=lambda c: -c["S"],
        )
        extra = {}
        others = [m for m in group_of.get(fig.figure_id, ()) if m != fig.figure_id]
        if others:
            extra["siblings"] = [cand[m] for m in others]
        records.append({
            **base,
            "decision": "proposed_high" if a.lane is Lane.HIGH else "proposed_review",
            "image": fr.image,
            "frame_id": fr.frame_id,
            "frame_bbox": [round(v, 1) for v in fr.bbox],
            "y_axis_side": fr.y_side,
            "x_transform": s.x_transform,
            "y_transform": s.y_transform,
            "S": _r(a.score.hit),
            "null": _r(a.score.null),
            "contrast": _r(a.score.contrast),
            "inside": _r(a.score.inside),
            "M": _r(a.margin),
            "competitors": competitors,
            "paper_rank": rank[rep],
            **extra,
        })
    return records


def merge_candidate_records(
    existing: Sequence[Mapping],
    new: Sequence[Mapping],
    new_source: str,
    *,
    old_source: str,
) -> list[dict]:
    """Add a new image source's records to the committed candidates.

    Existing records are kept as they are (only `image_source` is filled in
    when missing); a paper that already has records is never reprocessed, so a
    paper appearing in both is an error.
    """
    seen = {r["paper_id"] for r in existing}
    clash = sorted({r["paper_id"] for r in new} & seen, key=int)
    if clash:
        raise ValueError(f"paper {clash[0]} already has candidate records")
    merged = [{**r, "image_source": r.get("image_source", old_source)} for r in existing]
    merged += [{**r, "image_source": new_source} for r in new]
    merged.sort(key=lambda r: (int(r["paper_id"]), int(r["figure_id"])))
    return merged
