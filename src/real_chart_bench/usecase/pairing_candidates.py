"""One paper's pairing candidate records (docs/design/pairing-automation.md
§8, §12): every Starrydata figure gets exactly one record, assigned or not,
so the audit trail shows what the rule did and why.

Nothing here is adopted: `proposed_high` and `proposed_review` both go to a
human (design §12.2).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
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


def decide_paper(
    paper_id: str,
    frames: Sequence[FrameInfo],
    figures: Sequence[FigureInfo],
    scored: Mapping[tuple[str, str], ScoredPair],
    *,
    rule: str,
) -> list[dict]:
    pairs = [s.pair for s in scored.values()]
    assignments = assign_frames_to_figures(
        [f.frame_id for f in frames], [g.figure_id for g in figures], pairs
    )
    by_figure = {a.figure_id: a for a in assignments}
    frame_by_id = {f.frame_id: f for f in frames}
    ranked = sorted(assignments, key=lambda a: -a.score.hit)
    rank = {a.figure_id: i + 1 for i, a in enumerate(ranked)}

    records: list[dict] = []
    for fig in figures:
        base = {
            "candidate_id": f"{paper_id}-{fig.figure_id}",
            "paper_id": paper_id,
            "figure_id": fig.figure_id,
            "figure_reference": fig.figure_reference,
            "split": fig.split,
            "gt_curve_ids": list(fig.curve_ids),
            "decided_by": rule,
        }
        a = by_figure.get(fig.figure_id)
        if a is None:
            eligible = [s for (f, g), s in scored.items() if g == fig.figure_id and s.pair.eligible]
            if not frames:
                reason = "no_calibrated_frame"
            elif not eligible:
                reason = "no_eligible_frame"
            else:
                reason = "lost_assignment"
            records.append({**base, "decision": "unassigned", "reason": reason})
            continue
        s = scored[(a.frame_id, a.figure_id)]
        fr = frame_by_id[a.frame_id]
        competitors = sorted(
            (
                {"frame": f, "S": _r(o.pair.hit)}
                for (f, g), o in scored.items()
                if g == fig.figure_id and f != a.frame_id and o.pair.eligible
            ),
            key=lambda c: -c["S"],
        )
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
            "paper_rank": rank[fig.figure_id],
        })
    return records
