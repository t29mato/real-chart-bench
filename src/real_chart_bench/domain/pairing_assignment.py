"""Per-paper assignment of plot frames to Starrydata figures
(docs/design/pairing-automation.md §4, §6 and §12).

Pairing is not 2,555 independent yes/no decisions but one assignment problem
per paper: P plot frames found in the paper's images, F Starrydata figures.
One figure is drawn in at most one frame and one frame shows at most one
figure, so the pairs are chosen jointly (Hungarian method with dummy columns
that let a frame stay unassigned) rather than best-first.

The score of a (frame, figure) pair is design §5's C7: the share of the
figure's ground-truth points that, projected through the frame's automatic
axis calibration, land on drawn ink (`hit`), against the same points shifted
off their place (`null`). `hit` is design's S; `contrast = hit - null` guards
against ink-everywhere frames.

THRESHOLDS ARE PROVISIONAL. design §5 makes C7 the lowest-confidence check
and requires an experiment (positives vs. every dimension-compatible wrong
pairing) before any value is trusted; until it is run, no pair is adopted by
rule -- both lanes go to a human (§12.4). The numbers are the ones the
starrydata training-data gate already uses (min_hit 0.7, min_contrast 0.35)
and the design's own adoption point (S >= 0.80, M >= 0.30).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.optimize import linear_sum_assignment

# A pair below these is not a candidate at all: it is recorded as unassigned
# and not shown for review (design §6: a lost figure is cheap, a wrong
# adoption is not).
MIN_HIT = 0.50
MIN_CONTRAST = 0.20

# The strong lane (still reviewed by a human during burn-in, design §7).
HIGH_MIN_HIT = 0.80  # design §6 S
HIGH_MARGIN = 0.30  # design §6 M
HIGH_MIN_CONTRAST = 0.35  # starrydata_build.py's gate

_BIG = 1e6


class Lane(Enum):
    HIGH = "high"
    REVIEW = "review"


@dataclass(frozen=True)
class PairScore:
    frame_id: str
    figure_id: str
    hit: float
    null: float
    inside: float  # share of GT points that project inside the frame (diagnostic)

    @property
    def contrast(self) -> float:
        return self.hit - self.null

    @property
    def eligible(self) -> bool:
        # tolerance: hit and null are floats derived from counts
        return self.hit >= MIN_HIT - 1e-9 and self.contrast >= MIN_CONTRAST - 1e-9


@dataclass(frozen=True)
class Assignment:
    frame_id: str
    figure_id: str
    score: PairScore
    margin: float  # S(best) - S(strongest eligible rival); 1.0 without a rival
    lane: Lane


def _lane(score: PairScore, margin: float) -> Lane:
    eps = 1e-9
    if (
        score.hit >= HIGH_MIN_HIT - eps
        and margin >= HIGH_MARGIN - eps
        and score.contrast >= HIGH_MIN_CONTRAST - eps
    ):
        return Lane.HIGH
    return Lane.REVIEW


def assign_frames_to_figures(
    frame_ids: Sequence[str], figure_ids: Sequence[str], scores: Sequence[PairScore]
) -> list[Assignment]:
    """Jointly best one-to-one pairs among the eligible (frame, figure) scores.

    Frames can stay unassigned (dummy columns cost a hair more than the worst
    eligible pair, so any non-conflicting eligible pair is taken). Scores for
    ids not listed are ignored."""
    if not frame_ids or not figure_ids:
        return []
    f_idx = {f: i for i, f in enumerate(frame_ids)}
    g_idx = {g: j for j, g in enumerate(figure_ids)}
    eligible = {
        (s.frame_id, s.figure_id): s
        for s in scores
        if s.eligible and s.frame_id in f_idx and s.figure_id in g_idx
    }
    if not eligible:
        return []
    n_f, n_g = len(frame_ids), len(figure_ids)
    dummy_cost = (1.0 - MIN_HIT) + 1e-6
    cost = np.full((n_f, n_g + n_f), _BIG)
    for (f, g), s in eligible.items():
        cost[f_idx[f], g_idx[g]] = 1.0 - s.hit
    for i in range(n_f):
        cost[i, n_g + i] = dummy_cost
    rows, cols = linear_sum_assignment(cost)
    out: list[Assignment] = []
    for r, c in zip(rows, cols, strict=True):
        if c >= n_g:
            continue
        s = eligible[(frame_ids[r], figure_ids[c])]
        rivals = [
            o.hit
            for (f, g), o in eligible.items()
            if (f, g) != (s.frame_id, s.figure_id)
            and (f == s.frame_id or g == s.figure_id)
        ]
        margin = 1.0 if not rivals else s.hit - max(rivals)
        out.append(Assignment(s.frame_id, s.figure_id, s, margin, _lane(s, margin)))
    return out
