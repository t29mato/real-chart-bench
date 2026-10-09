"""Score one Starrydata figure against one automatically calibrated plot
frame (docs/design/pairing-automation.md §5 C7, §12).

The figure's hand-digitized points are projected through the frame's tick
calibration (domain/tick_calibration.best_projection chooses, from the fixed
list of forms a stored quantity takes when printed -- scale, K->degC, 1000/T,
printed log10 -- the one under which the points sit in the frame and on the
ink) and the share landing on drawn ink is the pair's score. No model reads
the image; nothing here is an LLM output.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from real_chart_bench.adapter.auto_axis_calibration import ImageCalibration
from real_chart_bench.adapter.starrydata_figure_gt import FigureGt
from real_chart_bench.domain.pairing_assignment import PairScore
from real_chart_bench.domain.tick_calibration import best_projection, candidate_transforms

INK_GRAY = 200  # marker ink: anything clearly darker than paper ...
INK_CHROMA = 60  # ... or clearly coloured


@dataclass(frozen=True)
class ScoredProjection:
    pair: PairScore
    x_transform: str
    y_transform: str
    points_px: list  # per curve: [(x, y), ...] in image pixels
    # whether the figure's quantity is drawn as stored (design pairing-automation.md 12.9):
    # no log10 conversion, and a log(...) quantity only on a linear axis. Breaks the tie
    # between sibling digitizations (sigma / log sigma) of the same plot.
    matches_axis_scale: bool = False


def ink_mask(rgb: np.ndarray) -> np.ndarray:
    gray = rgb.astype(np.float32).mean(2)
    chroma = rgb.max(2).astype(int) - rgb.min(2).astype(int)
    return (gray < INK_GRAY) | (chroma > INK_CHROMA)


def score_figure_on_frame(
    frame_id: str, figure: FigureGt, cal: ImageCalibration, ink: np.ndarray
) -> ScoredProjection | None:
    """None when no printed form of the figure's values keeps its points in
    the frame (design C1/C3/C4 in their practical form: the quantity cannot
    be this plot)."""
    if cal.x_fit is None or cal.y_fit is None or not figure.curves:
        return None
    projection = best_projection(
        [(c.xs, c.ys) for c in figure.curves],
        cal.x_fit,
        cal.y_fit,
        cal.frame,
        ink,
        candidate_transforms(figure.prop_x, figure.unit_x),
        candidate_transforms(figure.prop_y, figure.unit_y),
    )
    if projection is None:
        return None
    pair = PairScore(
        frame_id=frame_id,
        figure_id=figure.figure_id,
        hit=projection.hit,
        null=projection.null,
        inside=projection.inside,
    )
    log_quantity = figure.prop_y.strip().lower().startswith("log")
    matches = projection.y_transform.kind != "log10" and not (
        log_quantity and cal.y_fit.scale == "log"
    )
    return ScoredProjection(
        pair, projection.x_transform.name, projection.y_transform.name, projection.points_px,
        matches,
    )
