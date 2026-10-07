"""Shape of a single parsed Starrydata curve row (design §7.9).

Defined in the use case layer (not the adapter) so that both
``adapter/starrydata_csv.py`` (producer) and
``usecase/build_ground_truth_manifest.py`` (consumer) depend on it without
either depending on the other — keeps the clean-architecture dependency
direction adapter -> usecase intact.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParsedCurveRow:
    sid: str
    doi: str
    figure_id: str
    figure_name: str
    series_label: str
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    # The only fields that distinguish curves within one figure -- prop_y is the
    # same string for every curve, so series_label is too. composition is the
    # database's canonical formula, NOT the legend text printed in the figure;
    # see docs/experiments/2026-10-07-series-labels-available.md.
    composition: str = ""
    sample_id: str = ""
