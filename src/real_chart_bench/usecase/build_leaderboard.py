"""Builds sorted leaderboard rows from results/*.json payloads (design
§7.15, 司令塔加速指示: リーダーボードv0). Pure transform — no file I/O,
no HTML — kept separate from the static-site rendering (infrastructure
concern) so the ranking logic is unit-testable on its own.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LeaderboardRow:
    rank: int
    model_id: str
    model_name: str
    mean_summary_score: float
    n_figures: int
    dataset_version: str
    run_at: str


def build_leaderboard_rows(results: list[dict]) -> list[LeaderboardRow]:
    ordered = sorted(results, key=lambda r: (-r["mean_summary_score"], r["model_id"]))
    return [
        LeaderboardRow(
            rank=i,
            model_id=r["model_id"],
            model_name=r["model_name"],
            mean_summary_score=r["mean_summary_score"],
            n_figures=r["n_figures"],
            dataset_version=r["dataset_version"],
            run_at=r["run_at"],
        )
        for i, r in enumerate(ordered, start=1)
    ]
