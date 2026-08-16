"""Builds sorted leaderboard rows from results/*.json payloads (design
§7.15, 司令塔加速指示: リーダーボードv0). Pure transform — no file I/O,
no HTML — kept separate from the static-site rendering (infrastructure
concern) so the ranking logic is unit-testable on its own.

A results entry is either a completed run (has "mean_summary_score") or a
"pending_external_run" placeholder (design §7.19: LineFormer via a Colab
notebook the owner runs manually — registered on the leaderboard before
any score exists, so the model isn't silently missing). Pending rows are
never scored and always rank after every scored row.
"""

from __future__ import annotations

from dataclasses import dataclass

_PENDING_STATUS = "pending_external_run"
_SCORED_STATUS = "scored"


@dataclass(frozen=True)
class LeaderboardRow:
    rank: int
    model_id: str
    model_name: str
    status: str
    mean_summary_score: float | None
    n_figures: int | None
    dataset_version: str | None
    run_at: str | None
    note: str | None = None


def _is_pending(result: dict) -> bool:
    return result.get("status") == _PENDING_STATUS


def _sort_key(result: dict) -> tuple[int, float, str]:
    if _is_pending(result):
        return (1, 0.0, result["model_id"])
    return (0, -result["mean_summary_score"], result["model_id"])


def build_leaderboard_rows(results: list[dict]) -> list[LeaderboardRow]:
    ordered = sorted(results, key=_sort_key)
    rows = []
    for i, r in enumerate(ordered, start=1):
        if _is_pending(r):
            rows.append(
                LeaderboardRow(
                    rank=i,
                    model_id=r["model_id"],
                    model_name=r["model_name"],
                    status=_PENDING_STATUS,
                    mean_summary_score=None,
                    n_figures=None,
                    dataset_version=None,
                    run_at=None,
                    note=r.get("note"),
                )
            )
        else:
            rows.append(
                LeaderboardRow(
                    rank=i,
                    model_id=r["model_id"],
                    model_name=r["model_name"],
                    status=_SCORED_STATUS,
                    mean_summary_score=r["mean_summary_score"],
                    n_figures=r["n_figures"],
                    dataset_version=r["dataset_version"],
                    run_at=r["run_at"],
                )
            )
    return rows
