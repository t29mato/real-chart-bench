"""Builds sorted leaderboard rows from results/*.json payloads (design
§7.15, 司令塔加速指示: リーダーボードv0). Pure transform — no file I/O,
no HTML — kept separate from the static-site rendering (infrastructure
concern) so the ranking logic is unit-testable on its own.

A results entry is either a completed run (has "mean_summary_score") or a
"pending_external_run" placeholder (design §7.19: LineFormer via a Colab
notebook the owner runs manually — registered on the leaderboard before
any score exists, so the model isn't silently missing). Pending rows are
never scored and always rank after every scored row.

Ranking is scoped to dataset_version (2026-09-02, HQ instruction: "never
rank across figure sets"). Two runs are only a fair comparison if they were
scored on the *same* figures — a naive baseline scored on every currently
VERIFIED figure and a specialist model stuck at an older, harder subset
(design §7.16: some models can only be re-run externally) are not
comparable just because both produced a "mean_summary_score". Showing them
in one globally-sorted table with a single "Rank 1" on the higher raw
number reads as "the baseline beats the specialist," which is false: on the
figures they share, the specialist wins. So results are grouped by
dataset_version and ranked *within* each group — each group's own best
scorer is rank 1, regardless of how its raw score compares to a different
group's scores.

Groups are themselves ordered deterministically, most-representative
first:

  1. by figure count (n_figures) descending — the largest evaluated set is
     the most representative snapshot of current benchmark coverage, so it
     leads the page. (A "representative" set could instead be argued as
     "the one most models were run on," i.e. most rows in the group — but
     figure count was chosen because it is intrinsic to the dataset_version
     itself and doesn't shift as unrelated models are added or removed from
     a group; it also matches the existing version-banner logic, which
     already privileges "the current, largest verified-pairs registry
     snapshot" as the headline dataset.)
  2. tie-break: dataset_version string, ascending — arbitrary but stable
     and reproducible across runs.

A scored result missing "dataset_version" (defensive: a malformed or
legacy results file) is *not* dropped or allowed to crash the build — it
gets grouped under the key None, ranked internally like any other group,
and that group is always ordered after every labeled group (an unlabeled,
untraceable figure set must never visually outrank a labeled one) but
still before any pending row.

Within a group, rows are ranked by the point-level F1 (design §7.67:
macro point_f1 at the payload's primary tau), the benchmark's primary
metric since 2026-10-03; mean_summary_score (the curve distance) is carried
as a reference column. A row whose file has no point_metrics (scored before
§7.67, or derived from such a file) ranks after every point-scored row of
its group, by mean_summary_score -- it cannot be placed on the point scale.
"""

from __future__ import annotations

from dataclasses import dataclass

_PENDING_STATUS = "pending_external_run"
_SCORED_STATUS = "scored"


@dataclass(frozen=True)
class LeaderboardRow:
    rank: int | None
    model_id: str
    model_name: str
    status: str
    mean_summary_score: float | None
    n_figures: int | None
    dataset_version: str | None
    run_at: str | None
    note: str | None = None
    # design §7.67: macro values at the primary tau; None without point_metrics
    point_f1: float | None = None
    point_recall: float | None = None
    point_precision: float | None = None
    point_loc_error: float | None = None
    point_tau: float | None = None


def _is_pending(result: dict) -> bool:
    return result.get("status") == _PENDING_STATUS


def _primary_point_cell(result: dict) -> dict | None:
    block = result.get("point_metrics")
    if not block:
        return None
    return block["by_tau"][f"{block['primary_tau']:g}"]["macro"]


def _within_group_sort_key(result: dict) -> tuple:
    cell = _primary_point_cell(result)
    if cell is None:
        return (1, 0.0, -result["mean_summary_score"], result["model_id"])
    return (0, -cell["point_f1"], -result["mean_summary_score"], result["model_id"])


def _group_sort_key(group_key: str | None, groups: dict[str | None, list[dict]]):
    group_results = groups[group_key]
    n_figures = max((r.get("n_figures") or 0) for r in group_results)
    is_unlabeled = group_key is None
    return (is_unlabeled, -n_figures, group_key or "")


def _scored_row(result: dict, rank: int) -> LeaderboardRow:
    cell = _primary_point_cell(result) or {}
    return LeaderboardRow(
        rank=rank,
        model_id=result["model_id"],
        model_name=result["model_name"],
        status=_SCORED_STATUS,
        mean_summary_score=result["mean_summary_score"],
        n_figures=result["n_figures"],
        dataset_version=result.get("dataset_version"),
        run_at=result["run_at"],
        point_f1=cell.get("point_f1"),
        point_recall=cell.get("point_recall"),
        point_precision=cell.get("point_precision"),
        point_loc_error=cell.get("point_loc_error"),
        point_tau=result["point_metrics"]["primary_tau"] if cell else None,
    )


def _pending_row(result: dict) -> LeaderboardRow:
    return LeaderboardRow(
        rank=None,
        model_id=result["model_id"],
        model_name=result["model_name"],
        status=_PENDING_STATUS,
        mean_summary_score=None,
        n_figures=None,
        dataset_version=None,
        run_at=None,
        note=result.get("note"),
    )


def build_leaderboard_rows(results: list[dict]) -> list[LeaderboardRow]:
    scored = [r for r in results if not _is_pending(r)]
    pending = [r for r in results if _is_pending(r)]

    groups: dict[str | None, list[dict]] = {}
    for r in scored:
        groups.setdefault(r.get("dataset_version"), []).append(r)

    ordered_group_keys = sorted(groups, key=lambda k: _group_sort_key(k, groups))

    rows: list[LeaderboardRow] = []
    for group_key in ordered_group_keys:
        group_results = sorted(groups[group_key], key=_within_group_sort_key)
        rows.extend(_scored_row(r, rank=i) for i, r in enumerate(group_results, start=1))

    for r in sorted(pending, key=lambda r: r["model_id"]):
        rows.append(_pending_row(r))

    return rows
