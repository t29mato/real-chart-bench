"""Renders site/index.html from results/*.json (design §7.15, 司令塔加速指示:
リーダーボードv0 = GitHub Pages相当の静的HTML)。

Usage:
    python scripts/leaderboard/generate.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.usecase.build_leaderboard import build_leaderboard_rows  # noqa: E402
from real_chart_bench.usecase.build_leaderboard_breakdown import (  # noqa: E402
    build_model_breakdown,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
SITE_DIR = REPO_ROOT / "site"
REGISTRY_PATH = REPO_ROOT / "data/verified_pairs/registry.json"

# design §7.38 (HQ instruction 2026-08-27, "図タイプ別の内訳"): display
# labels for usecase/build_leaderboard_breakdown.py's category strings.
_CATEGORY_LABELS = {
    "real-linear-x": "Real figures (linear x-axis)",
    "real-log-x": "Real figures (log x-axis)",
    "synthetic": "Synthetic fixtures",
    "real-unknown": "Real figures (unclassified)",
}

_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>real-chart-bench leaderboard (v0, pre-alpha)</title>
<style>
  body {{
    font-family: -apple-system, sans-serif; max-width: 900px;
    margin: 2rem auto; padding: 0 1rem;
  }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #ccc; padding: 0.5rem 0.75rem; text-align: left; }}
  th {{ background: #f0f0f0; }}
  .score {{ font-variant-numeric: tabular-nums; }}
  tr.pending {{ color: #888; font-style: italic; }}
  .caveat {{
    background: #fff8e1; border: 1px solid #e0c060;
    padding: 0.75rem 1rem; border-radius: 4px;
  }}
  .version-banner {{
    background: #e8f0fe; border: 1px solid #a8c7fa;
    padding: 0.6rem 1rem; border-radius: 4px; margin-bottom: 1rem;
    font-variant-numeric: tabular-nums;
  }}
  td details {{ font-size: 0.85em; }}
  td details table {{ margin-top: 0.4rem; width: auto; min-width: 14rem; }}
  td details th, td details td {{ padding: 0.25rem 0.5rem; }}
  td details summary {{ cursor: pointer; color: #1a5fb4; }}
</style>
</head>
<body>
<h1>real-chart-bench leaderboard (v0)</h1>
<p class="version-banner">📌 <strong>Latest evaluated set: {latest_dataset_version}</strong>
(most recent run: {latest_run_at} UTC). Scores below may come from different runs against
different verified-pair counts as the registry grows over time -- always check each row's
own <strong>Dataset</strong>/<strong>Run at</strong> columns before comparing scores across
models; do not compare a row's score against a different dataset_version as if they were
the same benchmark.</p>
<p><strong>⚠️ pre-alpha:</strong> evaluation set is a small manually-verified pilot
gated on data/verified_pairs/registry.json (real-figure count varies by run, see the
Dataset column) + 3 synthetic fixtures, not the full v0 dataset. See docs/experiments/ and
docs/design/benchmark-architecture.md &sect;7.19/&sect;7.21/&sect;7.27 for methodology and
known limitations (automatic image&harr;figure pairing is unsolved outside the
verified registry; naive baselines cannot see black/gray line series or
achromatic markers).</p>
<table>
<thead><tr>
<th>Rank</th><th>Model</th><th>Mean score</th><th>#figures</th>
<th>Dataset</th><th>Run at (UTC)</th><th>Breakdown</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
</body>
</html>
"""

_ROW_TEMPLATE = (
    "<tr><td>{rank}</td><td>{model_name}</td>"
    '<td class="score">{score:.3f}</td><td>{n_figures}</td>'
    "<td>{dataset_version}</td><td>{run_at}</td><td>{breakdown_html}</td></tr>"
)

_PENDING_ROW_TEMPLATE = (
    '<tr class="pending"><td>{rank}</td><td>{model_name}</td>'
    '<td class="score">—</td><td colspan="4">pending external run — {note}</td></tr>'
)

_BREAKDOWN_TEMPLATE = (
    "<details><summary>by figure type</summary>"
    "<table><thead><tr><th>Type</th><th>Mean</th><th>#</th></tr></thead>"
    "<tbody>{category_rows}</tbody></table></details>"
)

_BREAKDOWN_CATEGORY_ROW_TEMPLATE = (
    "<tr><td>{label}</td>"
    '<td class="score">{score:.3f}</td><td>{n_figures}</td></tr>'
)


def _render_breakdown_html(breakdown: list) -> str:
    # design §7.38: an empty breakdown (e.g. a scored result with no
    # per_figure entries, which shouldn't happen in practice but must not
    # crash rendering) just shows a dash, same convention as a pending row's
    # score column.
    if not breakdown:
        return "—"
    category_rows = "\n".join(
        _BREAKDOWN_CATEGORY_ROW_TEMPLATE.format(
            label=_CATEGORY_LABELS.get(b.category, b.category),
            score=b.mean_summary_score,
            n_figures=b.n_figures,
        )
        for b in breakdown
    )
    return _BREAKDOWN_TEMPLATE.format(category_rows=category_rows)


def main() -> None:
    results = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    rows = build_leaderboard_rows(results)

    # design §7.38 (HQ instruction 2026-08-27): join each scored result
    # against the verified-pairs registry to break its score down by figure
    # type (real-linear-x / real-log-x / synthetic). Keyed by
    # f"{paper_id}-{figure_id}" to match how every DatasetItem's figure_id is
    # built (run_baselines.py, the LineFormer notebook -- see
    # evaluate_dataset.py's FigureResult).
    registry = load_registry(REGISTRY_PATH)
    pairings_by_figure_id = {f"{p.paper_id}-{p.figure_id}": p for p in registry}
    results_by_model_id = {r["model_id"]: r for r in results}

    rows_html = "\n".join(
        _PENDING_ROW_TEMPLATE.format(rank=r.rank, model_name=r.model_name, note=r.note)
        if r.status == "pending_external_run"
        else _ROW_TEMPLATE.format(
            rank=r.rank,
            model_name=r.model_name,
            score=r.mean_summary_score,
            n_figures=r.n_figures,
            dataset_version=r.dataset_version,
            run_at=r.run_at,
            breakdown_html=_render_breakdown_html(
                build_model_breakdown(results_by_model_id[r.model_id], pairings_by_figure_id)
            ),
        )
        for r in rows
    )
    if not rows:
        rows_html = '<tr><td colspan="7">No results yet.</td></tr>'

    # Design §7.27/HQ 2026-08-21: the version banner is derived from the most
    # recently-run scored result, not hardcoded -- a hardcoded string is
    # exactly what went stale across the 1->10->20-pair registry expansions
    # (results/*.json's own dataset_version had the same bug, fixed in
    # scripts/eval/run_baselines.py the same day).
    scored = [r for r in rows if r.status == "scored"]
    if scored:
        latest = max(scored, key=lambda r: r.run_at)
        latest_dataset_version = latest.dataset_version
        latest_run_at = latest.run_at
    else:
        latest_dataset_version = "(no scored runs yet)"
        latest_run_at = "-"

    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "index.html").write_text(
        _TEMPLATE.format(
            rows=rows_html,
            latest_dataset_version=latest_dataset_version,
            latest_run_at=latest_run_at,
        )
    )
    print(f"wrote {SITE_DIR / 'index.html'} ({len(rows)} model(s))")


if __name__ == "__main__":
    main()
