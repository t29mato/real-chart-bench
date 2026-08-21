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

from real_chart_bench.usecase.build_leaderboard import build_leaderboard_rows  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
SITE_DIR = REPO_ROOT / "site"

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
<th>Dataset</th><th>Run at (UTC)</th>
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
    "<td>{dataset_version}</td><td>{run_at}</td></tr>"
)

_PENDING_ROW_TEMPLATE = (
    '<tr class="pending"><td>{rank}</td><td>{model_name}</td>'
    '<td class="score">—</td><td colspan="3">pending external run — {note}</td></tr>'
)


def main() -> None:
    results = [json.loads(p.read_text()) for p in sorted(RESULTS_DIR.glob("*.json"))]
    rows = build_leaderboard_rows(results)

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
        )
        for r in rows
    )
    if not rows:
        rows_html = '<tr><td colspan="6">No results yet.</td></tr>'

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
