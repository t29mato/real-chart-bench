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
</style>
</head>
<body>
<h1>real-chart-bench leaderboard (v0)</h1>
<p><strong>⚠️ pre-alpha:</strong> evaluation set is a small manually-verified pilot
(10 real figures, gated on data/verified_pairs/registry.json + 3 synthetic
fixtures), not the full v0 dataset. See docs/experiments/ and
docs/design/benchmark-architecture.md &sect;7.19/&sect;7.21 for methodology and
known limitations (automatic image&harr;figure pairing is unsolved outside the
verified registry; naive baselines cannot see black/gray line series or
log-y-axis charts).</p>
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

    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "index.html").write_text(_TEMPLATE.format(rows=rows_html))
    print(f"wrote {SITE_DIR / 'index.html'} ({len(rows)} model(s))")


if __name__ == "__main__":
    main()
