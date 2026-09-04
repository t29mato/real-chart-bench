"""Renders site/index.html from results/*.json (design §7.15, 司令塔加速指示:
リーダーボードv0 = GitHub Pages相当の静的HTML)。

Usage:
    python scripts/leaderboard/generate.py
"""

from __future__ import annotations

import itertools
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
  .head-to-head {{
    background: #f3f3f3; border: 1px solid #ccc;
    padding: 0.75rem 1rem; border-radius: 4px; margin-top: 1rem;
    font-variant-numeric: tabular-nums;
  }}
  .head-to-head table {{ margin-top: 0.5rem; }}
  td details {{ font-size: 0.85em; }}
  td details table {{ margin-top: 0.4rem; width: auto; min-width: 14rem; }}
  td details th, td details td {{ padding: 0.25rem 0.5rem; }}
  td details summary {{ cursor: pointer; color: #1a5fb4; }}
  h2.dataset-section {{
    margin-top: 2rem; padding-top: 0.5rem; border-top: 2px solid #ddd;
    font-size: 1.05rem;
  }}
  h2.dataset-section .figure-count {{ font-weight: normal; color: #555; }}
  section.dataset-section table {{ margin-bottom: 0; }}
</style>
</head>
<body>
<h1>real-chart-bench leaderboard (v0)</h1>
<p class="version-banner">📌 <strong>Latest evaluated set: {latest_dataset_version}</strong>
(most recent run: {latest_run_at} UTC). <strong>Rank is only meaningful within a section
below</strong> -- each section is its own figure set (dataset_version), scored runs are
never ranked against a different section's runs, and a higher raw score in a smaller
section does NOT mean that model beat a model ranked #1 in a different section. Always
check which section a row is in (and its own <strong>Run at</strong> column) before
comparing scores.</p>
<p><strong>⚠️ pre-alpha:</strong> evaluation set is a small manually-verified pilot
gated on data/verified_pairs/registry.json (real-figure count varies by run, see each
section's heading below) + 3 synthetic fixtures, not the full v0 dataset. See
docs/experiments/ and docs/design/benchmark-architecture.md
&sect;7.19/&sect;7.21/&sect;7.27 for methodology and known limitations (automatic
image&harr;figure pairing is unsolved outside the verified registry; naive baselines
cannot see black/gray line series or achromatic markers).</p>
{sections}
{pending_section}
{head_to_head}
</body>
</html>
"""

# 2026-09-02 (HQ: "never rank across figure sets"): one <section> per
# dataset_version, each with its own Rank column starting at 1 -- see
# usecase/build_leaderboard.py's module docstring for why a rank number
# must never be visually comparable across different figure sets. The
# heading states the figure count and the dataset_version string so a
# skimming reader sees "this rank is scoped to N figures" without having to
# open a details/Breakdown panel or cross-reference a Dataset column.
_SECTION_TEMPLATE = """<section class="dataset-section">
<h2 class="dataset-section">{heading}</h2>
<table>
<thead><tr>
<th>Rank</th><th>Model</th><th>Mean score</th><th>#figures</th>
<th>Run at (UTC)</th><th>Breakdown</th>
</tr></thead>
<tbody>
{rows}
</tbody>
</table>
</section>"""

_PENDING_SECTION_TEMPLATE = """<section class="dataset-section">
<h2 class="dataset-section">Pending -- not yet run</h2>
<p>Registered but not yet scored. Never ranked against any section above.</p>
<table>
<thead><tr><th>Model</th><th>Status</th></tr></thead>
<tbody>
{rows}
</tbody>
</table>
</section>"""

# design task 2026-09-02 ("Task 1"): naive-cv-v0 is scored against every
# currently-VERIFIED figure (110 as of this writing) while
# lineformer-pretrained.json is stuck at the 45 it was last run against
# (LineFormer can't be re-run here -- needs mmcv/mmdetection via a Colab
# notebook, design §7.16) -- those two mean_summary_scores are NOT
# comparable on their own. run_baselines.py additionally emits
# naive-cv-v0-lineformer-subset.json: naive-cv rescored on exactly
# LineFormer's own figure set (see scripts/eval/run_baselines.py's
# _lineformer_comparable_subset). This callout surfaces the one number pair
# that IS a fair head-to-head, right next to the table, instead of leaving
# the reader to notice the matching dataset_version themselves.
#
# 2026-09-04: LineFormer's original figure set can drift out from under it
# -- paper 21682/figure 21284 was VERIFIED when LineFormer was scored on it
# but was later REJECTED (rejection_category "image", printed y-axis has no
# recoverable unit). When that happens, naive-cv-v0-lineformer-subset.json
# no longer shares lineformer-pretrained.json's dataset_version (see
# _lineformer_comparable_subset) -- it gets a distinct one naming the
# shrunk figure count instead, since reusing LineFormer's original version
# string would misrepresent comparability to LineFormer's *original*,
# larger-figure-set score. In that case the fair counterpart is
# results/lineformer-pretrained-comparable-subset.json: LineFormer's own
# mean recomputed on that identical shrunk set by pure arithmetic over its
# already-published per_figure scores (not a re-run) --
# _render_head_to_head_html prefers that file over lineformer-pretrained.json
# whenever both exist and only the comparable-subset one actually shares the
# naive-cv subset's dataset_version.
_HEAD_TO_HEAD_TEMPLATE = """<div class="head-to-head">
<strong>Head-to-head, same figures only:</strong> the sections above are
each scored against a different figure count (see each section's own
heading) -- comparing a row's raw score across sections is misleading. The
one comparison below IS apples-to-apples: both rows are scored on the exact
same {n_figures}-figure set (dataset_version <code>{dataset_version}</code>).
<table><thead><tr><th>Model</th><th>Mean score</th><th>#figures</th></tr></thead>
<tbody>
{head_to_head_rows}
</tbody></table>
</div>"""

_HEAD_TO_HEAD_ROW_TEMPLATE = (
    '<tr><td>{model_name}</td><td class="score">{score:.3f}</td><td>{n_figures}</td></tr>'
)


def _render_head_to_head_html(results_by_model_id: dict) -> str:
    subset = results_by_model_id.get("naive-cv-v0-lineformer-subset")
    if subset is None:
        return ""
    # Prefer the recomputed-on-the-same-subset LineFormer figure (only
    # written when the comparable set has drifted from LineFormer's
    # original run, see scripts/eval/run_baselines.py's
    # _lineformer_recomputed_subset); fall back to LineFormer's own
    # published result for the common no-drift case.
    lineformer = results_by_model_id.get("lineformer-pretrained-comparable-subset")
    if lineformer is None or lineformer["dataset_version"] != subset["dataset_version"]:
        lineformer = results_by_model_id.get("lineformer-pretrained")
    if lineformer is None or lineformer.get("status") == "pending_external_run":
        return ""
    if lineformer["dataset_version"] != subset["dataset_version"]:
        # The two rows were meant to be built on the same figure set (see
        # _lineformer_comparable_subset) -- if they've drifted apart
        # (e.g. someone regenerated one but not the other), showing them
        # side by side would repeat exactly the mistake this callout exists
        # to prevent. Say nothing rather than show a stale comparison.
        return ""
    contenders = sorted([lineformer, subset], key=lambda r: -r["mean_summary_score"])
    head_to_head_rows = "\n".join(
        _HEAD_TO_HEAD_ROW_TEMPLATE.format(
            model_name=r["model_name"], score=r["mean_summary_score"], n_figures=r["n_figures"]
        )
        for r in contenders
    )
    return _HEAD_TO_HEAD_TEMPLATE.format(
        n_figures=subset["n_figures"],
        dataset_version=subset["dataset_version"],
        head_to_head_rows=head_to_head_rows,
    )

_ROW_TEMPLATE = (
    "<tr><td>{rank}</td><td>{model_name}</td>"
    '<td class="score">{score:.3f}</td><td>{n_figures}</td>'
    "<td>{run_at}</td><td>{breakdown_html}</td></tr>"
)

_PENDING_ROW_TEMPLATE = (
    '<tr class="pending"><td>{model_name}</td>'
    "<td>pending external run — {note}</td></tr>"
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


def _section_heading(dataset_version: str | None, group_rows: list) -> str:
    label = (
        f"Dataset: <code>{dataset_version}</code>"
        if dataset_version is not None
        else "No dataset_version recorded (legacy/malformed result)"
    )
    n_figures_values = sorted({r.n_figures for r in group_rows if r.n_figures is not None})
    if len(n_figures_values) == 1:
        figures_text = f"{n_figures_values[0]} figures"
    elif n_figures_values:
        # Defensive: rows sharing a dataset_version are expected to share a
        # figure count (same set of figures) -- if they don't, say so
        # loudly instead of picking one number and hiding the mismatch.
        figures_text = f"{', '.join(str(n) for n in n_figures_values)} figures (mismatched!)"
    else:
        figures_text = "unknown figure count"
    return f'{label} <span class="figure-count">&mdash; {figures_text}</span>'


def _render_sections_html(
    scored_rows: list, results_by_model_id: dict, pairings_by_figure_id: dict
) -> str:
    if not scored_rows:
        return "<p>No scored results yet.</p>"
    sections = []
    by_dataset_version = itertools.groupby(scored_rows, key=lambda r: r.dataset_version)
    for dataset_version, group_iter in by_dataset_version:
        group_rows = list(group_iter)
        rows_html = "\n".join(
            _ROW_TEMPLATE.format(
                rank=r.rank,
                model_name=r.model_name,
                score=r.mean_summary_score,
                n_figures=r.n_figures,
                run_at=r.run_at,
                breakdown_html=_render_breakdown_html(
                    build_model_breakdown(results_by_model_id[r.model_id], pairings_by_figure_id)
                ),
            )
            for r in group_rows
        )
        sections.append(
            _SECTION_TEMPLATE.format(
                heading=_section_heading(dataset_version, group_rows),
                rows=rows_html,
            )
        )
    return "\n".join(sections)


def _render_pending_section_html(pending_rows: list) -> str:
    if not pending_rows:
        return ""
    rows_html = "\n".join(
        _PENDING_ROW_TEMPLATE.format(model_name=r.model_name, note=r.note) for r in pending_rows
    )
    return _PENDING_SECTION_TEMPLATE.format(rows=rows_html)


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

    # 2026-09-02 (HQ: "never rank across figure sets"): render one <section>
    # per dataset_version (build_leaderboard_rows already groups+orders
    # scored rows so same-dataset_version rows are contiguous) so a rank
    # number is never visually next to a rank from a different figure set.
    # Pending rows are unranked and get their own trailing section.
    scored_rows = [r for r in rows if r.status == "scored"]
    pending_rows = [r for r in rows if r.status == "pending_external_run"]
    sections_html = _render_sections_html(scored_rows, results_by_model_id, pairings_by_figure_id)
    pending_section_html = _render_pending_section_html(pending_rows)

    # Design §7.27/HQ 2026-08-21: the version banner is derived from the most
    # recently-run scored result, not hardcoded -- a hardcoded string is
    # exactly what went stale across the 1->10->20-pair registry expansions
    # (results/*.json's own dataset_version had the same bug, fixed in
    # scripts/eval/run_baselines.py the same day).
    if scored_rows:
        latest = max(scored_rows, key=lambda r: r.run_at)
        latest_dataset_version = latest.dataset_version
        latest_run_at = latest.run_at
    else:
        latest_dataset_version = "(no scored runs yet)"
        latest_run_at = "-"

    SITE_DIR.mkdir(exist_ok=True)
    (SITE_DIR / "index.html").write_text(
        _TEMPLATE.format(
            sections=sections_html,
            pending_section=pending_section_html,
            latest_dataset_version=latest_dataset_version,
            latest_run_at=latest_run_at,
            head_to_head=_render_head_to_head_html(results_by_model_id),
        )
    )
    print(f"wrote {SITE_DIR / 'index.html'} ({len(rows)} model(s))")


if __name__ == "__main__":
    main()
