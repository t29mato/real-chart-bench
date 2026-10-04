# real-chart-bench

**An open benchmark for chart data extraction — built from real experimental charts in open-access papers.**

<!-- machine-readable capability summary for agents/tools parsing this README -->
`{"tool":"real-chart-bench","purpose":"benchmark chart-data-extraction accuracy (LLMs, dedicated models, classic tools) against real open-access research figures","interface":"cli","cli_entry_point":"real-chart-bench","cli_output_formats":["json","text"],"status":"pre-alpha","leaderboard_url":"https://t29mato.github.io/real-chart-bench/"}`

> ⚠️ **Pre-alpha.** The v0 ground-truth dataset, evaluation harness, and a live leaderboard all exist and work today (see below), but the verified real-image evaluation set is still small (growing — see [Status](#status)) and the API may change. See [`llms.txt`](llms.txt) for a curated map of this repo if you're an LLM/agent exploring it, or [`AGENTS.md`](AGENTS.md) if you're a coding agent about to modify it.

**Leaderboard: <https://t29mato.github.io/real-chart-bench/>**

## Why

Existing chart-extraction evaluations (LineEX, LineFormer, etc.) rely on **synthetic charts**. Real experimental figures in scientific papers are messier: overlapping markers, log scales, poor scan quality, dense legends, inconsistent image orientation. This benchmark collects **real charts from open-access papers** (license-checked for redistribution, CC BY 4.0 basis) and evaluates how well existing models — LLMs (Claude, GPT, Gemini), dedicated models (LineFormer, …), and classic tools — recover the underlying XY data, against ground truth from **Starrydata**'s human-digitized curves — the published, citable dataset, not the starrydata2.org service or its API — so this benchmark depends only on data anyone can download and re-check. Cite: [Katsura et al., *STAM: Methods* 5(1), 2025](https://doi.org/10.1080/27660400.2025.2506976) for the database, and the [figshare snapshot](https://figshare.com/projects/Starrydata_datasets/155129) (CC BY 4.0) for the data itself. Note that figshare publishes a *new dated snapshot each month*, each with its own DOI (the most recent verified at the time of writing is [`10.6084/m9.figshare.33399463.v1`](https://doi.org/10.6084/m9.figshare.33399463.v1), 2026-08-31) — there is no single permanent dataset DOI, so exact reproducibility requires pinning a snapshot.

## Task definition

**v0 — curve tracing, given calibration.** You are handed the chart image
*and* its axis calibration (`x_range`, `y_range`, and `x_scale`/`y_scale`,
linear or log, independently per axis). Your job is to return the XY data of
each plotted series. You never read tick labels and never convert units —
just map pixels into the given range. This mirrors the second half of the
CHART-Infographics task 6a/6b split (design §3.1).

**v1 — end-to-end, axis reading included (planned, not yet live).** Same
figures, but the calibration is *not* given: a method must read the axes
itself and return data in the paper's printed units. This is the task
general-purpose VLMs are actually being asked to do in practice, and it is
where a direct comparison against VLM-based digitizers becomes meaningful.
The v0 task stays available and scored separately — a method that is strong
at tracing but weak at axis reading should be visible as exactly that.

Being explicit because the repository name promises more than v0 delivers:
**today's leaderboard numbers are v0 numbers**, and a v0 score is not an
end-to-end chart-understanding score.

## Status

| Layer | State |
|---|---|
| **Ground-truth manifest (v0)** | Live. 603 CC BY 4.0 papers, 2,555 figures, 10,057 digitized curves — [`data/manifest/v0/`](data/manifest/v0/) |
| **Verified real-image pairs** | Live. **97 figures**, all CC BY, every one drawn with markers (the 3 line-only figures were excluded on 2026-10-03, `excluded_reason: no_markers`). `dataset_version v0-eval-pilot-n97`. Registry holds 139 VERIFIED entries total; entries with `excluded_reason` are not scored. See [`data/verified_pairs/registry.json`](data/verified_pairs/registry.json). |
| **Evaluation harness** | Live. Pure-domain metrics (`src/real_chart_bench/domain/metrics.py`, `matching.py`, `evaluation.py`) ranked by the point-level F1 (`domain/point_metrics.py`, design §7.67); the curve-distance `summary_score` (normalized y-distance, span floor 5%, design §7.66) is kept as a reference column — see [How the score is computed](#how-the-score-is-computed) + a naive-CV baseline. |
| **Leaderboard** | Live, auto-deployed from `results/*.json` on every push — <https://t29mato.github.io/real-chart-bench/>. Ranks are scoped to a `dataset_version`: a score is only comparable to another score on the *same* figure set. |
| **LineFormer** | **n=97, v0 task:** point F1 0.006 on the 80 non-dense figures (recall 0.166, precision 0.003), summary_score 0.7462 over all 97 — [`results/lineformer-pretrained-n97.json`](results/lineformer-pretrained-n97.json). It outputs each line as a dense run of pixels, so it never singles out the markers: precision, and with it F1, is near zero. Tick-calibrated mapping on 45 figures ([`results/lineformer-pretrained-tickcal.json`](results/lineformer-pretrained-tickcal.json)): point recall 0.876 on the 37 non-dense figures (vs. 0.159 standard) but F1 still 0.027; summary_score 0.8725 vs. 0.7289. Local RTX 4090. |
| **Classic-CV baselines** | **n=97, v0 task:** naive-CV (hue) point F1 0.016 on the 80 non-dense figures (summary_score 0.7430 over all 97) — [`results/naive-cv-v0.json`](results/naive-cv-v0.json); achromatic-CV (luminance) point F1 0.011 (summary_score 0.6634) — [`results/achromatic-cv-v0.json`](results/achromatic-cv-v0.json). Like LineFormer, both trace lines as dense pixels rather than locating markers. |
| **Human ceiling** | Harness live, awaiting data. Independent re-digitizations of a stratified 25-figure subset get scored with the *same* metric as models, so the ground truth's own error bar sits on the leaderboard next to every model score. Registered as a pending row until real annotations exist — see `data/human_ceiling/FORMAT.md`. |
| **LLM baselines (Claude)** | **n=97, v0 task**, axis calibration given (2026-10-01 run) and withheld (no-axis), plus the September runs on the same figures: see [Current results](#current-results-n97) for the point table (80 non-dense figures) and the dense-marker table (17 figures) — [`results/claude-*-v0-r2.json`](results/claude-opus-5-5-v0-r2.json), [`results/claude-*-v0-r2-noaxis.json`](results/claude-opus-5-5-v0-r2-noaxis.json), [`results/claude-*-v0-full.json`](results/claude-sonnet-5-v0-full.json). |
| **Local VLMs** | **n=97, v0 task**, both conditions (2026-10-03): Qwen3.5-9B, Qwen3.8-27B, Gemma 4 31B (all 8bit, mlx-vlm on an M3 Max laptop, offline). Best: Gemma 4 31B, point F1 0.733 (axis given) / 0.745 (withheld) on the 80 non-dense figures — [`results/*-v0-local-v2.json`](results/gemma-4-31b-8bit-v0-local-v2.json), [`results/*-v0-local-v2-noaxis.json`](results/gemma-4-31b-8bit-v0-local-v2-noaxis.json); raw outputs in [`data/local_vlm_run_v2/`](data/local_vlm_run_v2/), inference code in [`scripts/eval/local_vlm/`](scripts/eval/local_vlm/). |
| **Results explorer** | [`scripts/viz/`](scripts/viz/) (export_results_explorer.py + results_explorer.html): render every model's predicted curves overlaid on source images. |

## Current results (n=97)

The 97 figures are split by how densely their markers sit (design §7.72): a
figure whose median within-series nearest-neighbour spacing of ground-truth
points, in the axis-normalized space, is below 2τ = 0.04 is **dense** — markers
overlap, one-to-one point matching is ambiguous there, and the human
digitization itself skips markers. 17 figures are dense, 80 are not.

**Runs** says where the model ran. *cloud*: Claude, launched as a Claude Code
subagent with tools (an agent, free to write code). *local*: an open-weight VLM
on a laptop (Apple M3 Max 128GB, mlx-vlm, 8bit), one inference per figure, no
tools, no network (`HF_HUB_OFFLINE=1`) — the same 97 figures, tasks and ground
truth as the Claude 2026-10-01 run, with a single-shot version of its prompt
(design §7.69). Local outputs that did not parse (mostly cut off at the
8192-token limit) are scored as total misses: 2–5 of 97 figures per model and
condition, listed in each file's `local_run`. Seconds per figure are recorded
but not comparable (the three models ran concurrently).

**Main table — point-level, 80 non-dense figures** (macro, τ = 0.02).
`summary_score` (reference) is over all 97 figures.

| Condition | Model | Runs | n | Point F1 | Recall | Precision | summary_score (97) |
|---|---|---|---:|---:|---:|---:|---:|
| axis given | Claude Sonnet 5.5 | cloud | 80 | **0.953** | 0.949 | 0.960 | 0.9831 |
| axis given | Claude Opus 5.5 | cloud | 80 | **0.951** | 0.989 | 0.932 | 0.9691 |
| axis given | Claude Fable 5 (Sept.) | cloud | 80 | **0.941** | 0.942 | 0.942 | 0.9843 |
| axis given | Claude Opus 5 (Sept.) | cloud | 80 | **0.918** | 0.917 | 0.920 | 0.9847 |
| axis given | Claude Fable 5.1 | cloud | 80 | **0.917** | 0.921 | 0.915 | 0.9786 |
| axis given | Gemma 4 31B (8bit) | local | 80 | **0.733** | 0.737 | 0.732 | 0.9428 |
| axis given | Claude Sonnet 5 (Sept.) | cloud | 80 | **0.672** | 0.673 | 0.675 | 0.9638 |
| axis given | Qwen3.8-27B (8bit) | local | 80 | **0.631** | 0.635 | 0.631 | 0.9243 |
| axis given | Qwen3.5-9B (8bit) | local | 80 | **0.446** | 0.461 | 0.442 | 0.8995 |
| axis given | Claude Haiku 4.5 (Sept.) | cloud | 80 | **0.039** | 0.075 | 0.030 | 0.5597 |
| axis given | Claude Haiku 4.5 | cloud | 80 | **0.030** | 0.123 | 0.020 | 0.6031 |
| axis given | naive-CV | — | 80 | **0.016** | 0.255 | 0.009 | 0.7430 |
| axis given | achromatic-CV | — | 80 | **0.011** | 0.206 | 0.006 | 0.6634 |
| axis given | LineFormer (pretrained) | — | 80 | **0.006** | 0.166 | 0.003 | 0.7462 |
| axis withheld | Claude Opus 5.5 | cloud | 80 | **0.986** | 0.987 | 0.985 | 0.9848 |
| axis withheld | Claude Sonnet 5.5 | cloud | 80 | **0.973** | 0.970 | 0.978 | 0.9838 |
| axis withheld | Claude Fable 5.1 | cloud | 80 | **0.934** | 0.939 | 0.931 | 0.9818 |
| axis withheld | Gemma 4 31B (8bit) | local | 80 | **0.745** | 0.746 | 0.747 | 0.9529 |
| axis withheld | Qwen3.8-27B (8bit) | local | 80 | **0.658** | 0.658 | 0.662 | 0.9221 |
| axis withheld | Qwen3.5-9B (8bit) | local | 80 | **0.470** | 0.476 | 0.468 | 0.8729 |
| axis withheld | Claude Haiku 4.5 | cloud | 80 | **0.031** | 0.030 | 0.034 | 0.2767 |

**Second table — curve distance, 17 dense-marker figures** (mean
`summary_score`; ground-truth points treated as samples of the curve).

| Condition | Model | Runs | n | summary_score | Match rate | Curve distance | Coverage |
|---|---|---|---:|---:|---:|---:|---:|
| axis given | Claude Fable 5 (Sept.) | cloud | 17 | **0.9741** | 1.000 | 0.0550 | 0.977 |
| axis given | Claude Sonnet 5.5 | cloud | 17 | **0.9645** | 0.971 | 0.0581 | 0.981 |
| axis given | Claude Opus 5 (Sept.) | cloud | 17 | **0.9640** | 1.000 | 0.0825 | 0.974 |
| axis given | Claude Sonnet 5 (Sept.) | cloud | 17 | **0.9532** | 1.000 | 0.1175 | 0.977 |
| axis given | Claude Fable 5.1 | cloud | 17 | **0.9424** | 0.931 | 0.0788 | 0.975 |
| axis given | Claude Opus 5.5 | cloud | 17 | **0.9417** | 0.902 | 0.0419 | 0.965 |
| axis given | Qwen3.8-27B (8bit) | local | 17 | **0.9264** | 0.931 | 0.1339 | 0.982 |
| axis given | Gemma 4 31B (8bit) | local | 17 | **0.8103** | 0.843 | 0.2778 | 0.866 |
| axis given | Qwen3.5-9B (8bit) | local | 17 | **0.7852** | 0.853 | 0.3521 | 0.855 |
| axis given | achromatic-CV | — | 17 | **0.7055** | 0.610 | 0.4751 | 0.981 |
| axis given | naive-CV | — | 17 | **0.6736** | 0.604 | 0.3794 | 0.796 |
| axis given | LineFormer (pretrained) | — | 17 | **0.6631** | 0.569 | 0.2539 | 0.674 |
| axis given | Claude Haiku 4.5 | cloud | 17 | **0.6183** | 0.571 | 0.4277 | 0.711 |
| axis given | Claude Haiku 4.5 (Sept.) | cloud | 17 | **0.5808** | 0.534 | 0.4242 | 0.632 |
| axis withheld | Claude Sonnet 5.5 | cloud | 17 | **0.9439** | 0.971 | 0.0629 | 0.924 |
| axis withheld | Claude Fable 5.1 | cloud | 17 | **0.9432** | 0.931 | 0.0492 | 0.948 |
| axis withheld | Claude Opus 5.5 | cloud | 17 | **0.9431** | 0.902 | 0.0362 | 0.964 |
| axis withheld | Gemma 4 31B (8bit) | local | 17 | **0.8546** | 0.902 | 0.2086 | 0.870 |
| axis withheld | Qwen3.8-27B (8bit) | local | 17 | **0.8268** | 0.902 | 0.2722 | 0.851 |
| axis withheld | Qwen3.5-9B (8bit) | local | 17 | **0.7659** | 0.853 | 0.3543 | 0.799 |
| axis withheld | Claude Haiku 4.5 | cloud | 17 | **0.2730** | 0.407 | 0.8210 | 0.233 |

Every results file records the split: `per_figure[].marker_density`
(`median_nn_spacing`, `dense`), `point_metrics` (non-dense figures only,
`n_dense_figures_excluded`, `dense_criterion`) and `dense_marker_metrics`.

## Evaluate your own model

```bash
git clone https://github.com/t29mato/real-chart-bench.git && cd real-chart-bench
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# data/raw/images/ is gitignored (regeneratable). Fetch just the images the
# verified-pairs registry references (a handful of targeted PDF re-fetches,
# not the full 603-paper collection):
python scripts/eval/fetch_verified_images.py

# Run the naive-CV reference baseline end to end (writes results/naive-cv-v0.json):
python scripts/eval/run_baselines.py
```

To evaluate **your own model**, implement `ModelRunnerPort`
(`src/real_chart_bench/usecase/model_runner.py`) — one method:

```python
class ModelRunnerPort(Protocol):
    def extract(self, task: ExtractionTask) -> list[Curve]: ...
```

`ExtractionTask` gives you the chart image (`image_bytes`) plus the *given* axis
calibration (`x_range`, `y_range`, `x_scale`/`y_scale` — linear or log,
independently per axis) — see [Task definition](#task-definition) for the v0
scope and the planned v1 end-to-end task. Return one `Curve` per detected series
in data space (whatever unit the given calibration implies — you never need
to know or convert units yourself, just map pixels to the given range); the
harness handles matching and scoring.

**Units (design §7.47):** most `verified_pairs` entries store `x_range`/
`y_range`/`ground_truth` in the *paper's own printed display units* (e.g.
µV/K, S/cm), not Starrydata's original SI units — converted once, backed by
independently-verified axis-tick readings, specifically so a human auditing
the benchmark can compare ground truth against the source chart with no
mental unit conversion. A minority of entries (no confident conversion
factor yet, or a genuinely non-linear axis like a raw-log10-printed scale)
remain in SI; either way, `x_range`/`y_range` and the curves under that
`figure_id` are always in the *same* unit space as each other, so nothing
about implementing `ModelRunnerPort` changes — the given calibration is
self-consistent regardless of which unit convention a particular entry uses.

See `src/real_chart_bench/adapter/naive_cv_extractor.py` for a complete
reference implementation, and `scripts/eval/run_baselines.py` for how to wire
a `ModelRunnerPort` into `evaluate_model_on_dataset()` against the verified
real-image pairs + synthetic fixtures, and write a `results/<model_id>.json`
in the schema the leaderboard reads (see any existing `results/*.json` for
the exact shape — `model_id`, `model_name`, `dataset_version`, `run_at`,
`n_figures`, `mean_summary_score`, `point_metrics`, `per_figure`).

### How the score is computed

**Primary: point-level F1** (`evaluate_points`,
`src/real_chart_bench/domain/point_metrics.py`, design §7.67). Human
digitizers record markers, so ground-truth points are marker positions, and
the question is whether a method found those points — not whether it traced
a line near them. Both predicted and ground-truth points are normalized by the
axis range (log axes in log10 space). Within each (predicted series,
ground-truth series) pair, points are matched one to one (Hungarian) and a
pair counts when its Euclidean distance is ≤ τ; series are assigned one to one
on `1 − F1_τ`. Unassigned ground-truth series are missed points, unassigned
predicted series are extra points. Per figure this gives `point_recall`,
`point_precision`, `point_f1` and `point_loc_error` (mean distance of matched
points). The leaderboard ranks by **macro `point_f1` at τ = 0.02** (mean over
figures); τ = 0.01 / 0.05 and micro (pooled-point) values are in each results
file's `point_metrics` block, and every figure's counts are in
`per_figure[].point`. Figures whose markers are too dense for one-to-one
matching (median within-series nearest-neighbour spacing < 2τ, design §7.72)
are left out of the point aggregate and ranked on `summary_score` in a
separate table. See `tests/domain/test_point_metrics.py` for the exact
boundary-case behavior.

**Reference: `summary_score`** (`NormalizedYDistanceMetric`,
`src/real_chart_bench/domain/metrics.py`). It linearly interpolates the
predicted curve at each ground-truth x-coordinate and normalizes the y-error
by the ground-truth y-range (floored at 5% of a linear y axis, design §7.66);
curves are matched via the Hungarian algorithm (`HungarianCurveMatcher`,
`domain/matching.py`), and a figure's `summary_score` combines match rate,
mean curve distance and mean coverage ratio with equal weights (design §7.4).
It is kept so every earlier row stays comparable, but it rewards a line that
merely passes near the markers, so it no longer decides the ranking.

### Add your results to the leaderboard

There's no automated submission pipeline yet (planned for a later version —
design §7.5). For now: run your model as above, get a `results/<model_id>.json`,
regenerate the page (`python scripts/leaderboard/generate.py`), and open a
pull request adding both files. A maintainer will review and merge.

## Data

- [`data/manifest/v0/`](data/manifest/v0/) — the full ground-truth manifest (papers, figures, curves metadata). Committed, CC BY 4.0, ~2MB.
- [`data/verified_pairs/registry.json`](data/verified_pairs/registry.json) — the audit trail of manually numeric-cross-verified image↔ground-truth pairs (both accepted and rejected candidates, with evidence). Committed. Each entry records its own `license_id` (design §7.30).
- [`data/verified_pairs/crops/`](data/verified_pairs/crops/) — a small number of manually-corrected figure crops committed directly (needed where automated extraction gets the panel boundaries or image orientation wrong — see design §7.21/§7.24/§7.27 for why each one exists). Also CC BY 4.0.
- `data/raw/` — gitignored. PDFs and extracted candidate images from the full 603-paper collection. Regenerate the subset you need with `scripts/eval/fetch_verified_images.py` (targeted) or `scripts/collect/collect_v0_dataset.py` (full collection, only if you need it — please respect publisher rate limits).

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

pytest -q            # tests
ruff check .          # lint
lint-imports          # clean-architecture dependency-direction check
```

CLI (JSON output by default, for machine/agent consumption; add `--format text` for humans):

```bash
real-chart-bench capabilities
```

See [`AGENTS.md`](AGENTS.md) for conventions (clean architecture, TDD) if you're contributing code, and [`docs/design/benchmark-architecture.md`](docs/design/benchmark-architecture.md) for the full design history and rationale behind every decision above.

## Roadmap

- [x] Design: data collection pipeline, license filtering, ground-truth pairing, metrics
- [x] Phase 0-3: project scaffolding, domain-layer metrics (TDD, 100% coverage), pilot + full-scale data collection (603 CC BY 4.0 papers, 10,057 curves)
- [x] Evaluation harness + naive-CV baseline results
- [x] Public leaderboard (GitHub Pages, auto-deployed)
- [x] Verified real-image pairs (growing — see [Status](#status))
- [ ] LLM baselines (Claude/GPT/Gemini) — scaffolded, pending owner approval to run
- [x] LineFormer baseline — first real-paper-figure score on the leaderboard: 0.627 (42 real figures)
- [ ] Automated leaderboard submission (currently PR-based)
- [ ] Full automatic image↔figure pairing (currently manually verified only)
- [ ] Human ceiling: independently re-digitize a stratified subset and publish the annotator-to-annotator agreement as a leaderboard row, so the ground truth's own error bar is visible next to every model score
- [ ] Ground-truth issue export: figures where the ground truth itself is confirmed wrong, exported with Starrydata identifiers so the upstream dataset can be corrected
- [ ] v1 task: end-to-end extraction with axis reading included (see [Task definition](#task-definition))

## Citation

If you use this benchmark, cite it via [`CITATION.cff`](CITATION.cff) (GitHub's
"Cite this repository" button reads it). If you use the ground-truth data,
please **also** cite the Starrydata dataset and paper listed in that file's
`references` — the curves are theirs, this repository only pairs them with
source figures and scores methods against them.

## License

Code: **MIT** (see [`LICENSE`](LICENSE)). Ground-truth data (`data/manifest/v0/`, `data/verified_pairs/`) and the figure crops committed under `data/verified_pairs/crops/`: **CC BY 4.0**, checked per-paper before inclusion (see [`data/verified_pairs/registry.json`](data/verified_pairs/registry.json)'s `license_id` field on every entry, and design §7.1/§7.2/§7.30 for the classification methodology).
