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
| **Verified real-image pairs** | Live, growing — **111 verified** (+9 rejected, kept as an audit trail). Image↔ground-truth pairing is *not yet solved in general* (design [§7.10](docs/design/benchmark-architecture.md), and [`docs/design/pairing-automation.md`](docs/design/pairing-automation.md) for the automation design), so only manually numeric-cross-verified pairs are scored — see [`data/verified_pairs/registry.json`](data/verified_pairs/registry.json) (reliability over quantity). Live count: `jq '[.[] \| select(.status=="verified")] \| length' data/verified_pairs/registry.json`. |
| **Evaluation harness** | Live. Pure-domain metrics (`src/real_chart_bench/domain/metrics.py`, `matching.py`, `evaluation.py`) + a naive-CV baseline — see [Evaluate your own model](#evaluate-your-own-model) below. |
| **Leaderboard** | Live, auto-deployed from `results/*.json` on every push — <https://t29mato.github.io/real-chart-bench/>. Ranks are scoped to a `dataset_version`: a score is only comparable to another score on the *same* figure set. |
| **LineFormer vs. naive-CV (head-to-head)** | On the **identical 45-figure set** (42 real + 3 synthetic): **LineFormer 0.647**, naive-CV 0.607. LineFormer's 0.627 on real figures against 0.917 on synthetic fixtures is the gap this benchmark exists to measure. LineFormer runs only via a Colab notebook (`notebooks/lineformer_colab.ipynb`; mmcv/mmdetection have no macOS wheels — design §7.16), so it has not been re-run on the larger set. |
| **naive-CV on all 111** | 0.731 (114 figures incl. 3 synthetic). **Not** comparable to LineFormer's 0.647 — it is a different, and on the evidence easier, set of figures (0.813 on the 69 added since LineFormer's run vs. 0.650 on the original 42). See [`docs/experiments/2026-09-02-failure-analysis.md`](docs/experiments/2026-09-02-failure-analysis.md). |
| **Human ceiling** | Harness live, awaiting data. Independent re-digitizations of a stratified 25-figure subset get scored with the *same* metric as models, so the ground truth's own error bar sits on the leaderboard next to every model score. Registered as a pending row until real annotations exist — see `data/human_ceiling/FORMAT.md`. |
| **LLM baselines (Claude/GPT/Gemini)** | Adapter scaffold implemented (`src/real_chart_bench/usecase/llm_client.py`), execution gated on API-cost owner approval — not yet run. |

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
`n_figures`, `mean_summary_score`, `per_figure`).

### How the score is computed

The primary v0 metric (`NormalizedYDistanceMetric`,
`src/real_chart_bench/domain/metrics.py`) linearly interpolates your
predicted curve at each ground-truth x-coordinate and normalizes the y-error
by the ground-truth y-range (a ChartOCR/LineFormer-style approach — see
design §3.1's comparison table). Predicted vs. ground-truth curves within a
figure are matched via the Hungarian algorithm
(`HungarianCurveMatcher`, `domain/matching.py`); an unmatched ground-truth
curve counts as a miss, an unmatched predicted curve as a false positive. A
figure's `summary_score` combines match rate, mean curve distance, and mean
coverage ratio with equal weights (design §7.4). See
`tests/domain/test_normalized_y_distance_metric.py` and
`tests/domain/test_hungarian_curve_matcher.py` for the exact boundary-case
behavior (zero overlap, degenerate ranges, etc.).

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
