# real-chart-bench

**An open benchmark for chart data extraction — built from real experimental charts in open-access papers.**

<!-- machine-readable capability summary for agents/tools parsing this README -->
`{"tool":"real-chart-bench","purpose":"benchmark chart-data-extraction accuracy (LLMs, dedicated models, classic tools) against real open-access research figures","interface":"cli","cli_entry_point":"real-chart-bench","cli_output_formats":["json","text"],"status":"pre-alpha","leaderboard_url":"https://t29mato.github.io/real-chart-bench/"}`

> ⚠️ **Pre-alpha.** The v0 ground-truth dataset, evaluation harness, and a live leaderboard all exist and work today (see below), but the verified real-image evaluation set is still small (growing — see [Status](#status)) and the API may change. See [`llms.txt`](llms.txt) for a curated map of this repo if you're an LLM/agent exploring it, or [`AGENTS.md`](AGENTS.md) if you're a coding agent about to modify it.

**Leaderboard: <https://t29mato.github.io/real-chart-bench/>**

## Why

Existing chart-extraction evaluations (LineEX, LineFormer, etc.) rely on **synthetic charts**. Real experimental figures in scientific papers are messier: overlapping markers, log scales, poor scan quality, dense legends, inconsistent image orientation. This benchmark collects **real charts from open-access papers** (license-checked for redistribution, CC BY 4.0 basis) and evaluates how well existing models — LLMs (Claude, GPT, Gemini), dedicated models (LineFormer, …), and classic tools — recover the underlying XY data, against ground truth from [Starrydata](https://starrydata.wordpress.com/)'s human-digitized curves.

## Status

| Layer | State |
|---|---|
| **Ground-truth manifest (v0)** | Live. 603 CC BY 4.0 papers, 2,555 figures, 10,057 digitized curves — [`data/manifest/v0/`](data/manifest/v0/) |
| **Verified real-image pairs** | Live, growing. Image↔ground-truth pairing is *not yet solved in general* (design [§7.10](docs/design/benchmark-architecture.md)), so only manually numeric-cross-verified pairs are used for scoring — see [`data/verified_pairs/registry.json`](data/verified_pairs/registry.json) (reliability over quantity; rejected candidates are kept as an audit trail, not deleted). Current count: check `jq '[.[] | select(.status=="verified")] | length' data/verified_pairs/registry.json`. |
| **Evaluation harness** | Live. Pure-domain metrics (`src/real_chart_bench/domain/metrics.py`, `matching.py`, `evaluation.py`) + a naive-CV baseline — see [Evaluate your own model](#evaluate-your-own-model) below. |
| **Leaderboard** | Live, auto-deployed from `results/*.json` on every push — <https://t29mato.github.io/real-chart-bench/> |
| **LLM baselines (Claude/GPT/Gemini)** | Adapter scaffold implemented (`src/real_chart_bench/usecase/llm_client.py`), execution gated on API-cost owner approval — not yet run. |
| **LineFormer baseline** | Implemented as a self-contained Google Colab notebook (`notebooks/lineformer_colab.ipynb`, mmcv/mmdetection have no macOS wheels — see design §7.16) — pending an owner run. |

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
independently per axis). v0 scope is curve-tracing given calibration, not
full chart understanding including reading tick labels (design §3.1, mirrors
CHART-Infographics' task 6a/6b split). Return one `Curve` per detected series
in data space; the harness handles matching and scoring.

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
- [ ] LineFormer baseline — notebook ready, pending an owner Colab run
- [ ] Automated leaderboard submission (currently PR-based)
- [ ] Full automatic image↔figure pairing (currently manually verified only)

## License

Code: TBD. Ground-truth data (`data/manifest/v0/`, `data/verified_pairs/`) and the figure crops committed under `data/verified_pairs/crops/`: **CC BY 4.0**, checked per-paper before inclusion (see [`data/verified_pairs/registry.json`](data/verified_pairs/registry.json)'s `license_id` field on every entry, and design §7.1/§7.2/§7.30 for the classification methodology).
