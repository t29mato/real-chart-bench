# real-chart-bench

**An open benchmark for chart data extraction — built from real experimental charts in open-access papers.**

> ⚠️ Work in progress. Design approved ([docs/design/benchmark-architecture.md](docs/design/benchmark-architecture.md)); Phase 0-2 done. Phase 3 ground-truth manifest (v0, thermoelectric materials, CC BY 4.0) is collected: 603 papers, 10,057 curves — see [`data/manifest/v0/`](data/manifest/v0/). Evaluation harness not built yet.

<!-- machine-readable capability summary for agents/tools parsing this README -->
`{"tool":"real-chart-bench","purpose":"benchmark chart-data-extraction accuracy (LLMs, dedicated models, classic tools) against real open-access research figures","interface":"cli","cli_entry_point":"real-chart-bench","cli_output_formats":["json","text"],"status":"pre-alpha"}`

## Why

Existing chart-extraction evaluations (LineEX, LineFormer, etc.) rely on **synthetic charts**. Real experimental figures in scientific papers are messier: overlapping markers, log scales, poor scan quality, dense legends. This benchmark collects **real charts from open-access papers** (license-checked for redistribution) and evaluates how well existing models — LLMs (Claude, GPT, Gemini), dedicated models (LineFormer, …), and classic tools — recover the underlying data.

Ground truth is planned to come from human-digitized XY data paired with source figures (e.g., the Starrydata ecosystem).

## Roadmap

- [x] Design: data collection pipeline, license filtering, ground-truth pairing, metrics ([docs/design/benchmark-architecture.md](docs/design/benchmark-architecture.md))
- [x] Phase 0: project scaffolding, CI, dependency-direction enforcement
- [x] Phase 1: domain-layer metrics + curve matching (TDD, 100% domain coverage)
- [x] Phase 2: pilot collection (license yield measured on real data: ~6% CC-BY in the thermoelectric-materials corpus)
- [x] Phase 3: dataset v0 ground truth (thermoelectric materials) — full-scale collection done: 603 CC-BY papers, 2,555 figures, **10,057 ground-truth curves** ([`data/manifest/v0/`](data/manifest/v0/), [findings](docs/experiments/2026-08-16-v0-full-collection.md)). Image pairs collected for 179 papers (2,458 candidate images, not yet auto-paired to figure IDs — see the [panel-splitting investigation](docs/design/benchmark-architecture.md)); permanent image hosting TBD
- [ ] Evaluation harness + baseline results for existing models
- [ ] Public leaderboard

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

## License

TBD (dataset and code may carry separate licenses; figures are included only where their license permits redistribution).
