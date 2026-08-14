# real-chart-bench

**An open benchmark for chart data extraction — built from real experimental charts in open-access papers.**

> ⚠️ Work in progress. Currently in the design phase; not yet usable.

## Why

Existing chart-extraction evaluations (LineEX, LineFormer, etc.) rely on **synthetic charts**. Real experimental figures in scientific papers are messier: overlapping markers, log scales, poor scan quality, dense legends. This benchmark collects **real charts from open-access papers** (license-checked for redistribution) and evaluates how well existing models — LLMs (Claude, GPT, Gemini), dedicated models (LineFormer, …), and classic tools — recover the underlying data.

Ground truth is planned to come from human-digitized XY data paired with source figures (e.g., the Starrydata ecosystem).

## Roadmap

1. Design: data collection pipeline, license filtering, ground-truth pairing, metrics
2. Dataset v0 (thermoelectric materials domain first)
3. Evaluation harness + baseline results for existing models
4. Public leaderboard

## License

TBD (dataset and code may carry separate licenses; figures are included only where their license permits redistribution).
