# AGENTS.md

Instructions for coding agents (Claude Code, Codex, Cursor, etc.) working in this
repository. See [llms.txt](llms.txt) for a broader map of the repo, and
[README.md](README.md) for the project's purpose and how to use it as a benchmark.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires Python >= 3.11.

## Before you're done: run these, all must pass

```bash
pytest -q
pytest --cov=real_chart_bench.domain --cov-report=term-missing --cov-fail-under=95 tests/domain -q
ruff check .
lint-imports
```

CI runs exactly these four checks on every push to `main` (`.github/workflows/ci.yml`) —
if any of them fail locally, they will fail in CI too.

## Architecture: clean architecture, mechanically enforced

```
infrastructure → adapter → usecase → domain
```

Dependencies only point rightward (toward `domain`); `lint-imports` enforces this as a
CI gate (`pyproject.toml`'s `[tool.importlinter]` contract), not just a convention.

- **`domain/`** — pure logic and value objects. No I/O, no third-party imports beyond
  `numpy`. Must stay at ≥95% test coverage (currently 100%; don't drop it).
- **`usecase/`** — orchestrates domain logic; defines ports (`Protocol` classes) that
  adapters implement (e.g. `ModelRunnerPort`, `PdfFetchPort`).
- **`adapter/`** — concrete implementations of usecase ports (HTTP, PyMuPDF, numpy image
  decoding, etc.). This is where I/O and third-party libraries live.
- **`infrastructure/`** — the CLI entry point and other outermost glue.

`notebooks/` and `scripts/` are excluded from the architecture contract and from
`ruff` (see `pyproject.toml`'s `extend-exclude`) — they're one-off eval/collection
scripts and a Colab notebook, not library code, but still keep them working (some are
directly runnable and documented in README).

## TDD

Tests are written before implementation for anything non-trivial, especially domain
logic (metrics, matching, licensing classification). Look at any file under
`tests/domain/` for the house style: descriptive test names that state the behavior
(`test_two_series_with_no_x_overlap_scores_as_worst_case`, not `test_edge_case_1`), one
behavior per test, boundary cases enumerated explicitly (empty inputs, degenerate
ranges, zero-overlap, etc.) rather than only the happy path.

## Conventions worth knowing before you touch specific areas

- **`data/verified_pairs/registry.json`** is a manually-curated audit trail (not
  generated data) — every entry, including rejected ones, records numeric evidence for
  why it was accepted or rejected. If you add an entry, follow the existing evidence
  style (state the actual numbers you cross-checked, not just "looks right"). See
  `docs/design/benchmark-architecture.md` §7.19/§7.21/§7.27 for the full rationale
  ("reliability over quantity" is a deliberate, repeatedly-reaffirmed project policy,
  not something to relax for convenience).
- **`data/raw/`, `data/cache/`, `data/pilot/`, `data/hf_dataset/`, `data/eval/`** are
  gitignored (regeneratable/large). **`data/manifest/`, `data/verified_pairs/`** are
  committed (small, curated). Don't flip either of these without a good reason recorded
  in the design doc.
- **LLMO**: the CLI (`real-chart-bench`) defaults to `--format json`; every command must
  support it. Keep the machine-readable one-line JSON summary at the top of README.md
  current if you change the project's status/capabilities.
- Design decisions and their rationale live in
  `docs/design/benchmark-architecture.md`, organized as dated, numbered sections
  (§1, §2, ... §7.1, §7.2, ...). When you make a non-trivial design decision, add a new
  numbered subsection rather than editing prose elsewhere — the section numbers are
  referenced from code comments and commit messages throughout the repo, so don't
  renumber existing ones.
- Prefer extending an existing script/module over creating a parallel one that does
  almost the same thing — e.g. `scripts/eval/run_baselines.py` and
  `scripts/eval/fetch_verified_images.py` are the canonical evaluation entry points;
  new baselines should be added to/wired through them, not duplicated.

## Commit messages

Explain *why*, not just *what*, especially for anything touching the verified-pairs
registry or a metric — a future reader (human or agent) should be able to tell whether
a change was a bug fix, a deliberate policy decision, or new data, without having to
dig through the design doc.
