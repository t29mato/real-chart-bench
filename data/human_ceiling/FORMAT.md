# `data/human_ceiling/` — independent re-digitization records

Part of 戦略メモ「柱B: GTの信頼性を定量化する」(quantifying ground-truth
reliability): a subset of `data/verified_pairs/registry.json`'s VERIFIED
figures gets independently re-digitized, and the agreement between that
re-digitization and the original Starrydata ground truth is scored with the
exact same metric the leaderboard scores models with
(`domain.human_ceiling.compare_annotations`, see that module's docstring for
why and how it is symmetrized). The published result is a **human ceiling**
— but only when every contributing annotation actually is one. This
directory, its schema, and the harness in
`src/real_chart_bench/{domain,usecase,adapter}/*human_ceiling*` are built so
that claim is enforced structurally, not just documented.

## Directory layout

```
data/human_ceiling/
  schema.json                 # formal JSON Schema for one annotation record
  FORMAT.md                   # this file
  annotations/
    <paper_id>-<figure_id>__<annotator_id>.json   # one file per (figure, annotator)
```

`annotations/` is intentionally empty in a fresh checkout (only a
`.gitkeep`). Real annotation files land there only once the actual
re-digitization work happens — this repo does not ship example or sample
annotation files in this directory, so that nothing here can later be
mistaken for real data. See `tests/adapter/test_human_ceiling_annotations.py`
and `tests/usecase/test_compute_human_ceiling.py` for fixture-shaped examples
instead.

## Record schema

See `schema.json` for the formal (JSON Schema draft-07) version; the
authoritative parser/validator is
`src/real_chart_bench/adapter/human_ceiling_annotations.py`, which raises a
clear, file-named error on any violation rather than silently
skipping/defaulting a bad record.

```json
{
  "paper_id": "4173",
  "figure_id": "20120",
  "annotation_source": "human",
  "annotator_id": "annotator-b",
  "annotated_at": "2026-09-10",
  "tool": "WebPlotDigitizer 4.7",
  "notes": "optional free text, e.g. an ambiguous series or axis calibration note",
  "curves": [
    { "series_label": "1150 degC", "x": [773.15, 1073.15], "y": [76.7, 82.5] }
  ]
}
```

Mandatory fields:

- **`paper_id` / `figure_id`** — must match an entry in
  `data/verified_pairs/registry.json` with `status: "verified"`. Joined as
  `f"{paper_id}-{figure_id}"`, the same convention every other
  `results/*.json`'s `per_figure[].figure_id` already uses.
- **`annotation_source`** — one of `"human"`, `"llm"`, `"automated"`. **This
  is the field the whole "human ceiling" claim rests on.** It is never
  optional and never defaulted: a missing or unrecognized value is a hard
  parse error (`adapter/human_ceiling_annotations.py`), not silently treated
  as `"human"`. Downstream, `domain.human_ceiling.require_human_ceiling()`
  refuses to let a result be labeled/emitted as `"human_ceiling"` unless
  *every* annotation that contributed to it has `annotation_source: "human"`
  — a mixed or fully non-human set is still scored and reported, but under an
  honestly different label (`mixed_source_agreement` /
  `machine_agreement`; see `usecase/build_human_ceiling_result.py`). This
  refusal is exercised in `tests/domain/test_human_ceiling.py` and
  `tests/usecase/test_build_human_ceiling_result.py`.
- **`annotator_id`** — identity of whoever/whatever produced the
  digitization (a person's handle for `human`; a model/tool id such as
  `"gpt-4o-digitizer-v1"` for `llm`/`automated`).
- **`annotated_at`** — ISO 8601 date (`YYYY-MM-DD`).
- **`curves`** — non-empty list of `{x: [...], y: [...], series_label?}`,
  one entry per data series digitized in this annotation. `x`/`y` must be
  equal-length, non-empty numeric arrays.

Optional fields: `tool`, `notes`.

## How a figure gets scored

`scripts/eval/compute_human_ceiling.py` compares, per figure, **two**
annotations: the figure's existing entry in
`data/verified_pairs/ground_truth.json` (the original Starrydata
digitization — treated as `annotation_source: human`, since that is what it
is) and the corresponding record(s) under `data/human_ceiling/annotations/`.
A figure with only one available annotation (i.e. no independent
re-digitization has been added yet) is skipped, not scored as some
placeholder value — see `usecase/compute_human_ceiling.py`'s
`PENDING_NO_ANNOTATIONS`/`skipped` handling. A figure with more than two
annotations is also skipped explicitly (the harness does not guess which two
to pair) rather than silently picking one arbitrarily.

## Choosing which figures to re-digitize

`scripts/eval/select_human_ceiling_subset.py` picks a 20–30 figure subset
from the 111 VERIFIED registry entries, selected to **cover the
distribution** (linear vs. log axes, single vs. multiple series, marker
density, number of points, y-quantity) rather than for convenience — see
that script's own docstring and `usecase/select_human_ceiling_subset.py`.
