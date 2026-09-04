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
The 25-figure subset actually selected for this round, ordered
quickest-first, is `WORKLIST.md` — that's the file to actually work through.

## Protocol for the re-digitization itself

This is the part that determines whether the resulting number means
anything at all. Read this whole section before digitizing your first
figure, not partway through.

### Independence is the entire point — do not look first

`compute_human_ceiling.py` measures **disagreement between two independent
attempts** at reading the same chart. That is the quantity, full stop. The
moment you look at the first attempt before making the second, you are no
longer measuring independent disagreement — you are measuring how well a
second reading can be pulled toward a first one it was shown, which biases
the score toward zero and makes the published "human ceiling" a lie about
its own reliability, not a conservative one.

Concretely, before digitizing a figure on the worklist, do **not**:

- open `data/verified_pairs/ground_truth.json` and look at that
  `figure_id`'s curves or values,
- open anything under `data/verified_pairs/audit/` for that figure
  (`plots/`, `overlays/`, or `review.html`) — the audit re-plot **is** the
  existing digitization rendered back over the chart; seeing it is
  equivalent to seeing the ground truth,
- ask a model or teammate "what does this chart show" in a way that could
  echo back digitized values,
- look at the number of series or points reported for that figure anywhere
  in this repo (including `WORKLIST.md`'s own table — those columns exist
  only so you can plan your time; see that file's own warning).

Work from the **image file only** (the `image path` column in
`WORKLIST.md`), plus the source paper's text/caption if you need it to
resolve what a series actually is. It's fine — expected, even — to
cross-check your own calibration and point placement against the printed
axis and legend as many times as you want; "independent" means independent
of the *other* digitization, not careless about your own.

If you slip and look at the ground truth or audit plot for a figure before
finishing it, say so in that annotation's `notes` field rather than quietly
discarding the lapse, and flag it to 司令塔 — a contaminated row should
probably be excluded from the published ceiling rather than silently
counted, and that's a policy call above this document's pay grade, not
something to decide unilaterally mid-task.

### Digitize in the paper's printed units, on the printed axis

Read values directly off the axis as printed in the figure (degrees
Celsius if that's what's printed, not Kelvin; whatever the printed y-axis
unit is). Do not convert units in your head while digitizing — that's an
extra manual step that only introduces its own arithmetic error. The
harness compares against `ground_truth.json`'s stored values using the
registry's/ground truth's own unit handling; converting units correctly is
the pipeline's job, not the annotator's.

### Every required field, every time

Every file under `annotations/` must be schema-valid (see "Record schema"
above) and, in particular, must always set:

- `annotation_source: "human"` — set explicitly, never left to a tool
  default,
- `annotator_id` — your own identity/handle, consistent across all your
  annotation files so the harness can tell "two different people" from
  "the same person twice",
- `annotated_at` — the actual date you did the digitizing.

`scripts/eval/import_human_ceiling_annotation.py` (below) takes these as
required CLI flags precisely so they can't be forgotten.

### Ambiguous or overlapping series: record the failure, don't hide it

Real figures sometimes have series that visually overlap, or a legend
that's genuinely ambiguous about which curve is which. Two different
situations can arise, and the metric treats them differently, so don't
conflate them:

- **You can tell the series apart and digitize it, just with lower
  confidence.** Digitize it as normal, and say so in `notes` (e.g. `"'1250
  degC' and '1300 degC' overlap for x > 900K; series assignment there is a
  best guess"`). This still contributes a curve, and disagreement in that
  region is exactly the kind of signal the human-ceiling measurement is
  supposed to surface.
- **You genuinely cannot separate/identify a series at all** (e.g. two
  lines are fully coincident for their whole range, or the legend doesn't
  let you tell which color is which quantity). Do **not** invent points for
  it, and do **not** just leave it out with no record — the schema requires
  every `curves[]` entry to have at least one point, so an unresolvable
  series can't be represented as an empty curve either way. Instead, name
  it explicitly in `notes` (e.g. `"could not digitize the 'undoped' series:
  fully overlaps 'x=0.02' for the whole plotted range, legend does not
  disambiguate"`). This matters because an *omitted* series (no mention at
  all) and a *disagreeing* series (digitized differently than the other
  annotation) mean very different things to `compare_annotations` — a
  silent omission would just look like "no data," not "this was hard,"
  and would quietly bias the scored subset toward the easy series.

### How long one figure takes

`WORKLIST.md`'s per-row estimates run from about 3 minutes (a 2-series,
6-point figure) to about 16 minutes (a 1-series, 284-point figure), roughly
tracking `points × 3s + series × 30s`. Total for the 25-figure subset is
about 2.7 hours of clicking; budget 3.5-5 hours across more than one
sitting once you include the learning curve on the first few figures and
normal breaks — see that file's own "Time estimate" section.

## Using starry-digitizer for the re-digitization

[starry-digitizer](https://github.com/t29mato/starry-digitizer) (also
present locally at `/home/mato/repos/starry-digitizer` on the owner's
machine) is the Starrydata project's own digitizer web UI — the same kind
of tool the original `ground_truth.json` digitization was almost certainly
done with, and the natural default choice here since it already supports
everything this protocol needs. Checked directly against its source
(`src/domain/services/axisSetCalculator.ts`,
`src/presentation/components/**`, `src/application/services/projectService/`):

- it loads a local image file (PNG/JPG) via an upload control — no server
  round-trip, so digitizing an unpublished/private crop under
  `data/verified_pairs/crops/` works fine;
- axis calibration (two pixel↔value points per axis, independent
  linear/log toggle per axis) matches exactly what this protocol needs
  ("digitize in the printed units, on the printed axis");
- each series is its own **dataset** (name it after the series/legend
  label — this name becomes `series_label` in the converted annotation);
- "Export Project" downloads a `.zip` containing `project.json` (full axis
  calibration + every dataset's clicked points, as pixel coordinates) and
  the source image — this is the file to hand to the converter below.

Workflow:

1. Open <https://t29mato.github.io/starry-digitizer/> (or run it locally
   from `/home/mato/repos/starry-digitizer` with `npm install && npm run
   dev`).
2. Upload the figure's image (the `image path` column in `WORKLIST.md`).
3. Calibrate the axis set: place X1/X2 and Y1/Y2 on two clearly-labeled tick
   marks each, enter their printed values, and toggle log scale on
   whichever axis is logarithmic (see `WORKLIST.md`'s log-axis note).
4. For each series: create a new dataset, name it after the series/legend
   label, and click its points (manual click, or the color-based auto
   detector if the series is a single solid color/marker — check its
   output before trusting it wholesale).
5. If a series can't be digitized at all, don't create an empty dataset for
   it — record it in `--notes` instead (see "Ambiguous or overlapping
   series" above).
6. Click "Export Project" and save the `.zip` it downloads.
7. Convert it into a schema-valid annotation:

   ```bash
   python scripts/eval/import_human_ceiling_annotation.py \
       --project ~/Downloads/sd-<timestamp>.zip \
       --paper-id <paper_id> --figure-id <figure_id> \
       --annotator-id <your-handle> --annotated-at YYYY-MM-DD \
       [--notes "..."]
   ```

   This reads `project.json`'s axis calibration and each dataset's pixel
   points, converts pixel → real value itself (replicating
   starry-digitizer's own linear/log formula — see
   `src/real_chart_bench/adapter/starry_digitizer_import.py`'s docstring),
   and writes
   `data/human_ceiling/annotations/<paper_id>-<figure_id>__<annotator_id>.json`.
   It refuses to run if any axis point was never placed (incomplete
   calibration) rather than silently producing garbage coordinates, and
   refuses to overwrite an existing annotation file unless you pass
   `--force`.

> ⚠️ **The converter has never been run against a real starry-digitizer
> export.** Its pixel→value formula was derived by reading
> starry-digitizer's `AxisSetCalculator` source, and its tests use
> hand-built fixtures — not one file that starry-digitizer actually
> produced. The `project.json` layout inside the exported `.zip` is the
> most likely thing to be wrong.
>
> So **do figure #1 of the worklist as a throwaway dry run**: digitize it,
> export, convert, and check the converted values land on the printed axis
> (compare against the axis tick labels, *not* against `ground_truth.json`
> — that would break independence for that figure, so pick a figure you are
> willing to discard and redo). Fix the converter before doing the other
> 24, rather than discovering the problem after five hours of clicking.

If, for some figure, starry-digitizer turns out not to work well (e.g. its
auto-detector fights a particularly noisy scan and manual clicking is
still needed regardless), that's fine — manual clicking was always the
fallback, not a failure. What matters is the resulting file being
schema-valid and independently produced, not which tool made it. If you
use a different tool (e.g. WebPlotDigitizer), write the annotation JSON by
hand or with your own script — there is no requirement to use
starry-digitizer specifically, it's just the one this repo has a ready
converter for.
