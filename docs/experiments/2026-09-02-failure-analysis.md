# Failure analysis by figure type (2026-09-02)

**Question**: which figure types does each method fail on?

**Inputs used** (all read-only; nothing in this doc changed `registry.json`,
`ground_truth.json`, or `axis_pixel_candidates.json`):

- `results/naive-cv-v0.json` — naive-CV on all 111 currently-VERIFIED real figures
  (+3 synthetic fixtures), `dataset_version: v0-eval-pilot-n111`.
- `results/lineformer-pretrained.json` — LineFormer on the 42 real figures (+3
  synthetic) it was run against on 2026-08-29, `dataset_version: v0-eval-pilot-n42`.
  LineFormer cannot be re-run in this environment (needs mmcv/mmdetection via a
  Colab notebook — design §7.16) so its figure set is frozen at 42.
- `results/naive-cv-v0-lineformer-subset.json` — naive-CV restricted to exactly
  LineFormer's 42+3 figure_ids (see Task 1, same PR). This is the only
  apples-to-apples pairing between the two methods used below.
- `data/verified_pairs/registry.json` (VERIFIED entries only, via
  `usecase/real_image_gate.select_verified_pairings`) for `x_scale`/`y_scale`.
- `data/verified_pairs/ground_truth.json` for series count and points-per-curve.
- The naive-CV extractor's own colored-pixel mask
  (`adapter/naive_cv_extractor.py`), re-run offline against the exact task inputs
  `scripts/eval/run_baselines.py` builds, to get a *code-verified* (not guessed)
  answer to "did this baseline see any non-gray pixels at all" for specific
  figures below.

**Type taxonomy**: reused verbatim from
`usecase/build_leaderboard_breakdown.py` (design §7.38) — `real-linear-x` /
`real-log-x` / `synthetic`, split on the registry's `x_scale`. No second,
competing taxonomy is introduced. Series-count and points-per-curve are used
below only as *supplementary* signals for individual figures, per the task
brief, not as new top-level categories.

## Headline: same-figure vs. different-figure failures

This is the analytically interesting part, so it goes first. Restricting to
the 42 real figures LineFormer was actually scored on (the only figures where
both methods have a number):

| | naive-CV | LineFormer |
|---|---|---|
| zero-score figures (of 42) | 7 | 3 |
| low-score (<0.3) figures (of 42) | 7 | 4 |
| **both** low-score (<0.3) | **1** (`47534-49581`) | |
| naive-CV low, LineFormer not | 6: `446-8724`, `446-8725`, `446-8726`, `5902-15114`, `83-9049`, `17037-20736` | |
| LineFormer low, naive-CV not | 3: `4176-20123`, `4176-20124`, `21682-21283` | |
| Pearson r between the two methods' scores, n=42 | **0.21** (weak) | |

**Reading this**: only one figure out of 42 is a failure for *both* methods,
and it has a specific, already-diagnosed cause (below) that implicates the
image, not the benchmark's ground truth. Every other low score is
method-specific — a figure one method handles fine and the other fails
outright. Combined with the weak (r≈0.21) score correlation across the shared
42, this says most of what's being measured here is *method* weaknesses
(color-only detection vs. line-instance-segmentation, each with a distinct
blind spot), not systematically bad or ambiguous figures in the benchmark.
That is a reassuring signal for the benchmark's own quality, within the
limits of n=42 (see Statistical weakness below).

### The one shared failure: `47534-49581`

Already investigated in design §7.41: the source image is color-inverted
(black background, white markers/text) and is a **pure scatter plot with no
connecting lines**. Re-confirmed here programmatically: naive-CV's
saturation+luminance mask finds **0 colored pixels** in this image (it is
genuinely black/white/gray throughout, not almost-black). Both a
hue-bucket line tracer and a line-instance-segmentation model fail on an
image with no color and no lines to detect — this is a property of the
image, not of either model's architecture in particular, and not a
ground-truth defect (the entry's numeric evidence was independently
verified, §7.32/§7.33). No registry action taken (matches §7.41's existing
conclusion).

### naive-CV-only failures (6 figures): confirmed achromatic-line blind spot

For every one of these 6, re-running the extractor's own colored-pixel mask
against the actual task image gives **0 colored pixels found**, i.e. this
isn't a mis-detection or a tracing bug — the baseline's saturation threshold
correctly reports "no non-gray content" for these images, because the
curves genuinely are drawn in black/gray/dark ink:

| figure_id | colored px | ground truth says |
|---|---|---|
| `446-8724`/`8725`/`8726` | 0 | "black squares", "gray circles", "dark triangles" (registry evidence text, verbatim) |
| `5902-15114` | 0 | single curve, color not stated in evidence — image itself is grayscale |
| `83-9049` | 0 | single curve, image itself is grayscale |
| `17037-20736` | 0 | single curve, image itself is grayscale |

This is exactly the limitation `naive_cv_extractor.py`'s own docstring
documents ("cannot see black/gray line series") and that design §7.32 already
flagged for paper 446 specifically — not new, but now confirmed to be the
*complete* explanation for naive-CV's zero scores on this 42-figure set (all
7 zero-score figures are either this pattern or the shared `47534-49581`
case; there is no zero-score figure in the 42 left unexplained).

### LineFormer-only failures: pure-scatter (no line) images

`4176-20123` (naive-CV 0.878) and `4176-20124` (naive-CV 0.710) are already
diagnosed in design §7.41: both panels are pure scatter plots (markers only,
no connecting line), and LineFormer is a line-instance-segmentation model —
it has nothing to segment. naive-CV succeeds here specifically *because* it
doesn't need lines (median pixel y per column works on marker clusters too).
Re-confirmed programmatically: naive-CV's hue-bucket mask does find 4 valid
color buckets for `4176-20123` (red/cyan/blue/purple, all above the
15px-per-series floor) matching its 5 ground-truth series reasonably well.

`21682-21283` (LineFormer 0.292, naive-CV 0.653) is a 5-series figure with a
lower LineFormer score than naive-CV, but I could not find a data signal
(scale, series count alone) that cleanly explains *why* LineFormer
specifically underperforms here relative to its own average — flagging as
**not yet determinable from the signals available**; would need to look at
the actual LineFormer line-instance output, which isn't in `per_figure`
(only aggregate scores are recorded).

## Score distribution by figure type

Categories per `build_leaderboard_breakdown.categorize_figure` (x_scale-based).

**naive-CV, n=111 real figures** (+3 synthetic, not shown per-type since
synthetic isn't a real-figure "type"):

| category | n | mean score |
|---|---|---|
| real-linear-x | 110 | 0.733 |
| real-log-x | **1** | 0.775 |

**LineFormer, n=42 real figures (its own set)**:

| category | n | mean score |
|---|---| --- |
| real-linear-x | 41 | 0.627 |
| real-log-x | **1** | 0.642 |

**Immediate caveat**: the registry has only **one** `x_scale=log` figure
among all 111 VERIFIED entries (`5902-15112`) — it happens to be in
LineFormer's 42-figure set too. A category with n=1 supports no conclusion
whatsoever about "log-x figures" as a type; both rows above are one data
point dressed up as a category mean. **Do not read anything into the
real-log-x row beyond "this one figure scored 0.775 / 0.642."**

`x_scale` undersells how many figures actually use a log axis, though:
`y_scale=log` (not part of the existing taxonomy, and not proposed as a new
category here — reported as a supplementary cut) covers 13 of the 111
VERIFIED figures. naive-CV's mean score on those 13 is 0.712 vs. 0.736 on
the 98 `y_scale=linear` figures — a small, likely-not-meaningful gap once
`47534-49581` (which is in the log-y set but whose zero score is explained
by color-inversion + no-lines, not by the log scale itself) is set aside.
**Log-y axes do not appear to be a distinct failure mode for naive-CV in
this data** — a mild surprise worth flagging, since a color-hue tracer has
no obvious reason to care about axis scale (it operates purely in pixel
space; the log/linear distinction only affects the calibration step that
converts pixel→data coordinates *after* tracing, so this null result is
architecturally plausible, not just noise). LineFormer's log-y figures (4 of
its 42) aren't broken out separately here since n=4 is equally too small.

## Supplementary signal: series count and points-per-curve

Not a new type taxonomy — grouped means over the same categorize_figure
input data, bucketed by `len(ground_truth[figure_id])` (series count) and
`min(len(curve.x) for curve in ...)` (sparsest curve in the figure). Cell
sizes are shown throughout; several are too small to support a claim (see
Statistical weakness section) and are flagged as such rather than omitted,
per the task brief.

**naive-CV (n=111) by series count**:

| series count | n | mean |
|---|---|---|
| 1 | 7 | 0.339 |
| 2 | 14 | 0.709 |
| 3–4 | 68 | 0.763 |
| 5–6 | 20 | 0.795 |
| 7+ | **2** | 0.648 |

The n=1 bucket's low mean (0.339) is fully explained by composition, not by
"single-series figures are structurally hard": 4 of its 7 members are the
achromatic figures already diagnosed above (`5902-15114`, `83-9049`,
`17037-20736`, plus `47534-49581` which has 2 series but similarly zero
color). Once those are set aside, single-series figures score normally. The
7+ bucket (n=2) is too small to say anything.

**naive-CV (n=111) by sparsest-curve point count**:

| points | n | mean |
|---|---|---|
| ≤5 | 7 | 0.722 |
| 6–15 | 73 | 0.800 |
| 16–30 | **8** | 0.443 |
| 31+ | 23 | 0.627 |

The 16–30 bucket's dip (n=8, mean 0.443) is worth a closer look before
trusting it — at n=8 a couple of outlier figures can swing the mean this
much on their own, and I did not individually re-diagnose all 8. Flagged as
**suggestive, not established**.

**LineFormer (n=42) by series count**:

| series count | n | mean |
|---|---|---|
| 1 | 7 | 0.725 |
| 2 | 14 | 0.679 |
| 3–4 | 14 | 0.654 |
| 5–6 | **5** | 0.318 |
| 7+ | **2** | 0.516 |

The 5–6 bucket is the most interesting line in this table — it lines up with
the general intuition that a line-instance-segmentation model has a harder
time when many curves cross and overlap (`21682-21283`, discussed above, is
in this bucket) — but **n=5 cannot support "LineFormer fails on high series
count" as a finding**. It is consistent with that hypothesis and no more.

**naive-CV's own structural ceiling on series count**: naive-CV's hue-bucket
palette has exactly 7 buckets (`_HUE_BUCKETS` in
`adapter/naive_cv_extractor.py`). Re-running its mask against
`5166-23909` (8 ground-truth series, naive-CV 0.550, LineFormer 0.312 — the
worst-series-count figure for both methods) confirms this concretely: only 5
of the 7 buckets clear the 15px minimum, so at most 5 of the 8 true series
can ever be recovered as distinct curves — a *hard*, architectural ceiling,
not a tunable threshold issue. This is a genuine, confirmed (not
hypothesized) explanation for why naive-CV struggles specifically as series
count climbs past ~6-7, independent of the n=2 cell-size problem above for
the "7+" bucket mean.

Two more worst-list naive-CV figures got the same offline color-mask check
and turned out to be a *different* failure mode from "too few colors, too
many series" — likely **non-chart color contamination** from crops that
include annotations or composite-figure content beyond the plot area:

- `22102-21245`/`22102-21246` (1 ground-truth curve each, scores 0.408/0.441):
  the mask finds **5** and **4** valid color buckets respectively despite
  only 1 true series — these are the "irregular composite layout ... charts
  + SEM images + line-profile plots mixed" crops the registry evidence
  itself already flags as manually cropped from a non-uniform page (see
  registry evidence for these entries). The extra detected colors are most
  likely SEM-image or annotation pixels inside the crop, not additional
  curves, producing spurious predicted series that the matcher can't
  reconcile with the single ground-truth curve.
- `17038-20816` (2 curves, score 0.606): mask finds 4 buckets — same likely
  pattern, not independently confirmed by visual inspection here.

## Worst figures, each method (its own evaluated set)

**naive-CV, worst 7 of 111** (all below 0.5; see above for the achromatic
diagnosis behind 6 of these 7):

| figure_id | score | cause |
|---|---|---|
| `446-8724`, `446-8725`, `446-8726` | 0.0 | achromatic (black/gray/dark) lines, confirmed (0 colored px) |
| `5902-15114`, `83-9049`, `17037-20736` | 0.0 | achromatic image, confirmed (0 colored px) |
| `47534-49581` | 0.0 | color-inverted + no connecting lines (shared failure, §7.41) |

(next-worst: `22102-21245` 0.408, `22102-21246` 0.441 — likely non-chart
color contamination in a composite-image crop, see above; not below 0.5
by definition of "worst 7" but included for completeness of the pattern.)

**LineFormer, worst 3 of 42** (its only zero scores):

| figure_id | score | cause |
|---|---|---|
| `47534-49581` | 0.0 | shared failure, see above |
| `4176-20123`, `4176-20124` | 0.0 | pure scatter, no connecting lines — architectural limitation of a line-segmentation model (§7.41) |

LineFormer's next-worst figures (`21682-21283` 0.292, `5166-23909` 0.312)
are discussed above under series-count; both are consistent with, but do not
individually prove, an overlapping-series difficulty.

## Composition note: naive-CV scores much higher on the 69 figures added since the 42-figure set

Not asked for directly, but surfaced by this analysis and worth flagging:
restricting to the 69 real figures in the current 111 that are *not* in
LineFormer's 42, naive-CV's mean score is **0.813**, vs. **0.650** on the
original 42 (**0.733** across all 111). And **zero** of those 69 score below
0.3 — every naive-CV low/zero score in the whole 111-figure set falls inside
the original 42.

This is descriptive, not causal — I have not determined *why* (candidate
explanations include: the newer 69 happen to have fewer achromatic-line
figures, later verified-pairs work may have selectively favored
easier-to-crop single-panel color charts, or it's simply noise from a
non-random, effort-ordered addition sequence rather than a random sample of
"paper figures in general"). Flagged here because it means **the 0.731
headline is not simply "the 42-figure number, diluted by more of the
same"** — the registry's growth has not been score-neutral, and any future
"has naive-CV gotten better/worse" comparison across dataset_version bumps
needs to account for this composition shift, not just cite the mean.

## Statistical weakness — explicit

- **n=42** is the full LineFormer sample. Every LineFormer-only table above
  (by-type, by-series-count, by-points) has cells as small as **n=1** (log-x)
  and **n=2**–**5** (7+ series, 5–6 series). None of those cells support a
  general claim; they are reported as descriptive counts with the explicit
  understanding that they are not evidence at this n.
- **n=111** is the naive-CV sample, larger but still thin once split: the
  `real-log-x` category is n=1 for *both* methods (the same one figure) —
  this benchmark currently cannot say anything at all about log-x-axis
  figures as a class, only about that one figure.
  The 16–30-points bucket (n=8) and the 7+-series bucket (n=2) are likewise
  too small.
- The **shared-figure overlap analysis (both-fail vs. one-fails) has n=42**,
  and within that, only **7** naive-CV failures and **3** LineFormer
  failures — the "1 figure shared" finding is a real, code-verified fact
  about these exact 10 figures, but generalizing it to "the methods fail
  independently in general" would need many more shared-evaluation figures
  before it's more than a suggestive pattern with a plausible mechanism
  (documented, distinct blind spots) behind it.
- **Claims this document does NOT make, because n does not support them**:
  "naive-CV is worse on figures with many series" (only n=2 in the 7+
  bucket for naive-CV — the 7-bucket *ceiling* explanation for `5166-23909`
  is a confirmed mechanism for that one figure, not a population claim);
  "LineFormer specifically struggles with overlapping series" (n=5 in its
  worst bucket); "log-x figures are hard/easy for either method" (n=1);
  "the newer 69 verified figures are easier because <specific reason>" (no
  causal test was run, only the score gap itself is established).
- What **would** move these from "not yet determinable" to "determinable":
  more log-x and log-y verified figures specifically (currently 1 and 13
  respectively, out of 111); more figures with 5+ series (currently 22 of
  111); and — the single most useful addition — a second real-figure
  LineFormer run at a larger n than 42, since every method-comparison
  finding here is bottlenecked by that number, not by naive-CV's larger 111.

## Files behind this analysis

Everything above was computed from files already in the repo — no new data
files were created or committed. The specific per-figure diagnostics
(colored-pixel counts, hue-bucket membership) were produced by loading
`scripts/eval/run_baselines.py`'s `build_dataset()` and re-running
`adapter/naive_cv_extractor.py`'s masking logic offline against each figure's
actual task image; this is exactly what `NaiveCvModelRunner.extract()` does
internally, just with the intermediate mask inspected instead of discarded.
