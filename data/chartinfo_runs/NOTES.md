# CHART-Infographics scatter measurement — notes

## Convention mismatches (not reading errors)

- **batch01 fig_29** (PMC4324675 Fig3): each point is labelled in the plot (region
  names), and the ground truth makes every labelled point its own series. Both
  Sonnet 5.5 and Opus 5.5 read all 11 values to within ~0.1 but returned one
  series, so series matching (one to one) scores 0.091 point F1 and ~0 data score.
- **batch01 fig_27** (PMC5575873 Fig5): the ground truth splits points by colour
  into five series; both models returned one series with all six points.

The prompt (`data/chartinfo_pilot/prompt_v2.md`) is kept unchanged for the whole
measurement so batches stay comparable; this class of mismatch is reported
separately instead of being prompted away.

## batch01 (30 figures, fully automatic, prompt v2) — 2026-10-06

| model | combined | name | data | point F1 | recall | precision |
|---|---|---|---|---|---|---|
| Claude Opus 5.5 | 0.690 | 0.870 | 0.630 | 0.842 | 0.841 | 0.844 |
| Claude Fable 5.1 | 0.683 | 0.876 | 0.618 | 0.830 | 0.826 | 0.839 |
| GPT-6.1-Sol (Codex CLI) | 0.671 | 0.887 | 0.599 | 0.828 | 0.845 | 0.836 |
| Claude Sonnet 5.5 | 0.681 | 0.871 | 0.617 | 0.794 | 0.786 | 0.810 |
| ICPR 2020 best (task 6b, upstream ground truth given) | 0.710 | — | — | — | — | — |

- The ground truth ZIP sits on the same machine (`~/.cache/real-chart-bench/chartinfo/`).
  GPT-6.1-Sol's event log shows no command outside its directory (17 commands).
  The Claude agents ran without a command log; they report staying inside.
- Sonnet 5.5 says its values are eyeball estimates this time.

## Self-reported irregularities (batches 02-04)

- batch04 / Sonnet 5.5 / fig_22: points generated from a fitted bell curve
  (0-168 step 2) rather than read marker by marker.
- batch02, batch04 / Sonnet 5.5: values read by eye (no pixel extraction).
- batch03 / Sonnet 5.5: mostly eyeballed; fig_06/07/12/15/17 interpolated
  between anchor readings (fig_12's 240 and fig_17's 71 points are not marker
  counts); a few hidden markers in fig_11/21/26 filled in by guess.
- batch03 / Opus 5.5: fig_15 ~30 hidden markers interpolated; fig_21 ~23 hidden
  points estimated; fig_26 hidden markers at y=0 assumed 0; fig_17 dense curve
  sampled (85 points).
- batch04 / Opus 5.5: fig_10, fig_11 crowded regions read/sampled rather than
  detected; fig_22 101 points fitted on an even grid; hidden points added in
  fig_14, fig_16, fig_30; fig_18 y reported in absolute units (ticks print x10^9,
  against the "as printed" rule).
- batch03, batch04 / Fable 5.1: **stopped mid-run** (owner's rule: stop Claude
  agents past 80% session usage; it read 92% at 02:21 on 2026-10-07). The
  archived files are partial; empty lists are figures not yet answered and score 0.
- batch04 / Fable 5.1: completed 2026-10-07 by a second agent that kept the
  first run's 29 answers and added fig_30 only.
- batch03 / Fable 5.1: completed 2026-10-07 by a second agent that kept the
  first run's 17 answers and added the other 13 (fig_17 dense band resampled;
  fig_26 hidden markers at 0 inferred).
- batch05 / Sonnet 5.5: eyeballed; fig_02, fig_23, fig_26 points generated from
  fitted curves; fig_19 a hand-listed subset (~190 of ~250).
- batch08 / Sonnet 5.5: eyeballed; fig_19 synthetic points generated along
  visible streaks rather than read marker by marker.
- batch05 / Opus 5.5: completed by a second agent (kept the first run's 10
  answers, added 20); fig_23 overlapping series given identical/modelled values;
  hidden points estimated in fig_11/16/30.
- batch10 / Sonnet 5.5: eyeballed; fig_05 reconstructed from fitted curves;
  fig_23 values guessed on an integer grid.
- batch05 / Fable 5.1: completed by a second agent (kept 6, added 24). Its
  per-figure counts were close to Opus's, so the answers were compared: 369 of
  1,792 points equal Opus's exactly, all of them on round grids (fig_19 integer
  sum scores, fig_22 one decimal, fig_16 mostly zeros); only fig_19 is identical
  as a whole. Read as both models snapping to the same grid, not copying; the
  sealed directories of different models are siblings, so this is not proof.
- batch15 / Sonnet 5.5: mostly eyeballed; fig_04, fig_06 interpolated on a
  regular grid from key values.
- batch18 / Sonnet 5.5: fig_14, 16, 19, 22 sampled along curves (fig_22 S_3
  extrapolated by scaling S_2; fig_16 two series identical).
- batch06/07 cross-model overlap check (Opus vs Fable): exact-equal points
  63/1082 (9 off a 0.1 grid) and checked for batch07 -- no sign of copying.
- batch07 / Opus 5.5: fig_15 red series copied from green peak positions
  (hidden under them); fig_02 all series read on the left axis.
- batch07 overlap detail: 214/1927 Opus points equal Fable's exactly; 118 of
  them in fig_16 (x on a 0.1 grid, y ~1e-6 i.e. zero), the rest scattered
  (fig_26: 4 points to 4 decimals). Both read marker centroids in pixels
  through the same ticks, so identical pixels give identical values; not
  treated as copying. Later batches' sealed directories should be separated
  per model.
