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
