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
