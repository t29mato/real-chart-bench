# 方式D v3, Claude as orchestrator with verify v3 and blob_extract (2026-10-08)

Written into each sealed directory as INSTRUCTIONS.md by
`scripts/eval/prepare_orchestrator_claude.py --v3`, with `{DIR}`, `{N}` and
the condition block filled in. New seed and task names (key in
data/llm_run_orch3/), every scored figure, the single "as printed" rule in
condition 1. Differences from v2 (claude_prompt_v2.md): the `blob_extract`
tool, verify v3 (line pieces, error-bar caps, fit-curve fragments and
markers split by a white cross are no longer counted as missed markers;
minor ticks with any subdivision no longer flag the calibration) and the v3
tick reading of `tick_calibration` auto (docs/design/local-model.md
「v3: 検証の誤検知と道具の追加」). The rules are otherwise unchanged.

---

You are taking part in a chart data-extraction benchmark as the
**orchestrator of a fixed toolbox**. Work only inside `{DIR}`. Do not read,
list or search anything outside that directory (no other directories, no git
repositories, no web). Your answers are compared against hidden ground truth;
looking for it would invalidate the run.

`{DIR}/images/` holds {N} chart images from published research papers, and
`{DIR}/tasks.json` lists them. For each image, extract the measured data
points plotted in the main chart: every marker series (each legend entry with
markers, or distinct marker style). Ignore insets and other panels.

- Marker series: one point per marker, at the marker centre.
- Do not output fit curves, regression or trend lines, guide-to-the-eye lines,
  or theoretical/model curves, even when they appear in the legend.
- A line joining the markers of one series is part of that series, not a
  separate one. Error bars are not series. Legend symbols are not data.

## Rules of this run (strict)

- **Do not write or run any code of your own.** No Python, no scripts, no
  one-liners, no other programs that open, read or process the images or the
  tool results. The only command you run is `{DIR}/tool` (plus `ls` and
  `cat` inside `{DIR}` to read files).
- **Do not type any number of the answer yourself.** Answers are written
  only by `./tool answer`, from saved tool results.
- You may look at the images and at the overlay PNGs the tools draw, as
  often as you like, and you decide everything: which tool, which settings,
  which colour, which mask, which results form the answer.
- Reading tick labels yourself is allowed in one place only: the `manual`
  mode of `tick_calibration` (condition 1), when the automatic reading fails
  or is wrong.

## The tool

Run from `{DIR}`:

```
./tool <tool> <task id> '<params JSON>' --save <name>
```

The result is printed as JSON and saved as `work/<task id>/<name>.json`.
Parameters that refer to an earlier result use its name (e.g. `"r1"`).
Coordinates are the image file's pixels (origin top-left, y downward).

| tool | params |
|---|---|
| `tick_calibration` | {CALIBRATION_ROW} |
| `marker_detector` | a trained marker detector. `threshold` (0.1–1, default 0.4; lower finds fainter markers), `group_threshold` (default 0.5; larger merges series, smaller splits them), `long_side` (768, 1024 or 1280, default 768; larger helps small markers), `min_points` (default 2), `dup_frac` (duplicate radius as a fraction of the long side, default 0.004), `mask`. Returns series split by marker look. |
| `dominant_colors` | the main non-background colours: `k` (default 8), `mask`. Candidate series colours. |
| `symbol_extract` | starry-digitizer's Symbol Extract: one point per blob of a colour. `color` ("#rrggbb"), `distance_pct` (colour tolerance %, default 1; 3–10 for JPEG / antialiased), `min_diameter_px` (default 5), `max_diameter_px` (default 100), `mask`. |
| `blob_extract` | one point per marker of a colour, for markers the other tools get wrong: pieces of one marker split by a white cross or bar are merged into one point at its centre, and blobs of overlapping or touching markers are split into one point per marker (marker size taken from the colour's typical isolated blob). `color`, `distance_pct` (default 1; 3–10 for JPEG / antialiased), `min_diameter_px` (3), `max_diameter_px` (100), `marker_px` (optional: the single-marker diameter, when the typical blob is not one marker), `merge` / `split` (default true), `merge_gap_px` (optional), `mask`. The result's `info` says how many were merged and split. |
| `line_extract` | starry-digitizer's Line Extract: a point every dx px along pixels of a colour. `color`, `distance_pct`, `dx_px` (10), `dy_px` (10), `mask`. Only for series drawn as lines without markers. |
| `mask` | check a mask: `include` [[x0,y0,x1,y1],…], `exclude` [[x0,y0,x1,y1],…], `frame` ([x0,y0,x1,y1] or a tick_calibration result name), `frame_margin` (px, default 2). Every extraction tool takes the same object, or this result's name, as `mask`. |
| `to_values` | `result` (points result name), `calibration` (calibration result name): the values the answer would get. |
| `render_overlay` | `results` (list of result names), `calibration` (optional): draws each series in its own colour labelled name:index (and the frame and ticks) over the figure, writes `work/<task id>/<save name>.png`. Look at it. |
| `verify` | `series` (exactly as for `answer`), `calibration` (the calibration result name; in condition 1 the one you will answer with), `mask` (optional: regions that are not the main plot, e.g. an inset or a legend box). Checks the series on the image: are the points on markers (hit vs shifted), are there marker-like places with no point (missed markers, a missed series), duplicates, points outside the frame, agreement with the detector's raw peaks, and (condition 1) whether the tick labels fit the calibration. Prints a `verdict` (accept / redo with reasons, and a score) and the `next` step. |

## Verify before answering (enforced)

For each figure, before `./tool answer`, run `./tool verify` on exactly the
series (and, in condition 1, the calibration) you are going to answer.

- If the verdict is **accept**, answer those series.
- If it is **redo**, read the reasons, fix them with a different tool,
  setting or mask (or a corrected calibration), and verify the new attempt.
- At most 2 redos per figure: after the third verify, answer the attempt
  with the highest verify score (`next` names it).

`./tool answer` refuses series that were not verified, that verify said to
redo while redos are left, or that are not the best-scoring attempt once
the redos are used up. Every verify counts as an attempt, so verify only
what you would answer.

Then, for each figure:

```
./tool verify <task id> '{"series": [{"from": "r2", "index": 0}, {"from": "r3", "index": -1}], "calibration": "r1"}'
./tool answer <task id> '{"series": [{"from": "r2", "index": 0, "label": "x=0.1"}, {"from": "r3", "index": -1, "label": ""}], "calibration": "r1"}'
```

`index` is a series index of that result, or -1 for all its series. A series
named twice is taken once. {ANSWER_NOTE} `./tool check` counts what
predictions.json holds and lists the missing task ids.

{CONDITION}

A good routine: calibrate, run the detector, overlay, look closely
(zoom by viewing the image), then fix what is wrong — mask the legend, change
thresholds or grouping, extract a series by its colour — overlay again,
verify the series that look right, and answer them (or redo as verify
says). Answer every task; if a figure is hard, still answer with the best
verified attempt. `./tool answer` can be re-run for a figure; the last one
counts.

When done, run `./tool check` and make sure nothing is missing. Final reply:
first line `MODEL: <your exact model id>`, then one line with the figure,
series and point counts from `./tool check`.

---

## `{CONDITION}`, noaxis

No axis calibration is given. `tick_calibration` with `{"mode": "auto"}` finds
the plot frame and reads the tick labels by OCR; check its ticks on an overlay.
If it fails or is wrong, read the tick labels yourself and give them with
`{"mode": "manual", "x": {"scale": "linear|log", "ticks": [[px, value], ...]},
"y": {...}}` (x ticks: px is the x pixel; y ticks: px is the y pixel; the
failed result lists detected tick-mark pixels in `tick_marks`). Report numbers
on the same scale as the printed tick labels: do not apply a multiplier written
in the axis title (e.g. '(10^4 S/m)' or 'x10^4'); if a tick label itself is
written like 5x10^4, use 50000. Each task's `x_report` / `y_report` repeats
this rule.

## `{CONDITION}`, pixcal

A person has calibrated the axes (each task in tasks.json carries it:
`x_ticks`, `y_ticks`, scales). `tick_calibration` returns this calibration with
the plot frame (`{"mode": "given"}`, the default). Answers are kept in image
pixels and converted to values with the person's calibration after the run,
so the `calibration` of `./tool answer` can be left out.
