# 方式D, Claude as orchestrator (2026-10-06)

Written into each sealed directory as INSTRUCTIONS.md by
`scripts/eval/prepare_orchestrator_claude.py`, with `{DIR}`, `{N}` and the
condition block filled in. Same figures and task ids as the v3 noaxis run
(condition 1) and the pixcal run (condition 2). The agent may only choose and
configure the fixed tools (docs/design/local-model.md「方式D」); this
measures how well tools can be chosen, an upper bound for the local
orchestrator.

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
| `line_extract` | starry-digitizer's Line Extract: a point every dx px along pixels of a colour. `color`, `distance_pct`, `dx_px` (10), `dy_px` (10), `mask`. Only for series drawn as lines without markers. |
| `mask` | check a mask: `include` [[x0,y0,x1,y1],…], `exclude` [[x0,y0,x1,y1],…], `frame` ([x0,y0,x1,y1] or a tick_calibration result name), `frame_margin` (px, default 2). Every extraction tool takes the same object, or this result's name, as `mask`. |
| `to_values` | `result` (points result name), `calibration` (calibration result name): the values the answer would get. |
| `render_overlay` | `results` (list of result names), `calibration` (optional): draws each series in its own colour labelled name:index (and the frame and ticks) over the figure, writes `work/<task id>/<save name>.png`. Look at it. |

Then, for each figure:

```
./tool answer <task id> '{"series": [{"from": "r2", "index": 0, "label": "x=0.1"}, {"from": "r3", "index": -1, "label": ""}], "calibration": "r1"}'
```

`index` is a series index of that result, or -1 for all its series. A series
named twice is taken once. {ANSWER_NOTE} `./tool check` counts what
predictions.json holds and lists the missing task ids.

{CONDITION}

A good routine: calibrate, run the detector, overlay, look closely
(zoom by viewing the image), then fix what is wrong — mask the legend, change
thresholds or grouping, extract a series by its colour — overlay again, and
answer with the series that are right. Answer every task; if a figure is
hard, still answer with the best tool result you have. `./tool answer` can be
re-run for a figure; the last one counts.

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
