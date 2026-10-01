# LLM run v2 prompt (2026-10-01)

Sent verbatim to each subagent by the orchestrating session, with `{DIR}`, `{N}`
and the condition block filled in. One agent per (condition, model, batch); see
`scripts/eval/prepare_llm_run_v2.py`. Kept in the repository because the
2026-09 runs' prompts were not, and could only be reconstructed from notes.

---

You are taking part in a chart data-extraction benchmark. Work only inside
`{DIR}`. Do not read, list or search anything outside that directory (no other
directories, no git repositories, no web). Your answers are compared against
hidden ground truth; looking for it would invalidate the run.

`{DIR}/images/` holds {N} chart images from published research papers, and
`{DIR}/tasks.json` lists them. For each image, extract the XY data of every
data series plotted in the main chart: each legend entry or distinct
marker/line style, including reference or comparison curves drawn in the plot.
Ignore insets and other panels.

- Marker series: one point per marker, at the marker centre.
- Lines without markers: enough points to follow the shape (at least one per x
  tick interval, more where it bends).
- A line joining the markers of one series is part of that series, not a
  separate one. Error bars are not series.

{CONDITION}

You may use any method: look at the images, crop and zoom, write and run
Python (python3 with numpy and Pillow is available).

Write `{DIR}/predictions.json`:

```json
{"fig_001.png": [{"label": "legend text or a short description", "x": [..], "y": [..]}, ...], ...}
```

one key per task id, plain numbers, x and y of equal length. {AXIS_NOTES}
Rewrite the file every few figures so finished work is never lost. Answer
every task; if a figure is hard to read, still give your best attempt.

When done, check that the file parses and has all {N} keys. Final reply: first
line `MODEL: <your exact model id>`, then one line with the figure, series and
point counts measured from the file you wrote.

---

## `{CONDITION}`, calibrated

Each task gives `x_range`, `y_range`, `x_scale` and `y_scale`: the data values
spanned by each axis and whether it is linear or log. Use them to convert
positions in the image to values, and report values in that same space.

## `{CONDITION}`, noaxis

No axis ranges are given: read the printed tick labels yourself to calibrate
each axis, and decide whether it is linear or log. Each task gives `x_report`
and `y_report`, the rule for which numbers to report on that axis. Follow them
exactly.

## `{AXIS_NOTES}`, noaxis only

Also write `{DIR}/axis_notes.json`:
`{"fig_001.png": {"x_range": [lo, hi], "y_range": [lo, hi], "x_scale": "linear|log", "y_scale": "linear|log", "note": "how you calibrated"}}`
with the axis calibration you read, in the same space as your reported values.
