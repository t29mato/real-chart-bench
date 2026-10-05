
You are taking part in a chart data-extraction benchmark. Work only inside
`{DIR}`. Do not read, list or search anything outside that directory (no other
directories, no git repositories, no web). Your answers are compared against
hidden ground truth; looking for it would invalidate the run.

`{DIR}/images/` holds {N} chart images from published research papers, and
`{DIR}/tasks.json` lists them. For each image, extract the measured data
points plotted in the main chart: every marker series (each legend entry with
markers, or distinct marker style). Ignore insets and other panels.

- Marker series: one point per marker, at the marker centre.
- Do not output fit curves, regression or trend lines, guide-to-the-eye lines,
  or theoretical/model curves, even when they appear in the legend.
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
