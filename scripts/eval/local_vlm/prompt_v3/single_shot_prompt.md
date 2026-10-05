You are taking part in a chart data-extraction benchmark. Your answer is
compared against hidden ground truth.

The attached image is a chart from a published research paper; the task below
describes it. Extract the measured data points plotted in the main chart:
every marker series (each legend entry with markers, or distinct marker style).
Ignore insets and other panels.

- Marker series: one point per marker, at the marker centre.
- Do not output fit curves, regression or trend lines, guide-to-the-eye lines,
  or theoretical/model curves, even when they appear in the legend.
- A line joining the markers of one series is part of that series, not a
  separate one. Error bars are not series.

{CONDITION}

Task:
{TASK_JSON}

Answer with this JSON and nothing else:

```json
{"{ID}": [{"label": "legend text or a short description", "x": [..], "y": [..]}, ...]}
```

one key, the task id, plain numbers, x and y of equal length. If the figure is
hard to read, still give your best attempt.
