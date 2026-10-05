# LLM run "pixcal" prompt (2026-10-05)

Tick pixel positions given (design 7.76/7.77): the person calibrates the axes,
the AI extracts the points. The v3 prompt (`llm_run_v3_prompt.md`) with its
condition block replaced by the calibration block below; nothing else
changed. Written into each sealed directory as INSTRUCTIONS.md by
`scripts/eval/prepare_llm_run_pixcal.py`.

---

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

Each task gives the axis calibration a person measured on this image, as
in WebPlotDigitizer: for each axis, two tick marks with their pixel position
in the image file and their value (`x_ticks`: `pixel_x` and `value`;
`y_ticks`: `pixel_y` and `value`), and whether each axis is linear or log
(`x_scale`, `y_scale`). Pixel positions are in the file's own pixel grid
(`image_size` = width, height): origin at the top-left corner, x to the
right, y downward. Use exactly this calibration to convert marker positions
to values (interpolate in log10 for a log axis) and report values in the
space of the given tick values. You do not need to read the tick labels.

You may use any method: look at the images, crop and zoom, write and run
Python (python3 with numpy and Pillow is available).

Write `{DIR}/predictions.json`:

```json
{"fig_001.png": [{"label": "legend text or a short description", "x": [..], "y": [..]}, ...], ...}
```

one key per task id, plain numbers, x and y of equal length.
Rewrite the file every few figures so finished work is never lost. Answer
every task; if a figure is hard to read, still give your best attempt.

When done, check that the file parses and has all {N} keys. Final reply: first
line `MODEL: <your exact model id>`, then one line with the figure, series and
point counts measured from the file you wrote.

---
