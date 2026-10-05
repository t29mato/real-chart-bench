# Third-party evaluation code (unmodified copies)

Kept verbatim so the reproduction of published chart-to-table numbers
(design §7.75) uses the authors' own scorer, not a re-implementation.
Excluded from the repo's lint (pyproject `extend-exclude`).

| file | source | licence |
|---|---|---|
| `deplot_metrics.py` | https://raw.githubusercontent.com/google-research/google-research/master/deplot/metrics.py (fetched 2026-10-05) | Apache-2.0, Copyright The Google Research Authors |
| `tinychart_eval_chart2table.py` | X-PLUG/mPLUG-DocOwl `TinyChart/tinychart/eval/eval_chart2table.py` (fetched 2026-10-05) | Apache-2.0, Copyright The Google Research Authors |

The two implement the same table metrics (RNSS = `table_number_accuracy`,
RMS-F1 = `table_datapoints_precision_recall`, text_theta 0.5, number_theta
0.1). Their function bodies are identical apart from whitespace and a
comment; the TinyChart copy inlines `anls_metric` (DePlot's imports it from
pix2struct) and adds TinyChart's `chart2table_evaluator`, so it is the one
the reproduction scripts import (needs `editdistance`, `scipy`, `numpy`).
