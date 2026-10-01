# `ground_truth_supplement/` — series the figure draws that Starrydata did not digitize

Starrydata records a paper's own data. Series a figure draws from **other
works** (reference or comparison curves, a literature sample) are therefore
not in `../ground_truth.json`. The benchmark task is to extract every series
the figure draws, so without them a model that reads such a curve correctly
is scored as a false positive (owner decision 2026-10-01, design §7.65).

These files add the missing series. They are kept apart from the Starrydata
ground truth, with who digitized them, when, and why, so the Starrydata part
stays exactly what anyone can download. Scoring merges both
(`src/real_chart_bench/adapter/ground_truth_store.py`); every merged curve
carries `source: "starrydata"` or `source: "real-chart-bench-supplement"`,
and `dataset_version` gets a `-gtsup{N}-{hash}` suffix so scores on the
supplemented ground truth are never ranked against scores on the original.

## Record (one file per figure: `<paper_id>-<figure_id>.json`)

```json
{
  "paper_id": "18869",
  "figure_id": "18874",
  "digitized_by": "t29mato",
  "digitized_at": "2026-10-02",
  "tool": "starry-digitizer",
  "reason": "3 reference curves from other works drawn in the figure",
  "notes": null,
  "curves": [{ "series_label": "TiCoSb0.85Sn0.15", "x": [300, 400], "y": [4.5, 6.4] }]
}
```

Don't write these by hand: digitize in Starry Digitizer and convert with
`scripts/eval/import_gt_supplement.py`, which validates the result.

## Workflow

1. Open <https://t29mato.github.io/starry-digitizer/> and upload the image
   (path in the table below).
2. Calibrate the axes with the values in the table. They are in the same
   space as `registry.json`'s `x_range`/`y_range` — usually the printed tick
   values, but not always (see 48032-51688).
3. Digitize **only the missing series**, one dataset each, named after the
   legend or in-figure label. Lines without markers: click points along the
   line, enough to follow its shape (roughly every tick interval, more where
   it bends).
4. Export Project, then:

   ```bash
   python scripts/eval/import_gt_supplement.py \
       --project ~/Downloads/sd-<timestamp>.zip \
       --paper-id <paper_id> --figure-id <figure_id> \
       --digitized-by t29mato --digitized-at YYYY-MM-DD \
       --reason "<what these series are>"
   ```

5. Rescore (no model is re-run; every answer is archived):

   ```bash
   python scripts/eval/run_baselines.py
   python scripts/eval/run_lineformer.py --score-only
   for c in full calibrated rest noaxis; do python scripts/eval/score_llm_predictions.py $c; done
   python scripts/leaderboard/generate.py
   ```

## Work list (2026-10-01) — on hold

The owner chose (2026-10-01) to exclude these figures as `gt_incomplete`
instead of digitizing the missing series now. The list stays here so they
can be brought back: add the supplement, then remove the figure's
`excluded_reason` in `registry.json`, then rescore.

Found from the registry's pairing evidence ("un-digitized … curves") and from
figures where the top LLMs agreed on series the ground truth lacks; each one
checked against the image.

| figure | image | missing series | X axis (enter) | Y axis (enter) |
|---|---|---|---|---|
| 18869-18874 | `data/verified_pairs/crops/18869/fig3c.png` | 3 reference curves: ZrCoSb0.80Sn0.20 (red dotted), ZrCoSb0.85Sn0.15 (blue dashed), TiCoSb0.85Sn0.15 (orange dash-dot) | linear, 300 / 1000 | linear, 0 / 40 |
| 18869-18875 | `data/verified_pairs/crops/18869/fig4b.png` | 3 dash-dot literature curves (TiCoSb / NbFeSb / ZrCoSb refs) | linear, 300 / 1000 | **log**, printed ticks |
| 3733-11781 | `data/verified_pairs/crops/3733/fig2c.png` | Cu2-xSe (blue triangles) | linear, 450 / 900 | linear, 0 / 10 |
| 27759-25222 | `data/verified_pairs/crops/27759/fig16a.png` | Ref. (Y0.56Al0.57B14) (purple diamonds) | linear, 400 / 1000 | **log**, 1e-8 / 1e-3 |
| 48032-51688 | `data/verified_pairs/images/48032/p05_embedded_6.jpg` | SrZrO3 [26] (orange line), SrHfO3 [36] (black line) | linear, 0.8 / 1.2 | **log**: the axis prints lg σ = −12 … −4; enter 1e-12 at "−12" and 1e-4 at "−4" |
| 36342-34994 | `data/verified_pairs/crops/36342/fig5c.png` | κ_min reference line (red, near the bottom) — **owner to decide** whether a theoretical-limit line counts as a series; no model extracted it | linear, 300 / 900 | linear, 0 / 6 |

Out of scope (different axis space, not a missing series): the inset
diffusivity plots in 18668-12231 and 18668-12232.
