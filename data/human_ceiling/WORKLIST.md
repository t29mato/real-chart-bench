# Human-ceiling re-digitization worklist

25 figures, selected by `scripts/eval/select_human_ceiling_subset.py`
(coverage subset over axis type / series count / point count / marker
density / y-quantity — see that script's `--target-size 25` output and
FORMAT.md's "Choosing which figures to re-digitize" section). This file is
the checklist for actually doing the re-digitization pass over them.

**Before you digitize a single point, read the "Independence" section of
`FORMAT.md`.** Looking at the existing ground truth, curve, or audit re-plot
for a figure before you digitize it invalidates that figure's row —
delete/redo it rather than keep it.

## Why this order

The rows below are sorted **quickest first**, not by paper or figure
number. Starting is the hard part of a 25-figure backlog, so the first
dozen rows are deliberately the ones with the fewest series and the fewest
points — a handful of 3-6 minute figures to get through axis-calibration
mechanics and build momentum, before the four multi-series,
high-point-count figures at the bottom (which run 8-16 minutes each). The
sort key is `points × 3s + series × 30s` (a rough per-click / per-series
cost, see below) — not points alone, because a figure with few points but
several series to color-separate is not actually faster than a
single-series figure with a few more points on it.

"series" and "points" below are **the existing ground-truth's own counts**
(from `data/verified_pairs/ground_truth.json`), given only so you can plan
your time — **do not open ground_truth.json, or look at the number of
series/points it reports, while actually digitizing the figure.** Knowing
"there should be 4 series" before you've looked at the chart is itself a
form of anchoring; if the printed legend is genuinely ambiguous about how
many series there are, resolve that from the image and the paper text, not
from this table.

## Time estimate

Sum of the per-row estimates below: **~2.7 hours of clicking**, plus
realistic overhead — expect your first 2-3 figures to take noticeably
longer than their estimate while you get used to starry-digitizer's axis
calibration UI (see FORMAT.md), and budget breaks. **Plan for 3.5-5 hours
total** across one or more sessions, not one sitting — a tired last few
figures are exactly where independence and care slip.

## Checklist

| # | done | paper_id-figure_id | figure ref | DOI | image path | series (existing GT) | points (existing GT) | est. time |
|---|---|---|---|---|---|---|---|---|
| 1 | [ ] | 17038-20816 | Fig. 4a | 10.1038/srep19014 | `data/verified_pairs/images/17038/p05_embedded_5.jpg` | 2 | 6 | ~3 min |
| 2 | [ ] | 46278-51437 | Fig. 4A sigma | 10.3389/fenrg.2014.00009 | `data/verified_pairs/crops/46278/fig4a.png` | 3 | 19 | ~4 min |
| 3 | [ ] | 5902-15112 | Fig. 1 | 10.12693/aphyspola.127.287 | `data/verified_pairs/crops/5902/corrected_fig1.png` | 1 | 46 | ~4 min |
| 4 | [ ] | 28331-28501 | Fig. 5(b) | 10.1007/s40145-020-0382-9 | `data/verified_pairs/crops/28331/fig5b.png` | 3 | 27 | ~4 min |
| 5 | [ ] | 22102-21245 | Fig. 3a | 10.1038/s41598-019-39786-y | `data/verified_pairs/crops/22102/fig3a.png` | 1 | 52 | ~5 min |
| 6 | [ ] | 10939-1534 | Fig. 4(d) | 10.1007/s11664-015-4242-2 | `data/verified_pairs/crops/10939/fig4d.png` | 4 | 27 | ~5 min |
| 7 | [ ] | 3733-11779 | Fig. 2a | 10.1007/s40243-014-0026-5 | `data/verified_pairs/crops/3733/fig2a.png` | 3 | 38 | ~5 min |
| 8 | [ ] | 3733-11782 | Fig. 2d | 10.1007/s40243-014-0026-5 | `data/verified_pairs/crops/3733/fig2d.png` | 3 | 38 | ~5 min |
| 9 | [ ] | 27759-25217 | Fig. 7 | 10.1038/s41598-020-65818-z | `data/verified_pairs/images/27759/p06_embedded_7.jpg` | 4 | 28 | ~5 min |
| 10 | [ ] | 27759-25218 | Fig. 8 | 10.1038/s41598-020-65818-z | `data/verified_pairs/images/27759/p07_embedded_8.jpg` | 4 | 28 | ~5 min |
| 11 | [ ] | 10939-1533 | Fig. 4(c) | 10.1007/s11664-015-4242-2 | `data/verified_pairs/crops/10939/fig4c.png` | 4 | 28 | ~5 min |
| 12 | [ ] | 83-9049 | Fig. 2 | 10.12693/aphyspola.124.728 | `data/verified_pairs/crops/83/corrected_fig2.png` | 1 | 60 | ~5 min |
| 13 | [ ] | 10939-1538 | Fig. 5(d) | 10.1007/s11664-015-4242-2 | `data/verified_pairs/crops/10939/fig5d.png` | 4 | 34 | ~5 min |
| 14 | [ ] | 10939-1529 | Fig. 3(c) | 10.1007/s11664-015-4242-2 | `data/verified_pairs/crops/10939/fig3c.png` | 4 | 39 | ~6 min |
| 15 | [ ] | 10939-1530 | Fig. 3(d) | 10.1007/s11664-015-4242-2 | `data/verified_pairs/crops/10939/fig3d.png` | 4 | 40 | ~6 min |
| 16 | [ ] | 18668-12229 | Fig. 3(e) | 10.1038/srep43262 | `data/verified_pairs/crops/18668/fig3e.png` | 4 | 44 | ~6 min |
| 17 | [ ] | 18668-12231 | Fig. 4(a) | 10.1038/srep43262 | `data/verified_pairs/crops/18668/fig4a.png` | 4 | 44 | ~6 min |
| 18 | [ ] | 18668-12233 | Fig. 4(c) | 10.1038/srep43262 | `data/verified_pairs/crops/18668/fig4c.png` | 4 | 44 | ~6 min |
| 19 | [ ] | 10939-1527 | Fig. 3(a) | 10.1007/s11664-015-4242-2 | `data/verified_pairs/crops/10939/fig3a.png` | 4 | 52 | ~6 min |
| 20 | [ ] | 17044-20739 | Fig. 2a | 10.1038/srep23415 | `data/verified_pairs/crops/17044/fig2a.png` | 2 | 98 | ~7 min |
| 21 | [ ] | 36342-34988 | Fig. 3(a) | 10.1007/s40145-021-0480-3 | `data/verified_pairs/crops/36342/fig3a.png` | 6 | 72 | ~8 min |
| 22 | [ ] | 21682-21283 | Fig. 4a | 10.1038/s41467-019-08784-z | `data/verified_pairs/crops/21682/fig4a.png` | 5 | 97 | ~9 min |
| 23 | [ ] | 5166-23909 | Fig. 5a (panel a) | 10.1038/srep13706 | `data/verified_pairs/crops/5166/fig5a.png` | 8 | 130 | ~12 min |
| 24 | [ ] | 17040-21020 | Fig. 2a | 10.1038/srep20402 | `data/verified_pairs/crops/17040/manual_crop_fig2a.png` | 4 | 220 | ~15 min |
| 25 | [ ] | 17037-20736 | Fig. 6d (panel d) | 10.1038/srep18805 | `data/verified_pairs/crops/17037/fig6d.png` | 1 | 284 | ~16 min |

Row 25 (`17037-20736`) is the one outlier worth calling out ahead of time:
single series, but 284 points — it's dense, and each point takes real
deliberation, not a quick click. Consider doing it on a day you're not
already tired from the previous rows, even though its series-count
simplicity put it in this position by the sort key.

Three rows use a log axis — set the corresponding axis to logarithmic in
starry-digitizer's axis calibration step (see FORMAT.md) before digitizing
them: `5902-15112` (log x-axis), and `27759-25217` / `46278-51437` (log
y-axis).

## After finishing a row

1. Export the project from starry-digitizer (see FORMAT.md's "Using
   starry-digitizer" section) and run
   `scripts/eval/import_human_ceiling_annotation.py` to produce
   `data/human_ceiling/annotations/<paper_id>-<figure_id>__<your-id>.json`.
2. Check the box in the table above.
3. Once all 25 rows are done (or whenever you want an interim read),
   run `python scripts/eval/compute_human_ceiling.py` to score agreement —
   it scores whatever annotations exist, it does not require all 25 to be
   present first.
