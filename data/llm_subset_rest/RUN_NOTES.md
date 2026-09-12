# remaining-101 LLM run — notes that belong next to the scores

Run date 2026-09-12. Condition: axis ranges supplied (the same information
`ExtractionTask` hands every CV baseline). Scores in
`results/claude-*-v0-rest.json`, union with the n=10 subset in
`results/claude-*-v0-full111.json`.

## Each model chose its own extraction method

This is not four models performing one task. It is four agents free to use
tools, and they did not choose the same approach, so the scores are not a
measure of "how well the model reads a chart".

| model | method | series count vs. truth | match_rate |
|---|---|---|---|
| Opus 5 | read the figures; magnified crops for dense cases | median 1.00, mean 0.99 | 0.9559 |
| Sonnet 5 | read the figures directly | median 1.00, mean 1.00 | 0.9653 |
| Fable 5 | read the figures; batched via a small merge helper | median 1.00, mean 1.00 | 0.9547 |
| Haiku 4.5 | wrote a connected-component CV pipeline, grouped by colour | median 2.50, mean 3.10 | 0.4373 |

Haiku's 0.5218 is an over-segmentation artefact: its pipeline fragments each
line into several series and the Hungarian matcher charges for the extras. On
the n=10 subset the same model read the images directly and scored 0.8731, so
0.8731 → 0.5218 is substantially a change of method, not of model.

## Haiku 4.5's run is not reproducible, and its self-report is unreliable

Sequence of events, all observed directly on the file:

1. Reported "101/101 figures, 1055 series, 12072 points". Verified on disk:
   101 keys, no empty entries, no x/y length mismatches, median 9 series per
   figure. Archived and scored — **this is the file the published 0.5218 is
   computed from**, and the number rests on that verification, not on the
   agent's claim.
2. The scratch file then reappeared holding only the first 10 figures. A
   re-archive copied the regression over the finished run; only the git copy
   from the previous commit made it recoverable. `archive_llm_subset.py` now
   refuses to replace a longer archived run with a shorter one.
3. A second completion notification arrived quoting the *same* summary
   (101/1055/12072) while the file on disk held 20 figures, 714 series and
   42507 points — different content for every figure, and a sampling density
   roughly 18x higher per figure. The agent was repeating a remembered
   summary, not measuring what it had just written.

So the Haiku row should be read as one verified run that has not been shown to
be reproducible. The other three models each reported once, and their reported
figure counts matched their files.

## Dataset defects the run surfaced

- **21 figures** supply an axis range in a different unit space than the
  printed axis: ×1e6 (V/K vs µV/K), ×1000 (1/T vs 1000/T), and three log axes
  stored as actual values against a log10-labelled axis. All four models
  independently reported noticing and converting, and the split shows no
  systematic penalty (−0.035 / +0.023 / −0.029 / −0.008), so this is a latent
  hazard rather than an active confound — a model that read the printed axis
  honestly and did not reconcile would be punished for being right. Design
  7.47's display-unit migration is less complete than the registry implies.
- **38965 (fig_084) and fig_086** are cropped such that the x axis is not
  visible at all. Opus and Fable both flagged that their x sampling there is
  assumption, not measurement.
- Multi-panel crops still needing a split: 40067 (fig_051), 40587 (fig_074).
  Every model had to guess which panel the supplied ranges referred to.
