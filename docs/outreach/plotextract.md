# Outreach draft — PlotExtract authors (Polak & Morgan)

> **STATUS: DRAFT — NOT SENT. Pending owner/HQ approval per `CLAUDE.md`.**
> No email has been sent, nothing posted anywhere. This file exists only for
> the owner to review, edit, or reject before any contact is made. See
> `docs/outreach/README.md` for the pre-send checklist — the repo is
> currently **private**, so links in the draft below would not resolve for
> an external recipient if sent today.

## Recipients

- Maciej P. Polak — `mppolak@wisc.edu`
- Dane Morgan — `ddmorgan@wisc.edu`
- Dept. of Materials Science and Engineering, University of Wisconsin–Madison

## Channel

Email, to the addresses published on the paper.

## Reference details (for the owner's verification)

- Paper title: **"Leveraging Vision Capabilities of Multimodal LLMs for
  Automated Data Extraction from Plots"** (arXiv:2503.12326, March 2025).
  "PlotExtract" is the method's name *inside* the paper, not the paper
  title — the draft below cites the paper by its actual title.
- Code/data: Figshare, DOI 10.6084/m9.figshare.28559639. No GitHub repo.
- Same authors wrote ChatExtract (*Nature Communications* 15, 1569, 2024).
- We have **not** run PlotExtract yet — unlike LineFormer, there is no score
  to show them. This draft is an invitation to collaborate on running it,
  not a report of a result.

---

## Draft email

**Subject:** Materials-science chart-extraction benchmark — would PlotExtract's real-figure numbers be useful to you?

Hi Maciej and Dane,

I'm building a benchmark (real-chart-bench) for chart-data-extraction methods,
using real experimental figures from open-access materials-science papers
scored against Starrydata's human-digitized ground truth. It's the same domain
as your PlotExtract work — most of our current figures are thermoelectric-
property plots (Seebeck coefficient, electrical conductivity, and similar), so
your method is likely to be tested on exactly the kind of chart it was built
for.

We haven't run PlotExtract on our figures yet — I wanted to reach out before
doing that, since you know the method's failure modes better than we would
guessing from the outside, and because your Figshare code/data release doesn't
include a ready-made inference entry point as far as I could tell, so I may
be missing something about how it's meant to be invoked.

What's actually there right now, for context: 111 verified real-figure/
ground-truth pairs, all CC BY 4.0 open-access papers. Our existing baselines
are a naive color-pixel tracer (0.731 mean score across all 111) and
LineFormer, a line-instance-segmentation model (0.627 on a 42-figure subset,
notably lower than the 0.917 it scores on our small synthetic fixture set —
a gap we're still trying to understand and would eventually like every method
in the benchmark to shed light on, not just LineFormer).

Some limitations worth knowing up front: only 111 figures so far, one domain
(thermoelectric materials — about 94% of the figures cluster in just 7
physical quantities), essentially no log-x-axis figures (1 of 111), and the
current task hands the model the axis calibration (ranges and linear/log
scale per axis) rather than asking it to read the axes itself — closer to
your evaluation's easier setting than a fully end-to-end one. An end-to-end
task (axis reading included) is planned but not live yet.

Small ask: is there a straightforward way to run PlotExtract against a batch
of figures with a script, or does it expect the ChatExtract-style interactive
flow? And is there anything about our task framing (axes given, per-figure
XY series output) that would misrepresent how PlotExtract is meant to be
used, before we plug it in and report a number?

The repository is currently private while we finish verifying ground truth,
so I can't share a live link yet — happy to send the 111-figure set and our
task format directly if useful.

Thanks for your time,
[owner name]

---

## Notes for the owner

- No fabricated PlotExtract score — this draft is explicitly framed as "we
  have not run it yet," per the task brief (there is nothing to show them).
- Domain overlap (materials science, same thermoelectric-adjacent
  quantities) is the opening hook, as instructed, not a project description.
- Repo-private blocker called out explicitly.
