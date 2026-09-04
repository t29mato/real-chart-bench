# Outreach draft — PlotPick author (Tommy Carstensen)

> **STATUS: DRAFT — NOT SENT. Pending owner/HQ approval per `CLAUDE.md`.**
> No GitHub issue has been opened, nothing posted anywhere. This file exists
> only for the owner to review, edit, or reject before any contact is made.
> See `docs/outreach/README.md` for the pre-send checklist — the repo is
> currently **private**, so links in the draft below would not resolve for
> an external recipient if sent today.

## Recipient

- Tommy Carstensen (single author) — Copenhagen Research Centre for
  Biological and Precision Psychiatry, Copenhagen University Hospital.

## Channel

**GitHub issue only** — no email is published anywhere for this author. The
only public contact channel is `github.com/tommycarstensen/plotpick/issues`.
Drafted as an issue, not an email, accordingly.

## Reference details (for the owner's verification)

- Paper: "PlotPick: AI-powered batch extraction of numerical data from
  scientific figures", arXiv:2605.06021, submitted May 2026.
- Repo: https://github.com/tommycarstensen/plotpick (MIT). Streamlit demo:
  https://plotpick.streamlit.app/.
- Their evaluation used ChartX and PlotQA, comparing 6 VLMs against DePlot.
- We have **not** run PlotPick yet — this is an invitation, not a report of
  a result, same caveat as the PlotExtract draft.

---

## Draft GitHub issue

**Repo:** `tommycarstensen/plotpick`
**Title:** Real-paper-figure benchmark looking to test PlotPick — quick questions before we do

**Body:**

Hi Tommy,

I'm building a small open benchmark (real-chart-bench) that scores chart-
data-extraction methods against real experimental figures from open-access
papers, using Starrydata's human-digitized curves as ground truth. Your
PlotPick paper evaluates against ChartX and PlotQA; our figures are a
different flavor — actual scanned/rendered figures from materials-science
papers rather than the ChartX/PlotQA test sets — so I thought a run against
ours might be a useful complementary data point for you, and I wanted to ask
before doing it rather than surprise you with a number later.

We haven't run PlotPick on our figures yet. What exists right now: 111
verified real-figure/ground-truth pairs from CC BY 4.0 open-access papers,
plus an evaluation harness and two baselines already scored — a naive
color-pixel tracer (0.731 mean across all 111) and LineFormer (0.627 on a
42-figure subset, notably below its 0.917 on our small synthetic fixture
set — a gap we're still trying to pin down).

Limitations up front, since I'd rather you know them before looking at
anything: only 111 figures so far, one domain (thermoelectric materials
— about 94% of figures cluster in just 7 physical quantities), essentially
no log-x-axis figures (1 of 111), and the current task version hands the
model the axis calibration (ranges + linear/log scale) rather than asking
it to read the axes itself, so it's testing curve-tracing more than
end-to-end chart reading. An axis-reading task is planned but not live.

Two quick questions before we try wiring PlotPick in: (1) is the Streamlit
demo the intended way to run a batch of figures, or is there a scriptable
entry point in the repo we should use instead for a ~111-figure batch run;
and (2) is there anything about handing the model pre-given axis calibration
(rather than asking it to read tick labels) that would misrepresent how
PlotPick is meant to work?

The repository is currently private while we finish verifying ground truth,
so I can't link you to it directly yet — happy to share the figure set and
task format here or by whatever channel works for you.

Thanks,
[owner GitHub handle]

---

## Notes for the owner

- Channel is GitHub issue, not email, because no email is published for
  this author — flagged explicitly per the task brief.
- No PlotPick score claimed; framed as "we would like to run this," not a
  report of a result, same as the PlotExtract draft.
- Repo-private blocker called out explicitly.
