# Outreach draft — LineFormer authors

> **STATUS: DRAFT — NOT SENT. Pending owner/HQ approval per `CLAUDE.md`.**
> No email has been sent, no issue opened, nothing posted anywhere. This file
> exists only for the owner to review, edit, or reject before any contact is
> made. See `docs/outreach/README.md` for the pre-send checklist — several
> items on it (repo is currently **private**) are not yet satisfied, so the
> links below would not resolve for an external recipient if sent today.

**Priority: highest.** This is the only method of the three we have already
scored.

## Recipients

- Jay Lal — `jayashok@buffalo.edu` (corresponding email on the paper; also
  the maintainer of the repo, GitHub `TheJaeLal`)
- cc (optional, owner's discretion): Aditya Mitkari, Mahesh Bhosale, David
  Doermann — A2IL, CSE, University at Buffalo (SUNY), co-authors on the paper

## Channel

Email, to the address published on the paper. Could also be filed as a
GitHub issue on `github.com/TheJaeLal/LineFormer` instead of/in addition to
email — owner's call.

## Reference details (for the owner's verification)

- Paper: "LineFormer: Rethinking Line Chart Data Extraction as Instance
  Segmentation", ICDAR 2023. arXiv:2305.01837, DOI
  10.1007/978-3-031-41734-4_24.
- Their own evaluation used the ICDAR/ICPR CHART-Infographics sets
  (AdobeSynth19 synthetic, UB-PMC22 real PMC figures).

---

## Draft email

**Subject:** Ran LineFormer on 42 real paper figures — could you sanity-check the result?

Hi Jay,

I'm building a small benchmark (real-chart-bench) that scores chart-data-extraction
methods against real experimental figures from open-access papers, using
Starrydata's human-digitized curves as ground truth. I ran your pretrained
LineFormer checkpoint against 42 of those figures (materials-science line/scatter
charts, mostly thermoelectric-property plots) and wanted to show you the result
before treating it as final, in case we ran it wrong.

The number that stands out is a within-our-own-run comparison: the same
checkpoint, scored by the same metric in the same harness, gets **0.917 on our 3
synthetic fixtures and 0.627 on our 42 real paper figures**. To be clear, both of
those are our measurements — the 0.917 is not your reported AdobeSynth19 number
and isn't comparable to it; our synthetic set is only 3 figures and exists mainly
as a sanity check. What makes the pair interesting is that the only thing that
changes between them is the figures themselves.

That gap is big enough that I'd rather ask than assume it's a real property of
the method. Possible explanations on our end: we may be feeding LineFormer images
cropped or scaled differently than it expects, our metric (interpolated per-x
y-distance, Hungarian-matched curves) may penalize things your CHART-Infographics
evaluation doesn't, or the 42 figures we picked may just be unusually hard (dense
legends, overlapping series, scanned figures) relative to UB-PMC22.

For context on what we're comparing against: on the identical 45-figure set
(the same 42 real figures plus 3 synthetic fixtures), LineFormer scores 0.647
and a naive color-hue pixel tracer we also run as a baseline scores 0.607 — so
LineFormer and the naive baseline are close on this combined set, which is
itself a little surprising and part of why I want a second pair of eyes.

Some limitations worth knowing before you look at any of this: the set is only
111 verified figures total (42 of which is what you were scored on), all from a
single domain (thermoelectric materials — about 94% of figures concentrate in
just 7 physical quantities), essentially no log-x-axis figures (1 of 111), and
the current task hands the model the axis calibration rather than asking it to
read the axes itself — so this isn't testing everything LineFormer can do.

Small ask: does 0.627 look like a plausible score for LineFormer on figures like
these, or does something in that setup sound off to you? If you're willing, I
can send the exact 42 figures, our LineFormer wiring (currently a Colab notebook
since mmcv/mmdetection has no macOS wheels in our environment), and the
per-figure scores so you can point at anything that looks wrong.

The repository is currently private while we finish verifying the ground truth,
so I can't share a live link yet — happy to send figures/code directly, or to
make the repo public first if that's easier for you to review, whichever you'd
prefer.

Thanks for your time either way — mostly I want to make sure we're representing
LineFormer fairly before this goes on a public leaderboard.

Best,
[owner name]

---

## Notes for the owner

- The 0.627 vs. 0.917 gap is presented as "please check we ran it right," not
  as a criticism — per the task brief, this is deliberate: it's the honest
  framing and also the one most likely to get a reply.
- I did not claim LineFormer and naive-CV are "tied" — I stated both numbers
  (0.647 LineFormer vs 0.607 naive-CV on the identical 45-figure set) without
  editorializing, since the failure-analysis doc found the two methods'
  errors are weakly correlated (r≈0.21) and fail on different figures for
  different reasons — that's a caveat, not a note this draft needs to carry
  to Jay Lal.
- Repo-private blocker called out explicitly and an alternative (send files
  directly) offered instead of a broken link.
