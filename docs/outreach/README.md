# Outreach drafts — pending approval

This directory contains **drafts only**. Nothing in here has been sent,
posted, or otherwise transmitted anywhere. Per `CLAUDE.md`, any
outward-facing contact from this project requires the owner's (HQ's)
sign-off before it goes out — these files exist so that review can happen
in writing, in the repo, before any message leaves it.

- [`lineformer.md`](lineformer.md) — draft email to LineFormer's authors
  (highest priority: this is the only method we've actually scored)
- [`plotextract.md`](plotextract.md) — draft email to the PlotExtract
  (Polak & Morgan) authors
- [`plotpick.md`](plotpick.md) — draft GitHub issue for the PlotPick author
  (no public email exists for this recipient, so it's an issue, not an
  email)

## Current blocker: the repository is private

Every draft above references links (the GitHub repo, the leaderboard, the
figure set) that **will not resolve for an external recipient today**,
because `t29mato/real-chart-bench` is currently a **private** repository.
None of these drafts should be sent as-is until that's resolved — either by
making the repo public (which itself requires owner/HQ approval per
`CLAUDE.md`) or by preparing a shareable alternative (e.g. a zip of the
relevant figures/results sent directly, or a temporary public mirror) for
recipients we don't want to grant full repo access to yet.

## Checklist — must be true before any of these are sent

- [ ] **Repository is public, or a shareable artifact is prepared.**
      Either `t29mato/real-chart-bench` has owner/HQ approval to go public
      (see `CLAUDE.md` — repo publication requires human sign-off through
      HQ), or the specific files each recipient needs (figures, task
      format, per-figure scores) are packaged and ready to send directly
      without requiring repo access.
- [ ] **The leaderboard is reachable by the recipient.** The
      GitHub-Pages leaderboard (`t29mato.github.io/real-chart-bench`) is
      only meaningful to link if the underlying repo/data it reads from is
      also visible — check it actually renders for a logged-out, outside
      viewer, not just for us.
- [ ] **The reviewer can reproduce the run we're describing.** For
      LineFormer specifically: the Colab notebook
      (`notebooks/lineformer_colab.ipynb`) needs to actually run end-to-end
      for someone outside this project, since the draft offers to show Jay
      Lal our wiring — a broken or environment-specific notebook would
      undercut the "please sanity-check us" framing. For PlotExtract/
      PlotPick, since we haven't run their methods yet, this instead means:
      confirm we *can* actually invoke their code (per each draft's
      "how do we run this" question) before promising a batch run.
- [ ] **Known ground-truth issues are either fixed or disclosed.** Anything
      already flagged in `docs/design/benchmark-architecture.md` or
      `docs/experiments/2026-09-02-failure-analysis.md` as a possible
      ground-truth defect (not just a model failure) should be fixed first,
      or explicitly called out in the message if it isn't — the whole pitch
      rests on "we're being straight with you," and finding an
      undisclosed ground-truth bug after the fact would undermine that
      more than disclosing it up front.
- [ ] **Owner has reviewed and approved each draft's wording**, not just
      the decision to reach out at all — the specifics (numbers cited,
      tone, what's asked of the recipient) are the actual content of the
      request for sign-off.
- [ ] **Send order confirmed.** The task brief and these drafts assume
      LineFormer goes first (we have an actual score to show them); confirm
      with the owner whether PlotExtract/PlotPick should wait until after
      LineFormer's response, or go out in parallel.

## What "sending" means once approved

- LineFormer, PlotExtract: an actual email, from a real mailbox the owner
  controls, to the addresses named in each draft.
- PlotPick: an actual GitHub issue opened on `tommycarstensen/plotpick`,
  under an account the owner controls.

None of this happens automatically or as a follow-up to this task — a
human (owner or HQ) needs to explicitly authorize each send.
