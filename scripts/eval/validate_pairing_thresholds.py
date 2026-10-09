"""Threshold-validation experiment for the automatic pairing rule
(docs/design/pairing-automation.md §5 C7 / §6 / §12.2;
docs/experiments/2026-10-10-pairing-threshold-validation.md).

Runs the production candidate pipeline (calibrate_frames -> score_figure_on_frame
-> decide_paper) on every image that exists locally for the papers in
data/verified_pairs/registry.json, then compares the automatic assignment with the
human-verified pairing and sweeps the S / M / contrast thresholds.

No network, no LLM, nothing is adopted or written to the manifest.

    # 1. slow: calibrate + score (cached as JSON, default build/pairing_validation_raw.json)
    python scripts/eval/validate_pairing_thresholds.py collect \
        --extra-root raw=/abs/path/data/raw/images
    # 2. fast: labelling, sweeps, markdown tables on stdout
    python scripts/eval/validate_pairing_thresholds.py analyse
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
from collections import Counter, defaultdict
from multiprocessing import Pool

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from real_chart_bench.usecase.pairing_threshold_validation import (  # noqa: E402
    AMBIGUOUS,
    CORRECT,
    SIBLING,
    UNLABELLED,
    WRONG,
    LabelledAssignment,
    clopper_pearson_upper,
    heldout_papers,
    label_assignment,
    normalise_reference,
    sweep_adoption_rules,
)

CURVES_CSV = (
    pathlib.Path.home()
    / ".cache/real-chart-bench/starrydata-work/ThermoelectricMaterials_curves.csv.gz"
)
RAW = ROOT / "build/pairing_validation_raw.json"
REGISTRY = ROOT / "data/verified_pairs/registry.json"
MIN_IMAGE_PX = 200  # same filters as scripts/collect/generate_pairing_candidates.py
MAX_DARK_SHARE = 0.4
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".img", ".gif", ".bmp", ".webp", ".tif", ".tiff"}


def _work(job):
    from real_chart_bench.adapter.auto_axis_calibration import calibrate_frames, load_rgb
    from real_chart_bench.adapter.pairing_scorer import ink_mask, score_figure_on_frame
    from real_chart_bench.usecase.pairing_candidates import FigureInfo, FrameInfo, decide_paper

    paper_id, images, figures = job  # images: [(key, abs path)]
    frames, scored, pairs, skipped = [], {}, [], []
    for key, path in images:
        rgb = load_rgb(pathlib.Path(path))
        if rgb is None or min(rgb.shape[:2]) < MIN_IMAGE_PX:
            skipped.append([key, "unreadable_or_small"])
            continue
        if (rgb.mean(2) < 140).mean() > MAX_DARK_SHARE:
            skipped.append([key, "dark"])
            continue
        ink = ink_mask(rgb)
        n_frames = 0
        for k, cal in enumerate(calibrate_frames(rgb)):
            n_frames += 1
            fid = f"{key}#{k}"
            frames.append(FrameInfo(fid, key, tuple(cal.frame), cal.y_side))
            for g, fig in figures.items():
                s = score_figure_on_frame(fid, fig, cal, ink)
                if s is not None:
                    scored[(fid, g)] = s
                    pairs.append({
                        "image": key, "frame": fid, "figure": g,
                        "S": s.pair.hit, "null": s.pair.null, "contrast": s.pair.contrast,
                        "inside": s.pair.inside, "eligible": s.pair.eligible,
                    })
        if n_frames == 0:
            skipped.append([key, "no_frame"])
    infos = [FigureInfo(g, fig.figure_name, "n/a", tuple(c.curve_id for c in fig.curves))
             for g, fig in figures.items()]
    records = decide_paper(paper_id, frames, infos, scored, rule="validation")
    return paper_id, {
        "records": records,
        "pairs": pairs,
        "n_frames": len(frames),
        "skipped": skipped,
        "images": [k for k, _ in images],
    }


def _local_images(paper: str, roots: dict[str, pathlib.Path]) -> list[tuple[str, str]]:
    """(key, absolute path); key is the registry's repo-relative path where there is one."""
    out = []
    for label, root in roots.items():
        d = root / paper
        if not d.is_dir():
            continue
        for f in sorted(d.iterdir()):
            if f.suffix.lower() in IMAGE_SUFFIXES:
                try:
                    key = str(f.relative_to(ROOT))
                except ValueError:
                    key = f"{label}/{paper}/{f.name}"
                out.append((key, str(f)))
    return out


def collect(args) -> None:
    from real_chart_bench.adapter.starrydata_figure_gt import load_figure_gt

    registry = json.loads(REGISTRY.read_text())
    papers = sorted({e["paper_id"] for e in registry}, key=int)
    roots = {
        "vp_images": ROOT / "data/verified_pairs/images",
        "vp_crops": ROOT / "data/verified_pairs/crops",
    }
    for spec in args.extra_root:
        label, _, path = spec.partition("=")
        roots[label] = pathlib.Path(path)
    gt = load_figure_gt(CURVES_CSV, papers)
    jobs = [(p, _local_images(p, roots), gt.get(p, {})) for p in papers]
    out, t0 = {}, time.time()
    with Pool(args.workers) as pool:
        for k, (pid, res) in enumerate(pool.imap_unordered(_work, jobs), 1):
            out[pid] = res
            print(f"[{time.time() - t0:5.0f}s] {k}/{len(jobs)} paper {pid}: "
                  f"{len(res['images'])} images, {res['n_frames']} frames", file=sys.stderr)
    args.raw.parent.mkdir(parents=True, exist_ok=True)
    args.raw.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%d"),
        "roots": {k: str(v) for k, v in roots.items()},
        "papers": out,
    }))


# ---------------------------------------------------------------- analysis


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def _labels():
    registry = json.loads(REGISTRY.read_text())
    verified, rejected, owners, scored = {}, defaultdict(set), defaultdict(set), set()
    owner_refs = defaultdict(set)
    for e in registry:
        key = (e["paper_id"], e["figure_id"])
        if not e["image_path"]:
            continue
        if e["status"] == "verified":
            verified[key] = e["image_path"]
            owners[e["image_path"]].add(e["figure_id"])
            owner_refs[e["image_path"]].add(normalise_reference(e["figure_reference"]))
            if not e.get("excluded_reason"):
                scored.add(key)
        elif e["status"] == "rejected" and e.get("rejection_category") == "pairing":
            # image- and gt_suspect-category rejections say nothing against the pairing
            rejected[key].add(e["image_path"])
    return (verified, {k: frozenset(v) for k, v in rejected.items()},
            {k: frozenset(v) for k, v in owners.items()}, scored,
            {k: frozenset(v) for k, v in owner_refs.items()})


def analyse(args) -> None:
    raw = json.loads(args.raw.read_text())["papers"]
    verified, rejected, owners, scored_keys, owner_refs = _labels()
    # only verified figures whose image is on disk and was analysed are evaluable
    analysed = {i for p in raw.values() for i in p["images"]}
    evaluable = {k for k, img in verified.items() if img in analysed}
    print(f"verified pairings with an image: {len(verified)}; image analysed locally: "
          f"{len(evaluable)}; of which scored in the benchmark: {len(evaluable & scored_keys)}; "
          f"papers: {len({k[0] for k in evaluable})}")
    skipped = Counter(why for p in raw.values() for _, why in p["skipped"])
    print(f"images analysed {len(analysed)}, skipped by the pipeline's filters: {dict(skipped)}")

    subsets = {"all verified": evaluable, "scored (n=91 set)": evaluable & scored_keys}
    rows = []  # (key, record, outcome)
    for pid, p in raw.items():
        for r in p["records"]:
            key = (pid, r["figure_id"])
            outcome = None
            if r["decision"].startswith("proposed"):
                outcome = label_assignment(
                    key, r["image"], verified=verified, rejected=rejected, owners=owners,
                    reference=r["figure_reference"], owner_references=owner_refs,
                )
            rows.append((key, r, outcome))

    held = heldout_papers({k[0] for k in verified})
    subsets["dev papers"] = {k for k in evaluable if k[0] not in held}
    subsets["held-out papers"] = {k for k in evaluable if k[0] in held}
    for name, keys in subsets.items():
        print(f"\n## Lanes vs human pairing — {name} (n={len(keys)})\n")
        by = defaultdict(Counter)
        for key, r, outcome in rows:
            if key in keys:
                lane = r["decision"].removeprefix("proposed_")
                by[lane][outcome or r.get("reason")] += 1
        print("| decision | n | correct | wrong | ambiguous | sibling | other |")
        print("|---|---|---|---|---|---|---|")
        for lane in ("high", "review", "unassigned"):
            c = by[lane]
            n = sum(c.values())
            other = {k: v for k, v in c.items()
                     if k not in (CORRECT, WRONG, AMBIGUOUS, SIBLING)}
            print(f"| {lane} | {n} | {c[CORRECT]} | {c[WRONG]} | {c[AMBIGUOUS]} | "
                  f"{c[SIBLING]} | {other or ''} |")

    # every proposal about a labelled situation (verified, rejected or stealing an owned image)
    items = [(key, r, o) for key, r, o in rows if o is not None]
    labelled = [LabelledAssignment(o, r["S"], r["M"], r["contrast"]) for _, r, o in items]
    n_ev = len(evaluable)
    print(f"\nproposals: {len(items)} (correct {sum(o == CORRECT for *_, o in items)}, "
          f"wrong {sum(o == WRONG for *_, o in items)}, "
          f"ambiguous {sum(o == AMBIGUOUS for *_, o in items)}, "
          f"sibling {sum(o == SIBLING for *_, o in items)}, "
          f"unlabelled {sum(o == UNLABELLED for *_, o in items)})")
    wrong = [(k, r) for k, r, o in items if o == WRONG]
    print("\n### Wrong proposals (figure, image, S, M, contrast, lane)\n")
    for k, r in sorted(wrong, key=lambda t: -t[1]["S"]):
        print(f"- {k[0]}/{k[1]} -> {r['image']} S={r['S']:.2f} M={r['M']:.2f} "
              f"contrast={r['contrast']:.2f} {r['decision']}")
    print("\n### Ambiguous / sibling proposals\n")
    for k, r, o in items:
        if o in (AMBIGUOUS, SIBLING):
            print(f"- {o}: {k[0]}/{k[1]} ({r['figure_reference']}) -> {r['image']} "
                  f"S={r['S']:.2f} M={r['M']:.2f} contrast={r['contrast']:.2f} {r['decision']}")

    def table(title, rules):
        print(f"\n### {title}\n")
        print("| S>= | M>= | contrast>= | adopted | correct | wrong | ambig | unlab | "
              "precision | pessimistic | err upper95 | recall(evaluable) |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in rules:
            s = r.stats
            print(f"| {r.s_min:.2f} | {r.m_min:.2f} | {r.c_min:.2f} | {s.adopted} | {s.correct} | "
                  f"{s.wrong} | {s.ambiguous} | {s.unlabelled} | {_pct(s.precision)} | "
                  f"{_pct(s.precision_pessimistic)} | {_pct(s.error_upper95)} | {_pct(s.recall)} |")

    grid = [round(0.5 + 0.05 * i, 2) for i in range(11)]
    one = lambda **kw: sweep_adoption_rules(labelled, n_verified=n_ev, **kw)  # noqa: E731
    table("S sweep (M, contrast unconstrained beyond eligibility)",
          one(s_grid=grid, m_grid=[0.0], c_grid=[0.2]))
    table("M sweep (S>=0.50)", one(s_grid=[0.5], m_grid=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9],
                                   c_grid=[0.2]))
    table("contrast sweep (S>=0.50)",
          one(s_grid=[0.5], m_grid=[0.0], c_grid=[0.2, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8]))
    table("current provisional 'high' lane", one(s_grid=[0.8], m_grid=[0.3], c_grid=[0.35]))

    full = sweep_adoption_rules(
        labelled, n_verified=n_ev,
        s_grid=[round(0.5 + 0.02 * i, 2) for i in range(26)],
        m_grid=[round(0.05 * i, 2) for i in range(0, 21)],
        c_grid=[round(0.2 + 0.05 * i, 2) for i in range(0, 17)],
    )
    for target in (0.99, 1.0):
        ok = [r for r in full if r.stats.precision_pessimistic is not None
              and r.stats.precision_pessimistic >= target - 1e-9 and r.stats.correct > 0]
        ok.sort(key=lambda r: (-r.stats.correct, r.s_min + r.m_min + r.c_min))
        table(f"best grid rules with pessimistic precision >= {target:.0%} "
              f"(top 8 by correct coverage)", ok[:8])
    ok = [r for r in full if r.stats.precision is not None
          and r.stats.precision >= 0.99 - 1e-9 and r.stats.correct > 0]
    ok.sort(key=lambda r: (-r.stats.correct, r.s_min + r.m_min + r.c_min))
    table("best grid rules with precision >= 99% when sibling digitizations are tolerated "
          "(top 8 by correct coverage)", ok[:8])
    print("\nexact 95% upper bounds on the error rate of a lane with 0 errors: "
          + ", ".join(f"n={n}: {_pct(clopper_pearson_upper(0, n))}"
                      for n in (20, 50, 91, 136, 300)))

    # §5-style pair level: every scored (frame, figure) pair, eligible or not
    print("\n## Pair level (design §5): best true pair per figure vs every wrong pair\n")
    ref_of = {(pid, r["figure_id"]): r["figure_reference"]
              for pid, p in raw.items() for r in p["records"]}
    best_true, neg, sib = {}, [], []
    for pid, p in raw.items():
        for q in p["pairs"]:
            key = (pid, q["figure"])
            o = label_assignment(key, q["image"], verified=verified, rejected=rejected,
                                 owners=owners, reference=ref_of.get(key, ""),
                                 owner_references=owner_refs)
            if o == CORRECT:
                if key not in best_true or q["S"] > best_true[key]["S"]:
                    best_true[key] = q
            elif o == WRONG:
                neg.append(q)
            elif o == SIBLING:
                sib.append(q)
    pos = list(best_true.values())
    print(f"- verified figures with any scored pair on their own image: {len(pos)}/{n_ev}; "
          f"wrong pairs (all, incl. ineligible): {len(neg)}; sibling pairs: {len(sib)}")
    print("\n| S>= | true pairs kept | wrong pairs passing | sibling pairs passing | "
          "pair precision (true/(true+wrong)) |")
    print("|---|---|---|---|---|")
    for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 1.0):
        tp = sum(q["S"] >= t - 1e-9 for q in pos)
        fp = sum(q["S"] >= t - 1e-9 for q in neg)
        sb = sum(q["S"] >= t - 1e-9 for q in sib)
        print(f"| {t:.2f} | {tp}/{len(pos)} | {fp}/{len(neg)} | {sb}/{len(sib)} | "
              f"{_pct(tp / (tp + fp) if tp + fp else None)} |")
    low = sorted(q["S"] for q in pos if not q["eligible"])
    print(f"\n- true pairs below the eligibility floor (hit<0.5 or contrast<0.2): {len(low)}, "
          f"their S: {[round(v, 2) for v in low]}")

    print("\n## Why verified figures were not correctly proposed\n")
    why = Counter()
    for key, r, o in rows:
        if key in evaluable and o != CORRECT:
            why[o or r.get("reason")] += 1
    print(dict(why))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--extra-root", action="append", default=[],
                   help="LABEL=DIR holding <paper_id>/<images>; repeatable")
    c.add_argument("--workers", type=int, default=8)
    c.add_argument("--raw", type=pathlib.Path, default=RAW)
    a = sub.add_parser("analyse")
    a.add_argument("--raw", type=pathlib.Path, default=RAW)
    args = ap.parse_args()
    collect(args) if args.cmd == "collect" else analyse(args)


if __name__ == "__main__":
    main()
