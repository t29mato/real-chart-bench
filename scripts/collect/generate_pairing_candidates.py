"""Pairing candidates for the CC BY papers with locally extracted images (design
pairing-automation.md §12, benchmark-architecture.md §7.89).

For each paper with extracted images: detect plot frames, calibrate their axes
(Tesseract, no LLM), project every Starrydata figure of the paper onto every
frame, and assign figures to frames jointly (Hungarian). Writes

    data/manifest/v0/pairing_candidates.json   one record per figure (committed)
    build/pairing_overlays.json                projected points for the review page (ignored)

No network. Two already-local image sources (design 12.6): data/raw/images
(re-fetched PDFs; pass --images-root to read the main checkout's copy from a
worktree) and ~/.cache/real-chart-bench/starrydata-work/images (starrydata_fetch).
Curves come from the local Starrydata CSV. Incremental: a paper that already has
records is never re-decided, so existing records stay byte-stable.

    python scripts/collect/generate_pairing_candidates.py --images-root /abs/data/raw/images
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import time
from multiprocessing import Pool

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from real_chart_bench.adapter.auto_axis_calibration import (  # noqa: E402
    calibrate_frames,
    load_rgb,
)
from real_chart_bench.adapter.pairing_scorer import ink_mask, score_figure_on_frame  # noqa: E402
from real_chart_bench.adapter.starrydata_figure_gt import load_figure_gt  # noqa: E402
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.usecase.pairing_candidates import (  # noqa: E402
    FigureInfo,
    FrameInfo,
    decide_paper,
    merge_candidate_records,
)
from real_chart_bench.usecase.real_image_gate import benchmark_paper_ids  # noqa: E402

WORK = pathlib.Path.home() / ".cache/real-chart-bench/starrydata-work"
CURVES_CSV = WORK / "ThermoelectricMaterials_curves.csv.gz"
SRC_RAW = "data_raw_refetch"
SRC_WORK = "starrydata_work"
MIN_IMAGE_PX = 200
MAX_DARK_SHARE = 0.4  # a photograph or micrograph


def _work(job):
    paper_id, image_dir, figures, figure_meta, names, rule = job
    frames, scored, overlays_src = [], {}, {}
    n_images = 0
    for name in names:
        rgb = load_rgb(image_dir / name)
        if rgb is None or min(rgb.shape[:2]) < MIN_IMAGE_PX:
            continue
        if (rgb.mean(2) < 140).mean() > MAX_DARK_SHARE:
            continue
        n_images += 1
        ink = None
        for k, cal in enumerate(calibrate_frames(rgb)):
            fid = f"{name}#{k}"
            frames.append(FrameInfo(fid, f"{paper_id}/{name}", tuple(cal.frame), cal.y_side))
            ink = ink_mask(rgb) if ink is None else ink
            for g, fig in figures.items():
                s = score_figure_on_frame(fid, fig, cal, ink)
                if s is not None:
                    scored[(fid, g)] = s
                    overlays_src[(fid, g)] = s.points_px
    infos = [FigureInfo(g, figure_meta[g]["figure_reference"], figure_meta[g]["split"],
                        tuple(c.curve_id for c in fig.curves)) for g, fig in figures.items()]
    records = decide_paper(paper_id, frames, infos, scored, rule=rule)
    overlays = {}
    for r in records:
        if r["decision"].startswith("proposed"):
            pts = overlays_src[(r["frame_id"], r["figure_id"])]
            overlays[r["candidate_id"]] = [[[round(x, 1), round(y, 1)] for x, y in c] for c in pts]
    return paper_id, records, overlays, {"frames": len(frames), "images": n_images}


def _sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=pathlib.Path, default=ROOT / "data/raw/images",
                    help=f"image source '{SRC_RAW}'")
    ap.add_argument("--work-root", type=pathlib.Path, default=WORK,
                    help=f"image source '{SRC_WORK}' (scripts/train/starrydata_fetch.py output)")
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "data/manifest/v0/pairing_candidates.json")
    ap.add_argument("--overlays", type=pathlib.Path, default=ROOT / "build/pairing_overlays.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--redecide", action="store_true",
                    help="re-decide every paper with the current rule and replace the file "
                         "(default: incremental, existing papers are never re-decided). Only "
                         "while no human verdict refers to the records.")
    args = ap.parse_args()

    prev = json.loads(args.out.read_text()) if args.out.exists() else {"candidates": []}
    old_papers = {r["paper_id"] for r in prev["candidates"]}
    if args.redecide:
        # Explicit re-decision of EVERY paper with the current rule (design
        # pairing-automation.md 12.7). Only safe while no human verdict refers
        # to these records; the old file is replaced, not merged.
        prev = {k: v for k, v in prev.items()
                if k in ("generated", "work_pdf_sha256")} | {"candidates": []}
    existing = prev["candidates"]
    done = {r["paper_id"] for r in existing}
    refetch = json.loads((ROOT / "data/manifest/v0/refetch_log.json").read_text())["papers"]
    raw_papers = {k: v for k, v in refetch.items() if v["n_images"] > 0}
    figs_meta: dict = {}
    for f in json.loads((ROOT / "data/manifest/v0/figures.json").read_text()):
        figs_meta.setdefault(f["paper_id"], {})[f["figure_id"]] = f
    cc_by = {p["paper_id"] for p in json.loads(
        (ROOT / "data/manifest/v0/papers.json").read_text()) if p["license_id"] == "cc-by"}
    benchmark = benchmark_paper_ids(load_registry(ROOT / "data/verified_pairs/registry.json"))
    work_dirs = sorted((d.name for d in (args.work_root / "images").iterdir() if d.is_dir()),
                       key=int)
    work_new = [p for p in work_dirs
                if p in cc_by and p not in benchmark and p not in raw_papers and p not in done]
    raw_new = sorted((set(raw_papers) - benchmark) - done, key=int)
    overlays = ({} if args.redecide or not args.overlays.exists()
                else json.loads(args.overlays.read_text()))
    # Overlays live in an ignored file; rebuild missing ones for papers already decided
    # (records are not touched; a changed decision is reported).
    redo_ids = sorted({r["paper_id"] for r in existing if r["decision"].startswith("proposed")
                       and r["candidate_id"] not in overlays}, key=int)
    gt = load_figure_gt(CURVES_CSV, sorted(set(raw_new) | set(redo_ids) | set(work_new), key=int))
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()
    rule = f"rule@{sha}"

    def job(p, image_dir, names):
        return (p, image_dir, {g: f for g, f in gt.get(p, {}).items() if g in figs_meta.get(p, {})},
                figs_meta.get(p, {}), names, rule)

    def raw_job(p):
        return job(p, args.images_root / p, sorted(i["name"] for i in raw_papers[p]["images"]))

    def work_job(p):
        d = args.work_root / "images" / p
        return job(p, d, sorted(n.name for n in d.iterdir()))

    batches = [(SRC_RAW, [raw_job(p) for p in raw_new]),
               (SRC_WORK, [work_job(p) for p in work_new]),
               ("redo", [raw_job(p) for p in redo_ids if p in raw_papers])]
    new_records: dict[str, list] = {SRC_RAW: [], SRC_WORK: []}
    per_paper: dict = {}
    t0 = time.time()
    with Pool(args.workers) as pl:
        for src, js in batches:
            for k, (pid, recs, ov, st) in enumerate(pl.imap_unordered(_work, js), 1):
                if src == "redo":
                    old = {r["candidate_id"]: r["decision"] for r in existing
                           if r["paper_id"] == pid}
                    drift = [r["candidate_id"] for r in recs
                             if old.get(r["candidate_id"]) != r["decision"]]
                    if drift:
                        print(f"WARNING overlay rebuild drifted for {drift}", file=sys.stderr)
                    overlays.update({c: v for c, v in ov.items() if c in old})
                    continue
                new_records[src] += recs
                overlays.update(ov)
                per_paper[pid] = st
                print(f"[{time.time() - t0:5.0f}s] {src} {k}/{len(js)} paper {pid}: {st}",
                      file=sys.stderr)

    shas = {(p, i["name"]): i["sha256"] for p, v in raw_papers.items() for i in v["images"]}
    pdf_sha = {p: _sha(args.work_root / "pdf" / f"{p}.pdf") for p in work_new
               if (args.work_root / "pdf" / f"{p}.pdf").exists()}
    for src, recs in new_records.items():
        for r in recs:
            r["licence_paper"] = "cc-by"
            r["licence_figure_verified"] = False
            if "image" in r:
                p, n = r["image"].split("/", 1)
                r["image_sha256"] = (shas[(p, n)] if src == SRC_RAW
                                     else _sha(args.work_root / "images" / p / n))
    merged = merge_candidate_records(existing, new_records[SRC_RAW], SRC_RAW, old_source=SRC_RAW)
    merged = merge_candidate_records(merged, new_records[SRC_WORK], SRC_WORK, old_source=SRC_RAW)
    lost = old_papers - {r["paper_id"] for r in merged}
    if args.redecide and lost:
        sys.exit(f"--redecide would drop papers {sorted(lost, key=int)}; nothing written")
    dec: dict = {}
    by_src: dict = {}
    for r in merged:
        dec[r["decision"]] = dec.get(r["decision"], 0) + 1
        by_src.setdefault(r["image_source"], set()).add(r["paper_id"])
    old_stats = prev.get("stats", {})
    out = {
        "description": (
            "Pairing candidates for CC BY papers with locally extracted images (design "
            "pairing-automation.md sections 12, 12.6). NOTHING here is adopted: every "
            "proposed_* record awaits human review. Thresholds are provisional "
            "(C7 experiment not run). decided_by is the rule commit that decided the paper."
        ),
        "rule": prev.get("rule", rule),
        "generated": prev.get("generated", time.strftime("%Y-%m-%d")),
        "updated": time.strftime("%Y-%m-%d"),
        "calibration": "Tesseract tick OCR + rule-based frames "
                       "(auto_axis_calibration.calibrate_frames, v3); no LLM",
        "ground_truth": "Starrydata hand digitization only",
        "image_sources": {
            SRC_RAW: "data/raw/images/<paper>/ (not in git): PDFs re-fetched by "
                     "scripts/collect/refetch_cc_by_pdfs.py; image_sha256 matches "
                     "data/manifest/v0/refetch_log.json",
            SRC_WORK: "~/.cache/real-chart-bench/starrydata-work/images/<paper>/ (not in git): "
                      "scripts/train/starrydata_fetch.py; PDF from the first working Unpaywall "
                      "location (URL not recorded), same PyMuPdfFigureExtractor, page renders "
                      "at 200 dpi, embedded images stored as raw .img bytes; image_sha256 is "
                      "of the stored file, PDF sha256 in work_pdf_sha256",
        },
        "work_pdf_sha256": {**prev.get("work_pdf_sha256", {}), **pdf_sha},
        "stats": {
            "papers": len({r["paper_id"] for r in merged}),
            "papers_by_source": {k: len(v) for k, v in sorted(by_src.items())},
            "papers_excluded_already_benchmark": sorted(set(raw_papers) & benchmark, key=int),
            "images_analysed": old_stats.get("images_analysed", 0)
            + sum(s["images"] for s in per_paper.values()),
            "frames_calibrated": old_stats.get("frames_calibrated", 0)
            + sum(s["frames"] for s in per_paper.values()),
            "figures": len(merged),
            "figures_public": sum(r["split"] == "public" for r in merged),
            "decisions": dec,
        },
        "candidates": merged,
    }
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    args.overlays.parent.mkdir(parents=True, exist_ok=True)
    args.overlays.write_text(json.dumps(overlays))
    print(json.dumps(out["stats"], indent=1))


if __name__ == "__main__":
    main()
