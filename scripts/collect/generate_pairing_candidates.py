"""Pairing candidates for the re-fetched CC BY papers (design
pairing-automation.md §12, benchmark-architecture.md §7.88).

For each paper with extracted images: detect plot frames, calibrate their axes
(Tesseract, no LLM), project every Starrydata figure of the paper onto every
frame, and assign figures to frames jointly (Hungarian). Writes

    data/manifest/v0/pairing_candidates.json   one record per figure (committed)
    build/pairing_overlays.json                projected points for the review page (ignored)

No network: images come from data/raw/images (pass --images-root to read the
main checkout's copy from a worktree), curves from the local Starrydata CSV.

    python scripts/collect/generate_pairing_candidates.py --images-root /abs/data/raw/images
"""

from __future__ import annotations

import argparse
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
)
from real_chart_bench.usecase.real_image_gate import benchmark_paper_ids  # noqa: E402

CURVES_CSV = (
    pathlib.Path.home()
    / ".cache/real-chart-bench/starrydata-work/ThermoelectricMaterials_curves.csv.gz"
)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-root", type=pathlib.Path, default=ROOT / "data/raw/images")
    ap.add_argument("--out", type=pathlib.Path,
                    default=ROOT / "data/manifest/v0/pairing_candidates.json")
    ap.add_argument("--overlays", type=pathlib.Path, default=ROOT / "build/pairing_overlays.json")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    refetch = json.loads((ROOT / "data/manifest/v0/refetch_log.json").read_text())["papers"]
    papers = {k: v for k, v in refetch.items() if v["n_images"] > 0}
    figs_meta: dict = {}
    for f in json.loads((ROOT / "data/manifest/v0/figures.json").read_text()):
        figs_meta.setdefault(f["paper_id"], {})[f["figure_id"]] = f
    benchmark = benchmark_paper_ids(load_registry(ROOT / "data/verified_pairs/registry.json"))
    pool = sorted(set(papers) - benchmark, key=int)
    gt = load_figure_gt(CURVES_CSV, pool)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True,
                         text=True, check=True).stdout.strip()
    rule = f"rule@{sha}"
    jobs = [(p, args.images_root / p,
             {g: f for g, f in gt.get(p, {}).items() if g in figs_meta.get(p, {})},
             figs_meta.get(p, {}),
             sorted(i["name"] for i in papers[p]["images"]), rule) for p in pool]
    records, overlays, per_paper = [], {}, {}
    t0 = time.time()
    with Pool(args.workers) as pl:
        for k, (pid, recs, ov, st) in enumerate(pl.imap_unordered(_work, jobs), 1):
            records += recs
            overlays.update(ov)
            per_paper[pid] = st
            print(f"[{time.time() - t0:5.0f}s] {k}/{len(jobs)} paper {pid}: {st}", file=sys.stderr)
    shas = {(p, i["name"]): i["sha256"] for p, v in papers.items() for i in v["images"]}
    for r in records:
        r["licence_paper"] = "cc-by"
        r["licence_figure_verified"] = False
        if "image" in r:
            p, n = r["image"].split("/", 1)
            r["image_sha256"] = shas[(p, n)]
    records.sort(key=lambda r: (int(r["paper_id"]), int(r["figure_id"])))
    dec = {}
    for r in records:
        dec[r["decision"]] = dec.get(r["decision"], 0) + 1
    out = {
        "description": (
            "Pairing candidates for the re-fetched CC BY papers (design "
            "pairing-automation.md section 12). NOTHING here is adopted: every "
            "proposed_* record awaits human review. Thresholds are provisional "
            "(C7 experiment not run)."
        ),
        "rule": rule,
        "generated": time.strftime("%Y-%m-%d"),
        "calibration": "Tesseract tick OCR + rule-based frames "
                       "(auto_axis_calibration.calibrate_frames, v3); no LLM",
        "ground_truth": "Starrydata hand digitization only",
        "stats": {
            "papers": len(pool),
            "papers_excluded_already_benchmark": sorted(set(papers) & benchmark, key=int),
            "images_analysed": sum(s["images"] for s in per_paper.values()),
            "frames_calibrated": sum(s["frames"] for s in per_paper.values()),
            "figures": len(records),
            "figures_public": sum(r["split"] == "public" for r in records),
            "decisions": dec,
        },
        "candidates": records,
    }
    args.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    args.overlays.parent.mkdir(parents=True, exist_ok=True)
    args.overlays.write_text(json.dumps(overlays))
    print(json.dumps(out["stats"], indent=1))


if __name__ == "__main__":
    main()
