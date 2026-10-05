"""Steps 2-4 of the real-figure training data (docs/design/local-model.md,
"データ: 実図(Starrydata)"): pair Starrydata curves with plot frames found
in the pool's figure images, calibrate each frame's axes by local OCR
(Tesseract) of its tick labels, project the hand-digitized values to pixels,
and keep a figure only when the projected points land on the drawn markers.

No model reads the image: frames and ticks come from pixel rules
(domain/axis_frame.py), tick values from Tesseract, curve values from
Starrydata. The unit transform per axis (uV/K vs V/K, degC vs K, 1000/T,
printed log10) is chosen from a fixed list by frame containment and ink.

Outputs (outside the repo):
    <out>/images/<sid>_<figure_id>.png   the frame crop, labels included
    <out>/labels.jsonl                   common schema (validate_label)
    <out>/candidates.jsonl               every scored figure x frame pair
    <out>/stats.json                     yield at each step

Usage:
    python scripts/train/starrydata_build.py [--papers SID,SID] [--workers 16]
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import pathlib
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from real_chart_bench.adapter.tesseract_ocr import ocr_words  # noqa: E402
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.domain.axis_frame import (  # noqa: E402
    detect_axis_frames,
    detect_ticks,
    outward_tick_extent,
)
from real_chart_bench.domain.tick_calibration import (  # noqa: E402
    best_projection,
    candidate_transforms,
    fit_axis,
    frame_band,
    ink_radius,
    readings_for_axis,
    scored_contrast,
    without_frame_lines,
)
from real_chart_bench.domain.training_data import (  # noqa: E402
    assert_no_benchmark_leak,
    validate_label,
)
from real_chart_bench.usecase.real_image_gate import benchmark_paper_ids  # noqa: E402

CACHE = pathlib.Path.home() / ".cache/real-chart-bench"
WORK = CACHE / "starrydata-work"
OUT = CACHE / "train-data/starrydata"

DARK = 140  # frame/tick binarisation (axes are black)
INK_GRAY = 200  # marker ink: anything clearly darker than paper ...
INK_CHROMA = 60  # ... or clearly coloured
MIN_IMAGE_PX = 200
SERIES_MIN_HIT = 0.7  # per series: below this the sample is not drawn here


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def load_rgb(path: pathlib.Path) -> np.ndarray | None:
    try:
        im = Image.open(io.BytesIO(path.read_bytes()))
        im.load()
    except Exception:  # noqa: BLE001
        try:
            import pymupdf

            pix = pymupdf.Pixmap(path.read_bytes())
            if pix.alpha:
                pix = pymupdf.Pixmap(pix, 0)
            if pix.colorspace is None or pix.colorspace.name != "DeviceRGB":
                pix = pymupdf.Pixmap(pymupdf.csRGB, pix)
            return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[
                :, :, :3
            ].copy()
        except Exception:  # noqa: BLE001
            return None
    if im.mode == "CMYK":
        # many publisher JPEGs are Adobe CMYK with inverted channels
        arr = np.asarray(im.convert("RGB"))
        if arr.mean() < 80:
            arr = 255 - arr
        return arr
    return np.asarray(im.convert("RGB"))


def calibrate_frame(gray: np.ndarray, dark: np.ndarray, frame) -> dict | None:
    x0, y0, x1, y1 = frame
    fw, fh = x1 - x0, y1 - y0
    h, w = gray.shape
    xt, yt = detect_ticks(dark, frame)
    below, left = outward_tick_extent(dark, frame)
    # x strip: under the axis line and its outward ticks
    sx0, sx1 = max(0, int(x0 - 0.1 * fw)), min(w, int(x1 + 0.1 * fw))
    sy0, sy1 = min(h, int(y1 + 2 + below + 1)), min(h, int(y1 + 0.2 * fh + 14 + below))
    # y strip: left of the axis line and its outward ticks
    ty0, ty1 = max(0, int(y0 - 0.1 * fh)), min(h, int(y1 + 0.1 * fh))
    tx0, tx1 = max(0, int(x0 - 0.4 * fw)), max(0, int(x0 - 2 - left))

    def read(axis, strip, ox, oy, ticks, direction):
        """Block mode (psm 6) reads a label column best; sparse mode (psm
        11) rescues scattered labels. Keep whichever fits more ticks."""
        best, best_n = (None, [], 0), -1
        for psm in (6, 11):
            ws = [(t, a + ox, b + oy, c + ox, d + oy, conf)
                  for t, a, b, c, d, conf in ocr_words(strip, psm=psm)]
            rd = readings_for_axis(ws, frame, axis, ticks=ticks)
            fit = fit_axis(rd, direction=direction)
            n = len(fit.ticks) if fit else 0
            if n > best_n:
                best, best_n = (fit, ws, len(rd)), n
            if n >= 4:
                break
        return best

    xf, xw, n_xr = read("x", gray[sy0:sy1, sx0:sx1], sx0, sy0, xt, +1)
    yf, yw, n_yr = read("y", gray[ty0:ty1, tx0:tx1], tx0, ty0, yt, -1)
    words = [w[:5] for w in xw + yw]
    return {"x_fit": xf, "y_fit": yf, "words": words, "n_xr": n_xr, "n_yr": n_yr}


def process_image(args):
    sid, img_path = args
    rgb = load_rgb(img_path)
    if rgb is None or min(rgb.shape[:2]) < MIN_IMAGE_PX:
        return sid, img_path.name, None, []
    gray = (rgb.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)).astype(np.uint8)
    dark = gray < DARK
    if dark.mean() > 0.4:  # a photograph or micrograph
        return sid, img_path.name, rgb.shape[:2], []
    frames = detect_axis_frames(dark)
    out = []
    for f in frames:
        cal = calibrate_frame(gray, dark, f)
        out.append((f, cal))
    return sid, img_path.name, rgb.shape[:2], out


def load_curves(pool: set[str]) -> dict[str, dict]:
    """SID -> figure_id -> list of curve rows with parsed x/y."""
    by: dict = defaultdict(lambda: defaultdict(list))
    with gzip.open(WORK / "ThermoelectricMaterials_curves.csv.gz", "rt",
                   encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            if r["SID"] not in pool:
                continue
            try:
                xs, ys = json.loads(r["x"]), json.loads(r["y"])
            except json.JSONDecodeError:
                continue
            if not xs or len(xs) != len(ys):
                continue
            r["xs"], r["ys"] = [float(v) for v in xs], [float(v) for v in ys]
            by[r["SID"]][r["figure_id"]].append(r)
    return by


def color_at(rgb: np.ndarray, pts) -> str | None:
    """Median colour of the darkest/most saturated pixel near each point."""
    h, w, _ = rgb.shape
    cols = []
    for x, y in pts:
        cx, cy = int(round(x)), int(round(y))
        win = rgb[max(0, cy - 2) : cy + 3, max(0, cx - 2) : cx + 3].reshape(-1, 3).astype(int)
        if win.size == 0:
            continue
        score = win.max(1) - win.min(1) + (255 - win.mean(1))
        cols.append(win[int(np.argmax(score))])
    if not cols:
        return None
    c = np.median(np.array(cols), axis=0).astype(int)
    return "#{:02x}{:02x}{:02x}".format(*c)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--papers", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", type=pathlib.Path, default=OUT)
    ap.add_argument("--min-hit", type=float, default=0.7)
    ap.add_argument("--min-contrast", type=float, default=0.35)
    args = ap.parse_args()
    out = args.out
    (out / "images").mkdir(parents=True, exist_ok=True)

    excluded = benchmark_paper_ids(load_registry(ROOT / "data/verified_pairs/registry.json"))
    status = json.loads((WORK / "fetch_status.json").read_text())
    papers = json.loads((ROOT / "data/manifest/v0/papers.json").read_text())
    doi = {p["paper_id"]: p["doi"] for p in papers}
    pool = {p["paper_id"] for p in papers if p["license_id"] == "cc-by"} - excluded
    with_images = sorted(s for s in pool if status.get(s, {}).get("status") == "ok")
    if args.papers:
        with_images = [s for s in with_images if s in set(args.papers.split(","))]
    curves = load_curves(set(with_images))
    stats: dict = {"pool_papers": len(pool), "papers_with_images": len(with_images),
                   "pool_figures_with_images": sum(len(curves[s]) for s in with_images),
                   "pool_curves_with_images": sum(len(c) for s in with_images
                                                  for c in curves[s].values())}

    jobs = [(s, p) for s in with_images for p in sorted((WORK / "images" / s).iterdir())]
    stats["images"] = len(jobs)
    log(f"{len(with_images)} papers, {len(jobs)} images")
    frames: dict = defaultdict(list)  # sid -> [(img_name, frame, cal)]
    shapes = {}
    with Pool(args.workers) as pool_:
        for k, (sid, name, shape, res) in enumerate(pool_.imap_unordered(process_image, jobs), 1):
            shapes[(sid, name)] = shape
            for f, cal in res:
                frames[sid].append((name, f, cal))
            if k % 200 == 0:
                log(f"images {k}/{len(jobs)}")
    all_frames = [x for v in frames.values() for x in v]
    stats["frames_detected"] = len(all_frames)
    stats["frames_x_calibrated"] = sum(1 for _, _, c in all_frames if c["x_fit"])
    stats["frames_y_calibrated"] = sum(1 for _, _, c in all_frames if c["y_fit"])
    stats["frames_calibrated"] = sum(1 for _, _, c in all_frames if c["x_fit"] and c["y_fit"])

    cand_rows, labels = [], []
    rgb_cache: dict = {}
    n_fig_tried = 0
    for sid in with_images:
        cal_frames = [(n, f, c) for n, f, c in frames[sid] if c["x_fit"] and c["y_fit"]]
        scored = []
        for fid, rows in curves[sid].items():
            # one axis pair per figure: the most common (prop_x, prop_y); a
            # second y quantity is usually a twin right axis we do not read
            key = max({(r["prop_x"], r["unit_x"], r["prop_y"], r["unit_y"]) for r in rows},
                      key=lambda k: sum(1 for r in rows if
                                        (r["prop_x"], r["unit_x"], r["prop_y"], r["unit_y"]) == k))
            rs = [r for r in rows if (r["prop_x"], r["unit_x"], r["prop_y"], r["unit_y"]) == key]
            n_fig_tried += 1
            for name, f, c in cal_frames:
                k = (sid, name)
                if k not in rgb_cache:
                    rgb_cache.clear()
                    rgb_cache[k] = load_rgb(WORK / "images" / sid / name)
                rgb = rgb_cache[k]
                gray = rgb.astype(np.float32).mean(2)
                ink = (gray < INK_GRAY) | ((rgb.max(2).astype(int) - rgb.min(2)) > INK_CHROMA)
                pr = best_projection(
                    [(r["xs"], r["ys"]) for r in rs], c["x_fit"], c["y_fit"], f, ink,
                    candidate_transforms(key[0], key[1]), candidate_transforms(key[2], key[3]),
                )
                if pr is None:
                    continue
                scored.append((pr.contrast, fid, name, f, c, pr, rs, key))
                cand_rows.append({
                    "paper_id": sid, "figure_id": fid, "image": name,
                    "frame": [round(v, 1) for v in f],
                    "x_transform": pr.x_transform.name, "y_transform": pr.y_transform.name,
                    "hit": round(pr.hit, 3), "null": round(pr.null, 3),
                    "inside": round(pr.inside, 3),
                })
        # one frame per figure, one figure per frame, best contrast first
        scored.sort(key=lambda s: -s[0])
        used_fig, used_frame = set(), set()
        for contrast, fid, name, f, c, pr, rs, key in scored:
            fkey = (name, tuple(round(v) for v in f))
            if fid in used_fig or fkey in used_frame:
                continue
            others = [s[0] for s in scored if s[1] == fid and s[2:4] != (name, f)]
            rivals = [s[0] for s in scored if s[1] != fid and (s[2], tuple(round(v) for v in s[3]))
                      == fkey]
            margin = contrast - max(others + rivals + [0.0])
            used_fig.add(fid)
            used_frame.add(fkey)
            lab = make_label(sid, doi.get(sid), fid, name, f, c, pr, rs, key, margin, out,
                             args.min_hit, args.min_contrast)
            if lab is not None:
                labels.append(lab)
    stats["figures_tried"] = n_fig_tried
    stats["figure_frame_pairs_in_frame"] = len(cand_rows)
    stats["figures_paired"] = len(labels)
    accepted = [lab for lab in labels if lab["extra"]["gate"] == "pass"]
    stats["figures_pass_gate"] = len(accepted)
    stats["curves_pass_gate"] = sum(len(lab["series"]) for lab in accepted)
    stats["points_pass_gate"] = sum(len(s["points_px"]) for lab in accepted
                                    for s in lab["series"])
    stats["papers_pass_gate"] = len({lab["paper_id"] for lab in accepted})

    assert_no_benchmark_leak(labels, excluded)
    bad = [(lab["image"], e) for lab in labels for e in validate_label(lab)]
    if bad:
        raise SystemExit(f"invalid labels: {bad[:5]}")
    with open(out / "labels.jsonl", "w") as fh:
        for lab in accepted:
            fh.write(json.dumps(lab) + "\n")
    with open(out / "rejected.jsonl", "w") as fh:
        for lab in labels:
            if lab["extra"]["gate"] != "pass":
                fh.write(json.dumps(lab) + "\n")
    with open(out / "candidates.jsonl", "w") as fh:
        for r in cand_rows:
            fh.write(json.dumps(r) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    write_manifest(stats, accepted, args)
    log(json.dumps(stats))


def write_manifest(stats: dict, accepted: list, args) -> None:
    """data/train_manifest/starrydata.json: what went in, from where, under
    which licence and gate -- metadata only, the images stay in ~/.cache."""
    by_paper: dict = defaultdict(list)
    for lab in accepted:
        by_paper[lab["paper_id"]].append(lab)
    manifest = {
        "name": "starrydata-real-figures",
        "design": "docs/design/local-model.md#データ-実図starrydata",
        "output_dir": "~/.cache/real-chart-bench/train-data/starrydata",
        "label_source": "Starrydata hand digitization (ThermoelectricMaterials_curves.csv.gz, "
                        "github.com/starrydata/starrydata_datasets release 'latest')",
        "calibration": "Tesseract 5.3.4 (Apache-2.0) tick OCR + rule-based frame/tick detection;"
                       " no LLM output anywhere in a label",
        "figure_license_policy": "CC-BY papers only (OpenAlex at v0 collection, re-checked "
                                 "against Unpaywall 2026-10-06); benchmark papers excluded "
                                 "(registry.json, any status)",
        "gate": {"min_hit": args.min_hit, "min_contrast": args.min_contrast,
                 "per_series_min_hit": SERIES_MIN_HIT},
        "stats": stats,
        "papers": [
            {"paper_id": sid, "doi": labs[0]["extra"]["doi"], "license": "CC-BY",
             "figures": [{"figure_id": lab["extra"]["figure_id"],
                          "figure_name": lab["extra"]["figure_name"],
                          "image": lab["image"], "n_series": len(lab["series"]),
                          "n_points": sum(len(s["points_px"]) for s in lab["series"]),
                          "contrast": lab["extra"]["confidence"]["contrast"]}
                         for lab in labs]}
            for sid, labs in sorted(by_paper.items(), key=lambda kv: int(kv[0]))
        ],
    }
    path = ROOT / "data/train_manifest/starrydata.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")


def make_label(sid, doi, fid, name, f, c, pr, rows, key, margin, out, min_hit, min_contrast):
    rgb = load_rgb(WORK / "images" / sid / name)
    h, w = rgb.shape[:2]
    x0, y0, x1, y1 = f
    fw, fh = x1 - x0, y1 - y0
    tick_boxes = c["words"]
    cx0 = max(0, int(min([x0 - 0.35 * fw] + [b[1] - 4 for b in tick_boxes if b[3] <= x0 + 3])))
    cx0 = max(cx0, int(x0 - 0.45 * fw))
    cy0 = max(0, int(y0 - 0.08 * fh - 10))
    cx1 = min(w, int(x1 + 0.08 * fw + 10))
    cy1 = min(h, int(y1 + 0.3 * fh + 20))
    crop = rgb[cy0:cy1, cx0:cx1]
    ink = (crop.astype(np.float32).mean(2) < INK_GRAY) | (
        (crop.max(2).astype(int) - crop.min(2)) > INK_CHROMA)

    def sh(p):
        return [round(p[0] - cx0, 2), round(p[1] - cy0, 2)]

    series, per_curve = [], []
    for r, px, pv in zip(rows, pr.points_px, pr.points_value, strict=True):
        keep = [(sh(a), [float(f"{b[0]:.6g}"), float(f"{b[1]:.6g}")])
                for a, b in zip(px, pv, strict=True)
                if 0 <= a[0] - cx0 <= crop.shape[1] and 0 <= a[1] - cy0 <= crop.shape[0]]
        if not keep:
            continue
        cpx = [k[0] for k in keep]
        rad = ink_radius(f)
        fc = (x0 - cx0, y0 - cy0, x1 - cx0, y1 - cy0)
        ink_c = without_frame_lines(ink, fc, frame_band(f, rad))
        hit, null = scored_contrast(cpx, ink_c, fc, rad)
        per_curve.append({"composition": r["composition"], "hit": round(hit, 3),
                          "null": round(null, 3), "n": len(keep)})
        if hit < SERIES_MIN_HIT:  # this sample's curve is not drawn here (twin axis, other panel)
            continue
        series.append({
            "label": r["composition"] or None, "marker": None, "filled": None,
            "color": color_at(crop, cpx),
            "points_px": cpx, "points_value": [k[1] for k in keep],
        })
    if not series:
        return None
    gate = "pass" if (pr.hit >= min_hit and pr.contrast >= min_contrast) else "fail"
    img_name = f"{sid}_{fid}.png"
    sub = "images" if gate == "pass" else "rejected"  # rejected crops: for threshold review
    (out / sub).mkdir(exist_ok=True)
    Image.fromarray(np.ascontiguousarray(crop)).save(out / sub / img_name)

    def axis(fit, transform):
        return {
            "scale": fit.scale,
            "ticks": [{"px": round(p - (cx0 if fit is c["x_fit"] else cy0), 2), "value": v}
                      for p, v in fit.ticks],
            "transform_from_starrydata": transform.name,
        }

    return {
        "image": f"{sub}/{img_name}", "width": int(crop.shape[1]), "height": int(crop.shape[0]),
        "source": "starrydata", "license": "CC-BY", "paper_id": sid,
        "axes": {"x": axis(c["x_fit"], pr.x_transform), "y": axis(c["y_fit"], pr.y_transform)},
        "plot_bbox": [round(x0 - cx0, 1), round(y0 - cy0, 1), round(x1 - cx0, 1),
                      round(y1 - cy0, 1)],
        "series": series,
        "extra": {
            "doi": doi, "figure_id": fid, "figure_name": rows[0]["figure_name"],
            "source_image": f"{sid}/{name}", "crop_in_source": [cx0, cy0, cx1, cy1],
            "prop_x": key[0], "unit_x": key[1], "prop_y": key[2], "unit_y": key[3],
            "series_complete": None,
            "gate": gate,
            "confidence": {"hit": round(pr.hit, 3), "null": round(pr.null, 3),
                           "contrast": round(pr.contrast, 3), "margin": round(margin, 3),
                           "inside": round(pr.inside, 3),
                           "x_tick_residual_px": round(c["x_fit"].residual_px, 2),
                           "y_tick_residual_px": round(c["y_fit"].residual_px, 2),
                           "n_x_ticks": len(c["x_fit"].ticks), "n_y_ticks": len(c["y_fit"].ticks)},
            "per_curve": per_curve,
            "calibration": "tesseract-5.3.4 tick OCR + domain/tick_calibration.fit_axis",
        },
    }


if __name__ == "__main__":
    main()
