"""Phase 3 v0 full-scale collection (design §7.9/§7.11, 司令塔承認 2026-08-16).

Runs the real collection pipeline (same adapters/usecases as the Phase 2/3
pilots — no mocks) against the *entire* ThermoelectricMaterials corpus:

1. Classify every paper's license via OpenAlex (batched, cached to disk so
   a re-run doesn't re-hit the API for papers already classified).
2. Build the ground-truth manifest for every REDISTRIBUTABLE paper
   (independent of PDF/image availability — see design §7.11 point 3).
3. For REDISTRIBUTABLE papers with a pdf_url, fetch the PDF and extract
   candidate figure images (rate-limited: politeness delay between
   requests, single-threaded — no concurrent hammering of publisher
   servers, matching the deep-digitizer pilot's approach).

Progress is logged to stderr every --log-every papers so a background run
can be tailed. Two-tier output (design §7.11):
  - data/manifest/v0/{papers,figures,curves}.json — ground truth, full
    REDISTRIBUTABLE population, committed to git (CC BY 4.0, metadata only)
  - data/raw/images/<SID>/ — extracted candidate images, gitignored,
    only for papers where PDF fetch succeeded

Usage:
    python scripts/collect/collect_v0_dataset.py --held-out-ratio 0.2
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import pathlib
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.adapter.figure_extraction import PyMuPdfFigureExtractor  # noqa: E402
from real_chart_bench.adapter.pdf_fetch import HttpPdfFetchAdapter  # noqa: E402
from real_chart_bench.adapter.starrydata_csv import parse_curve_row  # noqa: E402
from real_chart_bench.domain.collection_records import PaperRecord  # noqa: E402
from real_chart_bench.domain.licensing import LicenseStatus, classify_license  # noqa: E402
from real_chart_bench.usecase.build_ground_truth_manifest import (  # noqa: E402
    build_ground_truth_for_paper,
)
from real_chart_bench.usecase.pdf_fetch import PdfFetchStatus  # noqa: E402

RELEASE_BASE = "https://github.com/starrydata/starrydata_datasets/releases/download/latest"
USER_AGENT = "real-chart-bench-collector/0.1 (mailto:tomoya.matou@gmail.com)"
OPENALEX_BATCH_SIZE = 40
OPENALEX_DELAY_S = 0.3
PDF_FETCH_DELAY_S = 0.5  # matches deep-digitizer pilot's politeness interval
PDF_FETCH_TIMEOUT_S = 20


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _download(url: str, dest: pathlib.Path) -> None:
    if dest.exists():
        return
    log(f"downloading {url} -> {dest}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        dest.write_bytes(resp.read())


def _load_cache(path: pathlib.Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {}


def _save_cache(path: pathlib.Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def classify_all_papers(dois: list[str], cache_path: pathlib.Path) -> dict[str, dict]:
    cache = _load_cache(cache_path)
    todo = [d for d in dois if d not in cache]
    log(f"OpenAlex classification: {len(cache)} cached, {len(todo)} to fetch")

    for i in range(0, len(todo), OPENALEX_BATCH_SIZE):
        batch = todo[i : i + OPENALEX_BATCH_SIZE]
        params = urllib.parse.urlencode(
            {
                "filter": "doi:" + "|".join(batch),
                "select": "id,doi,open_access,primary_location,best_oa_location",
                "per-page": len(batch),
                "mailto": "tomoya.matou@gmail.com",
            }
        )
        url = f"https://api.openalex.org/works?{params}"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
                data = json.loads(resp.read())
            for work in data.get("results", []):
                # NOTE: OpenAlex normalizes DOI casing in its response, which
                # can differ from the source CSV's casing for the same DOI.
                # Always lowercase DOI keys everywhere (here and in
                # papers_by_doi) so lookups can't silently miss due to case
                # (root cause of the KeyError crash at paper 401/603).
                doi = (work.get("doi") or "").removeprefix("https://doi.org/").lower()
                primary = work.get("primary_location") or {}
                best_oa = work.get("best_oa_location") or {}
                cache[doi] = {
                    "is_oa": (work.get("open_access") or {}).get("is_oa"),
                    "license": primary.get("license"),
                    "pdf_url": best_oa.get("pdf_url") or primary.get("pdf_url"),
                }
            # DOIs OpenAlex didn't return anything for: mark as unresolved so
            # we don't re-query them forever on re-runs.
            for doi in batch:
                cache.setdefault(doi.lower(), {"is_oa": None, "license": None, "pdf_url": None})
        except urllib.error.URLError as e:
            log(f"  batch at {i} failed: {e} (will retry on next run)")
        if (i // OPENALEX_BATCH_SIZE) % 10 == 0:
            log(f"  classified {min(i + OPENALEX_BATCH_SIZE, len(todo))}/{len(todo)}")
            _save_cache(cache_path, cache)
        time.sleep(OPENALEX_DELAY_S)

    _save_cache(cache_path, cache)
    return cache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("data"))
    parser.add_argument("--held-out-ratio", type=float, default=0.2)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--skip-images", action="store_true", help="ground truth manifest only")
    args = parser.parse_args()

    raw_dir = args.out_dir / "raw"
    cache_dir = args.out_dir / "cache"
    manifest_dir = args.out_dir / "manifest" / "v0"
    for d in (raw_dir, cache_dir, manifest_dir, raw_dir / "images"):
        d.mkdir(parents=True, exist_ok=True)

    papers_gz = cache_dir / "ThermoelectricMaterials_papers.csv.gz"
    curves_gz = cache_dir / "ThermoelectricMaterials_curves.csv.gz"
    _download(f"{RELEASE_BASE}/ThermoelectricMaterials_papers.csv.gz", papers_gz)
    _download(f"{RELEASE_BASE}/ThermoelectricMaterials_curves.csv.gz", curves_gz)

    with gzip.open(papers_gz, "rt", newline="", encoding="utf-8") as f:
        papers_by_doi = {r["DOI"].lower(): r for r in csv.DictReader(f)}
    log(f"loaded {len(papers_by_doi)} papers from Starrydata")

    license_cache = classify_all_papers(list(papers_by_doi), cache_dir / "openalex_license.json")

    redistributable = [
        (doi, info)
        for doi, info in license_cache.items()
        if classify_license(info.get("license"), is_oa=info.get("is_oa"))
        is LicenseStatus.REDISTRIBUTABLE
    ]
    log(f"REDISTRIBUTABLE papers: {len(redistributable)} / {len(license_cache)}")

    log("loading curves.csv (grouping by SID)...")
    curves_by_sid: dict[str, list] = {}
    with gzip.open(curves_gz, "rt", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            curves_by_sid.setdefault(row["SID"], []).append(row)
    log(f"curves loaded for {len(curves_by_sid)} distinct papers")

    pdf_fetcher = HttpPdfFetchAdapter()
    extractor = PyMuPdfFigureExtractor()

    # Resume support: if a previous run crashed partway through, don't
    # re-fetch PDFs we already have (wasteful and impolite to re-hammer
    # publisher servers for papers already processed).
    all_papers = json.loads((manifest_dir / "papers.json").read_text()) \
        if (manifest_dir / "papers.json").exists() else []
    all_figures = json.loads((manifest_dir / "figures.json").read_text()) \
        if (manifest_dir / "figures.json").exists() else []
    all_curves = json.loads((manifest_dir / "curves.json").read_text()) \
        if (manifest_dir / "curves.json").exists() else []
    already_done = {p["paper_id"] for p in all_papers}
    if already_done:
        log(f"resuming: {len(already_done)} papers already processed in a prior run, skipping them")

    fetch_status_counts: dict[str, int] = {}
    for p in all_papers:
        status = p.get("pdf_status")
        if status:
            fetch_status_counts[status] = fetch_status_counts.get(status, 0) + 1
    n_images_total = sum(p.get("n_extracted_images") or 0 for p in all_papers)

    todo = [
        (doi, info) for doi, info in redistributable
        if papers_by_doi.get(doi, {}).get("SID") not in already_done
    ]
    log(f"{len(todo)} papers left to process ({len(already_done)} already done)")

    for idx, (doi, info) in enumerate(todo, start=1):
        paper_row = papers_by_doi.get(doi)
        if paper_row is None:
            log(f"  DOI {doi!r} classified by OpenAlex but not found in Starrydata CSV, skipping")
            continue
        sid = paper_row["SID"]
        paper = PaperRecord(
            paper_id=sid, doi=doi, title=paper_row.get("title", ""),
            license_status=LicenseStatus.REDISTRIBUTABLE, license_id=info.get("license") or "cc-by",
        )

        raw_rows = curves_by_sid.get(sid, [])
        try:
            parsed = [parse_curve_row(r) for r in raw_rows]
            figures, curves = build_ground_truth_for_paper(
                paper, parsed, held_out_ratio=args.held_out_ratio
            )
        except ValueError as e:
            log(f"  SID {sid}: ground truth build failed: {e}")
            figures, curves = (), ()

        n_images = None
        pdf_status = None
        if not args.skip_images and info.get("pdf_url"):
            fetch_result = pdf_fetcher.fetch(info["pdf_url"])
            pdf_status = fetch_result.status.value
            fetch_status_counts[pdf_status] = fetch_status_counts.get(pdf_status, 0) + 1
            if fetch_result.status is PdfFetchStatus.OK and fetch_result.content:
                try:
                    images = extractor.extract(fetch_result.content)
                    n_images = len(images)
                    n_images_total += n_images
                    img_dir = raw_dir / "images" / sid
                    img_dir.mkdir(parents=True, exist_ok=True)
                    for i, img in enumerate(images):
                        ext = "png" if img.source.value == "page_render" else "jpg"
                        name = f"p{img.page_number:02d}_{img.source.value}_{i}.{ext}"
                        (img_dir / name).write_bytes(img.image_bytes)
                except Exception as e:  # noqa: BLE001
                    log(f"  SID {sid}: extraction failed: {e}")
            time.sleep(PDF_FETCH_DELAY_S)

        all_papers.append(
            {
                "paper_id": sid, "doi": doi, "license_id": paper.license_id,
                "n_figures": len(figures), "n_curves": len(curves), "n_extracted_images": n_images,
                "pdf_status": pdf_status,
            }
        )
        all_figures.extend(
            {"figure_id": f.figure_id, "paper_id": f.paper_id,
             "figure_reference": f.figure_reference, "split": f.split.value}
            for f in figures
        )
        all_curves.extend(
            {"curve_id": c.curve_id, "figure_id": c.figure_id, "series_label": c.series_label,
             "n_points": len(c.x_values), "license": c.license}
            for c in curves
        )

        if idx % args.log_every == 0 or idx == len(todo):
            log(
                f"progress {idx}/{len(todo)} remaining ({len(all_papers)} total done) | "
                f"figures={len(all_figures)} curves={len(all_curves)} | "
                f"pdf_status={fetch_status_counts}"
            )
            (manifest_dir / "papers.json").write_text(json.dumps(all_papers, indent=2))
            (manifest_dir / "figures.json").write_text(json.dumps(all_figures, indent=2))
            (manifest_dir / "curves.json").write_text(json.dumps(all_curves, indent=2))

    summary = {
        "redistributable_papers": len(redistributable),
        "papers_with_ground_truth": len({f["paper_id"] for f in all_figures}),
        "n_figure_records": len(all_figures),
        "n_ground_truth_curves": len(all_curves),
        "pdf_fetch_status_counts": fetch_status_counts,
        "pdf_fetch_ok_rate": (
            fetch_status_counts.get("ok", 0) / max(1, sum(fetch_status_counts.values()))
        ),
        "n_papers_with_images": sum(1 for p in all_papers if p["n_extracted_images"]),
        "n_extracted_images_total": n_images_total,
        "held_out_ratio": args.held_out_ratio,
    }
    (manifest_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log("DONE. summary: " + json.dumps(summary))


if __name__ == "__main__":
    main()
