"""Phase 3 pilot: run the real collection pipeline against a batch of
CC-BY-confirmed ThermoelectricMaterials papers (design §7.9/§7.11).

Downloads Starrydata's `latest` CSV release, queries OpenAlex for license +
pdf_url, filters to REDISTRIBUTABLE papers, then runs the real
HttpPdfFetchAdapter -> PyMuPdfFigureExtractor -> build_ground_truth_for_paper
pipeline end to end. Writes a summary + manifest to --out-dir (gitignored by
default: point it at data/pilot/ or /tmp).

Not part of the pytest suite (it does real network I/O) — this is the same
kind of standalone, re-runnable pilot script as deep-digitizer's
scripts/pilot/01-03, using real-chart-bench's own adapters instead of
ad-hoc code so the pipeline being validated is the actual shipped code.

Usage:
    python scripts/pilot/phase3_collect_cc_by_batch.py --limit 30 --out-dir data/pilot
"""

from __future__ import annotations

import argparse
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
USER_AGENT = "real-chart-bench-pilot/0.1 (mailto:tomoya.matou@gmail.com)"


def _download(url: str, dest: pathlib.Path) -> None:
    if dest.exists():
        return
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        dest.write_bytes(resp.read())


def _fetch_openalex_batch(dois: list[str]) -> dict[str, dict]:
    params = urllib.parse.urlencode(
        {
            "filter": "doi:" + "|".join(dois),
            "select": "id,doi,open_access,primary_location,best_oa_location",
            "per-page": len(dois),
            "mailto": "tomoya.matou@gmail.com",
        }
    )
    url = f"https://api.openalex.org/works?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        data = json.loads(resp.read())

    out = {}
    for work in data.get("results", []):
        doi = (work.get("doi") or "").removeprefix("https://doi.org/")
        primary = work.get("primary_location") or {}
        best_oa = work.get("best_oa_location") or {}
        out[doi] = {
            "is_oa": (work.get("open_access") or {}).get("is_oa"),
            "license": primary.get("license"),
            "pdf_url": best_oa.get("pdf_url") or primary.get("pdf_url"),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("data/pilot"))
    parser.add_argument("--seed-sample-size", type=int, default=500)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    papers_gz = args.out_dir / "ThermoelectricMaterials_papers.csv.gz"
    curves_gz = args.out_dir / "ThermoelectricMaterials_curves.csv.gz"
    _download(f"{RELEASE_BASE}/ThermoelectricMaterials_papers.csv.gz", papers_gz)
    _download(f"{RELEASE_BASE}/ThermoelectricMaterials_curves.csv.gz", curves_gz)

    import csv
    import random

    with gzip.open(papers_gz, "rt", newline="", encoding="utf-8") as f:
        papers_by_doi = {r["DOI"]: r for r in csv.DictReader(f)}

    random.seed(7)  # same seed as the Phase 2 pilot, for a comparable sample
    sample_dois = random.sample(list(papers_by_doi), args.seed_sample_size)

    openalex_results: dict[str, dict] = {}
    for i in range(0, len(sample_dois), 40):
        batch = sample_dois[i : i + 40]
        try:
            openalex_results.update(_fetch_openalex_batch(batch))
        except urllib.error.URLError as e:
            print(f"OpenAlex batch {i} failed: {e}", file=sys.stderr)
        time.sleep(0.3)

    candidates = []
    for doi, r in openalex_results.items():
        status = classify_license(r["license"], is_oa=r["is_oa"])
        if status is LicenseStatus.REDISTRIBUTABLE and r.get("pdf_url"):
            candidates.append((doi, r["pdf_url"]))
    candidates = candidates[: args.limit]
    print(f"REDISTRIBUTABLE candidates with a pdf_url: {len(candidates)}")

    with gzip.open(curves_gz, "rt", newline="", encoding="utf-8") as f:
        curves_by_sid: dict[str, list] = {}
        for row in csv.DictReader(f):
            curves_by_sid.setdefault(row["SID"], []).append(row)

    pdf_fetcher = HttpPdfFetchAdapter()
    extractor = PyMuPdfFigureExtractor()

    fetch_status_counts: dict[str, int] = {}
    manifest_papers, manifest_figures, manifest_curves = [], [], []
    image_counts = []

    for doi, pdf_url in candidates:
        sid = papers_by_doi[doi]["SID"]
        paper = PaperRecord(
            paper_id=sid, doi=doi, title=papers_by_doi[doi].get("title", ""),
            license_status=LicenseStatus.REDISTRIBUTABLE, license_id="cc-by",
        )

        fetch_result = pdf_fetcher.fetch(pdf_url)
        fetch_status_counts[fetch_result.status.value] = (
            fetch_status_counts.get(fetch_result.status.value, 0) + 1
        )

        n_images = None
        if fetch_result.status is PdfFetchStatus.OK and fetch_result.content:
            try:
                images = extractor.extract(fetch_result.content)
                n_images = len(images)
            except Exception as e:  # noqa: BLE001
                print(f"  extract failed for SID {sid}: {e}", file=sys.stderr)
        if n_images is not None:
            image_counts.append(n_images)

        rows = curves_by_sid.get(sid, [])
        try:
            parsed = [parse_curve_row(row) for row in rows]
            figures, curves = build_ground_truth_for_paper(paper, parsed, held_out_ratio=0.2)
        except ValueError as e:
            print(f"  ground truth build failed for SID {sid}: {e}", file=sys.stderr)
            figures, curves = (), ()

        manifest_papers.append({"paper_id": sid, "doi": doi, "n_extracted_images": n_images})
        manifest_figures.extend(
            {"figure_id": f.figure_id, "paper_id": f.paper_id, "split": f.split.value}
            for f in figures
        )
        manifest_curves.extend(
            {"curve_id": c.curve_id, "figure_id": c.figure_id, "n_points": len(c.x_values)}
            for c in curves
        )
        time.sleep(0.5)

    summary = {
        "candidates_attempted": len(candidates),
        "pdf_fetch_status_counts": fetch_status_counts,
        "pdf_fetch_ok_rate": fetch_status_counts.get("ok", 0) / max(1, len(candidates)),
        "mean_extracted_images_per_ok_pdf": (
            sum(image_counts) / len(image_counts) if image_counts else None
        ),
        "n_papers_with_ground_truth": len({f["paper_id"] for f in manifest_figures}),
        "n_figure_records": len(manifest_figures),
        "n_ground_truth_curves": len(manifest_curves),
    }
    print(json.dumps(summary, indent=2))

    (args.out_dir / "phase3_pilot_summary.json").write_text(json.dumps(summary, indent=2))
    (args.out_dir / "phase3_pilot_manifest.json").write_text(
        json.dumps(
            {"papers": manifest_papers, "figures": manifest_figures, "curves": manifest_curves},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
