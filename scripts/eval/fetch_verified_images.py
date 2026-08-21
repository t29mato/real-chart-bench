"""Fetches just the images referenced by data/verified_pairs/registry.json's
VERIFIED entries (design §7.31, HQ usability audit 2026-08-22).

Why this exists: `data/raw/images/` is gitignored (regeneratable, 546MB for
the full v0 collection — see design §6), so a fresh clone of this repo has
no images at all. Most VERIFIED registry entries reference a bare filename
under `data/raw/images/{paper_id}/` (an embedded or page-rendered image
extracted from that paper's PDF), so `scripts/eval/run_baselines.py` would
fail with FileNotFoundError on a fresh clone without this step first.

This is a *targeted* re-fetch — only the handful of papers referenced by the
verified-pairs registry, not the full 603-paper v0 collection — so it's a
single-digit number of PDF requests, well within politeness norms for the
source publisher servers (PDF_FETCH_DELAY_S between requests, same as the
full collection script). Entries whose image_path already contains '/' (a
committed, manually-corrected crop under data/verified_pairs/crops/, see
design §7.21/§7.27) need no fetch — they're already in git.

Usage:
    python scripts/eval/fetch_verified_images.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.adapter.figure_extraction import PyMuPdfFigureExtractor  # noqa: E402
from real_chart_bench.adapter.pdf_fetch import HttpPdfFetchAdapter  # noqa: E402
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.usecase.pdf_fetch import PdfFetchStatus  # noqa: E402
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data/verified_pairs/registry.json"
PAPERS_PATH = REPO_ROOT / "data/manifest/v0/papers.json"
IMAGES_DIR = REPO_ROOT / "data/raw/images"
PDF_FETCH_DELAY_S = 0.5  # politeness -- same as scripts/collect/collect_v0_dataset.py

_USER_AGENT = "real-chart-bench/0.0.1 (https://github.com/t29mato/real-chart-bench)"


def _resolve_pdf_url(doi: str) -> str | None:
    """Same OpenAlex lookup as collect_v0_dataset.py's classify_all_papers()
    -- papers.json intentionally doesn't store pdf_url (small committed
    metadata, design §6), so it's re-resolved from the paper's DOI."""
    params = urllib.parse.urlencode({"filter": f"doi:{doi}", "per-page": 1})
    req = urllib.request.Request(
        f"https://api.openalex.org/works?{params}",
        headers={"User-Agent": _USER_AGENT},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        data = json.load(resp)
    results = data.get("results") or []
    if not results:
        return None
    work = results[0]
    best_oa = work.get("best_oa_location") or {}
    primary = work.get("primary_location") or {}
    return best_oa.get("pdf_url") or primary.get("pdf_url")


def _named_images(extracted) -> dict[str, bytes]:
    """Same naming convention as collect_v0_dataset.py, so a registry
    entry's image_path (e.g. 'p04_embedded_4.jpg') addresses the same
    extracted image again."""
    named = {}
    for i, img in enumerate(extracted):
        ext = "png" if img.source.value == "page_render" else "jpg"
        named[f"p{img.page_number:02d}_{img.source.value}_{i}.{ext}"] = img.image_bytes
    return named


def main() -> None:
    registry = load_registry(REGISTRY_PATH)
    verified = select_verified_pairings(registry)
    papers_by_id = {p["paper_id"]: p for p in json.loads(PAPERS_PATH.read_text())}

    # Only bare-filename entries need fetching; "/"-path entries are
    # committed crops already present after `git clone` (design §7.21).
    to_fetch = {
        p.paper_id: p.image_path
        for p in verified
        if p.image_path is not None and "/" not in p.image_path
    }

    already_present = [
        pid
        for pid, image_path in to_fetch.items()
        if (IMAGES_DIR / pid / image_path).exists()
    ]
    pending = {pid: img for pid, img in to_fetch.items() if pid not in already_present}

    print(f"{len(to_fetch)} paper(s) referenced by bare-filename image_path")
    print(f"  {len(already_present)} already present, {len(pending)} to fetch")

    if not pending:
        print("Nothing to fetch. Done.")
        return

    pdf_fetcher = HttpPdfFetchAdapter()
    extractor = PyMuPdfFigureExtractor()

    for i, (paper_id, image_path) in enumerate(pending.items(), start=1):
        paper = papers_by_id.get(paper_id)
        if paper is None:
            print(f"  [{i}/{len(pending)}] paper {paper_id}: not found in papers.json, skipping")
            continue

        pdf_url = _resolve_pdf_url(paper["doi"])
        if not pdf_url:
            print(f"  [{i}/{len(pending)}] paper {paper_id}: no pdf_url resolvable, skipping")
            continue

        fetch_result = pdf_fetcher.fetch(pdf_url)
        if fetch_result.status is not PdfFetchStatus.OK or not fetch_result.content:
            print(
                f"  [{i}/{len(pending)}] paper {paper_id}: "
                f"PDF fetch failed ({fetch_result.status})"
            )
            continue

        extracted = extractor.extract(fetch_result.content)
        named = _named_images(extracted)
        if image_path not in named:
            print(
                f"  [{i}/{len(pending)}] paper {paper_id}: expected image {image_path!r} "
                f"not found among {len(named)} re-extracted images (PDF may have changed)"
            )
            continue

        out_dir = IMAGES_DIR / paper_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / image_path).write_bytes(named[image_path])
        print(f"  [{i}/{len(pending)}] paper {paper_id}: fetched {image_path}")

        time.sleep(PDF_FETCH_DELAY_S)

    print("Done.")


if __name__ == "__main__":
    main()
