"""Step 1 of the real-figure training data (docs/design/local-model.md,
"データ: 実図(Starrydata)"): the candidate pool and its images.

Pool = the thermoelectric corpus's CC-BY papers (data/manifest/v0/papers.json,
license checked via OpenAlex at collection time, design 7.13) minus every
paper the benchmark has ever considered (registry.json, any status).

For each pool paper with an open-access pdf_url (Unpaywall), fetch the PDF once (rate
limited, resumable) and extract candidate images with the same extractor the
v0 collection used (embedded images; a page render where a page has none).
Nothing here reads a figure's values; that is starrydata_build.py's job.

Usage:
    python scripts/train/starrydata_fetch.py [--work ~/.cache/real-chart-bench/starrydata-work]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from real_chart_bench.adapter.figure_extraction import PyMuPdfFigureExtractor  # noqa: E402
from real_chart_bench.adapter.pdf_fetch import HttpPdfFetchAdapter  # noqa: E402
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.usecase.pdf_fetch import PdfFetchStatus  # noqa: E402
from real_chart_bench.usecase.real_image_gate import benchmark_paper_ids  # noqa: E402

RELEASE_BASE = "https://github.com/starrydata/starrydata_datasets/releases/download/latest"
USER_AGENT = "real-chart-bench-collector/0.1 (mailto:tomoya.matou@gmail.com)"
PDF_DELAY_S = 0.5
RENDER_DPI = 200  # page renders (vector charts); a little above v0's 150 for tick OCR


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def download(url: str, dest: pathlib.Path) -> None:
    if dest.exists():
        return
    log(f"downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=120) as resp:  # noqa: S310
        dest.write_bytes(resp.read())


def pdf_urls(dois: list[str], cache_path: pathlib.Path) -> dict:
    """DOI -> {license, pdf_urls} from Unpaywall (OpenAlex's daily quota was
    exhausted on 2026-10-06; Unpaywall carries the same OA locations)."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for n, doi in enumerate(d for d in dois if d not in cache):
        url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email=tomoya.matou@gmail.com"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
                data = json.loads(resp.read())
        except OSError as e:
            log(f"unpaywall {doi}: {e}")
            cache[doi] = {"license": None, "pdf_urls": []}
            continue
        locs = [data.get("best_oa_location"), *(data.get("oa_locations") or [])]
        urls = []
        for loc in locs:
            if loc and loc.get("url_for_pdf") and loc["url_for_pdf"] not in urls:
                urls.append(loc["url_for_pdf"])
        best = data.get("best_oa_location") or {}
        cache[doi] = {"license": best.get("license"), "pdf_urls": urls}
        if n % 25 == 0:
            cache_path.write_text(json.dumps(cache))
        time.sleep(0.2)
    cache_path.write_text(json.dumps(cache))
    return cache


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=pathlib.Path,
                    default=pathlib.Path.home() / ".cache/real-chart-bench/starrydata-work")
    args = ap.parse_args()
    work = args.work
    (work / "pdf").mkdir(parents=True, exist_ok=True)
    (work / "images").mkdir(parents=True, exist_ok=True)

    download(f"{RELEASE_BASE}/ThermoelectricMaterials_papers.csv.gz",
             work / "ThermoelectricMaterials_papers.csv.gz")
    download(f"{RELEASE_BASE}/ThermoelectricMaterials_curves.csv.gz",
             work / "ThermoelectricMaterials_curves.csv.gz")

    papers = json.loads((ROOT / "data/manifest/v0/papers.json").read_text())
    excluded = benchmark_paper_ids(load_registry(ROOT / "data/verified_pairs/registry.json"))
    cc_by = [p for p in papers if p["license_id"] == "cc-by"]
    pool = [p for p in cc_by if p["paper_id"] not in excluded]
    log(f"CC-BY papers {len(cc_by)}, benchmark papers excluded "
        f"{len(cc_by) - len(pool)}, pool {len(pool)}")

    oa = pdf_urls([p["doi"].lower() for p in pool], work / "unpaywall.json")
    status_path = work / "fetch_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    fetcher = HttpPdfFetchAdapter()
    extractor = PyMuPdfFigureExtractor(render_dpi=RENDER_DPI)

    for n, p in enumerate(pool, 1):
        sid = p["paper_id"]
        if sid in status:
            continue
        info = oa.get(p["doi"].lower()) or {}
        # The license must still read CC-BY today; anything else stays out.
        lic = (info.get("license") or "").lower()
        if lic and not lic.startswith("cc-by") or lic.startswith("cc-by-nc") \
                or lic.startswith("cc-by-nd") or lic.startswith("cc-by-sa"):
            status[sid] = {"status": "license_changed", "license": lic}
            continue
        result = None
        for url in info.get("pdf_urls") or []:
            result = fetcher.fetch(url)
            time.sleep(PDF_DELAY_S)
            if result.status is PdfFetchStatus.OK:
                break
        if result is None:
            status[sid] = {"status": "no_url"}
        elif result.status is not PdfFetchStatus.OK:
            status[sid] = {"status": result.status.value, "detail": result.detail}
        else:
            (work / "pdf" / f"{sid}.pdf").write_bytes(result.content)
            try:
                images = extractor.extract(result.content)
            except Exception as e:  # noqa: BLE001
                status[sid] = {"status": "extract_failed", "detail": str(e)}
                continue
            d = work / "images" / sid
            d.mkdir(exist_ok=True)
            names = []
            for i, img in enumerate(images):
                ext = "png" if img.source.value == "page_render" else "img"
                name = f"p{img.page_number:02d}_{img.source.value}_{i}.{ext}"
                (d / name).write_bytes(img.image_bytes)
                names.append(name)
            status[sid] = {"status": "ok", "n_images": len(names)}
        if n % 10 == 0:
            status_path.write_text(json.dumps(status, indent=1))
            ok = sum(1 for s in status.values() if s["status"] == "ok")
            log(f"{n}/{len(pool)} papers, {ok} with images")
    status_path.write_text(json.dumps(status, indent=1))
    ok = sum(1 for s in status.values() if s["status"] == "ok")
    log(f"done: {ok}/{len(pool)} pool papers with images")


if __name__ == "__main__":
    main()
