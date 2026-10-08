"""Phase 1 step 1 — acquisition only (scaling-verification.md 第1段
"取得経路の多経路化", figure-fetch-distribution.md §7.2 / §8.4 / §9 Phase 2,
benchmark-architecture.md §7.87).

The v0 collection (collect_v0_dataset.py) resolved ONE OpenAlex pdf_url per
paper, so 424 of the 603 licence-cleared CC BY papers have no extracted images
(``n_extracted_images`` is null in data/manifest/v0/papers.json). This script
re-fetches exactly those papers:

1. Unpaywall lookup today (same query/contact as starrydata_fetch.py's
   pdf_urls(), now UnpaywallOaLookupAdapter); the licence must still classify
   REDISTRIBUTABLE, else the paper is skipped and logged ``licence_changed``.
2. Every best_oa_location + oa_locations url_for_pdf, in order, through
   HttpPdfFetchAdapter (``is_pdf_content`` rejects HTML interstitials).
3. Images extracted with PyMuPdfFigureExtractor (v0 defaults) and the v0
   naming convention into data/raw/images/<paper_id>/ (gitignored); the PDF
   is kept at data/raw/pdf/<paper_id>.pdf (gitignored) so re-extraction never
   needs to hit a publisher again.
4. One log entry per paper (status, route, sha256 of PDF and every image,
   every attempt) in the committed data/manifest/v0/refetch_log.json.

Politeness (owner instruction 2026-10-09): single-threaded; >= --pdf-gap seconds
(default 60, i.e. about one paper per minute) between ANY two publisher/PDF
requests, globally across hosts; >= --delay seconds (default 1.0) between any two
requests at all (this only matters for the Unpaywall metadata API); descriptive
User-Agent with the project's existing contact. The first 403 or 429 from a host
blocks that host for the rest of the run (and for every later resume: hosts that
answered 403/429 in the log are pre-blocked);
its remaining URLs are logged ``http_error`` with "host skipped: ..." in the detail.
Never worked around (§8.2: MDPI blocks scripts; 2026-10-09: RSC rate-limited the
owner's network after 31 back-to-back requests).

Resumable: the log is rewritten atomically after every paper; papers already
logged are skipped, except the transient statuses (lookup_failed,
connection_error), which are retried. Ctrl-C / kill at any point is safe.

Usage:
    python scripts/collect/refetch_cc_by_pdfs.py                 # run / resume
    python scripts/collect/refetch_cc_by_pdfs.py --limit 5       # smoke test
    python scripts/collect/refetch_cc_by_pdfs.py --update-manifest  # apply log to papers.json
    python scripts/collect/refetch_cc_by_pdfs.py --report        # print counts only
"""

from __future__ import annotations

import argparse
import collections
import datetime
import json
import os
import pathlib
import sys
import time
import urllib.parse
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from real_chart_bench.adapter.figure_extraction import PyMuPdfFigureExtractor  # noqa: E402
from real_chart_bench.adapter.pdf_fetch import HttpPdfFetchAdapter  # noqa: E402
from real_chart_bench.adapter.unpaywall import UnpaywallOaLookupAdapter  # noqa: E402
from real_chart_bench.usecase.host_politeness import (  # noqa: E402
    BLOCKING_DETAILS,
    HostPoliteFetcher,
)
from real_chart_bench.usecase.pdf_refetch import RefetchStatus, refetch_paper  # noqa: E402

UNPAYWALL_EMAIL = "tomoya.matou@gmail.com"  # the contact every collection script already uses
USER_AGENT = (
    "real-chart-bench-collector/0.1 (+https://github.com/t29mato/real-chart-bench; "
    f"mailto:{UNPAYWALL_EMAIL}; open-access CC BY figure benchmark, 1 req/s)"
)
PAPERS_PATH = ROOT / "data/manifest/v0/papers.json"
FIGURES_PATH = ROOT / "data/manifest/v0/figures.json"
SUMMARY_PATH = ROOT / "data/manifest/v0/summary.json"
LOG_PATH = ROOT / "data/manifest/v0/refetch_log.json"
TRANSIENT = {RefetchStatus.LOOKUP_FAILED.value, RefetchStatus.CONNECTION_ERROR.value}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


class Throttle:
    """Minimum gap between any two outbound requests, across all hosts."""

    def __init__(self, delay_s: float) -> None:
        self.delay_s = delay_s
        self._last = 0.0

    def wait(self) -> None:
        gap = time.monotonic() - self._last
        if gap < self.delay_s:
            time.sleep(self.delay_s - gap)
        self._last = time.monotonic()


def make_transport(throttle: Throttle, timeout_s: float):
    def transport(url: str) -> bytes:
        throttle.wait()
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:  # noqa: S310
            return resp.read()

    return transport


def candidates(papers: list[dict]) -> list[dict]:
    """§7.2: licence-cleared papers with no images from the v0 collection."""
    return [p for p in papers if p["license_id"] == "cc-by" and not p.get("n_extracted_images")]


def write_json_atomic(path: pathlib.Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def load_log() -> dict:
    if LOG_PATH.exists():
        return json.loads(LOG_PATH.read_text())
    return {
        "description": (
            "Per-paper PDF re-fetch over every Unpaywall OA location for the licence-cleared "
            "papers without images (scripts/collect/refetch_cc_by_pdfs.py, "
            "benchmark-architecture.md §7.87). PDFs and images are NOT in git; "
            "sha256 values identify them."
        ),
        "papers": {},
    }


def blocked_hosts_from_log(run_log: dict) -> dict[str, str]:
    blocked: dict[str, str] = {}
    for entry in run_log["papers"].values():
        for a in entry.get("attempts") or []:
            if a["detail"] in BLOCKING_DETAILS:
                blocked.setdefault(urllib.parse.urlparse(a["url"]).netloc.lower(), a["detail"])
    return blocked


def figure_counts() -> dict[str, collections.Counter]:
    counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for f in json.loads(FIGURES_PATH.read_text()):
        counts[f["paper_id"]][f["split"]] += 1
    return counts


def run(args) -> None:
    raw = args.raw_dir
    (raw / "pdf").mkdir(parents=True, exist_ok=True)
    (raw / "images").mkdir(parents=True, exist_ok=True)

    papers = json.loads(PAPERS_PATH.read_text())
    todo_all = candidates(papers)
    run_log = load_log()
    done = run_log["papers"]
    todo = [p for p in todo_all
            if p["paper_id"] not in done or done[p["paper_id"]]["status"] in TRANSIENT]
    if args.limit:
        todo = todo[: args.limit]
    log(f"candidates {len(todo_all)}, already logged {len(done)}, to do now {len(todo)}")

    throttle = Throttle(args.delay)
    transport = make_transport(throttle, args.timeout)
    oa = UnpaywallOaLookupAdapter(email=UNPAYWALL_EMAIL, transport=transport)
    blocked = blocked_hosts_from_log(run_log)
    if blocked:
        log(f"pre-blocked hosts (403/429 earlier): {sorted(blocked)}")
    fetcher = HostPoliteFetcher(HttpPdfFetchAdapter(transport=transport),
                                min_gap_s=args.pdf_gap, blocked_hosts=blocked)
    extractor = PyMuPdfFigureExtractor()  # v0 defaults: same images, same names
    figs = figure_counts()

    for n, paper in enumerate(todo, 1):
        sid, doi = paper["paper_id"], paper["doi"]
        outcome = refetch_paper(oa.lookup(doi), fetcher, extractor)
        if outcome.pdf_bytes:
            (raw / "pdf" / f"{sid}.pdf").write_bytes(outcome.pdf_bytes)
        if outcome.images:
            img_dir = raw / "images" / sid
            img_dir.mkdir(parents=True, exist_ok=True)
            for img in outcome.images:
                (img_dir / img.name).write_bytes(img.image_bytes)
        done[sid] = {
            "doi": doi,
            "n_figures": paper["n_figures"],
            "n_figures_public": figs[sid]["public"],
            "fetched_at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
            **outcome.to_log_entry(),
        }
        write_json_atomic(LOG_PATH, run_log)
        log(f"{n}/{len(todo)} {sid} {doi}: {outcome.status.value}"
            f" images={outcome.n_images} attempts={len(outcome.attempts)}")
    report(run_log)


def _publisher(entry: dict) -> str:
    urls = [a["url"] for a in entry.get("attempts") or []]
    if urls:
        return urllib.parse.urlparse(urls[0]).netloc
    return "doi:" + entry["doi"].split("/")[0]


def report(run_log: dict) -> None:
    entries = run_log["papers"]
    by_status = collections.Counter(e["status"] for e in entries.values())
    ok = [e for e in entries.values() if e["status"] == "ok"]
    with_images = [e for e in ok if e["n_images"]]
    route_host = collections.Counter(urllib.parse.urlparse(e["route"]["url"]).netloc for e in ok)
    route_type = collections.Counter(e["route"]["host_type"] for e in ok)
    route_rank = collections.Counter(
        next(i for i, a in enumerate(e["attempts"]) if a["status"] == "ok") for e in ok)
    fail_by_pub = collections.Counter(
        (e["status"], _publisher(e)) for e in entries.values() if e["status"] != "ok")
    licence_today = collections.Counter(e["licence_today"] for e in entries.values())
    dist = collections.Counter(e["n_images"] for e in ok)
    out = {
        "papers_logged": len(entries),
        "by_status": dict(by_status),
        "licence_today": {str(k): v for k, v in licence_today.items()},
        "ok_route_host": dict(route_host.most_common()),
        "ok_route_host_type": dict(route_type),
        "ok_route_rank_(0=best_oa_location)": dict(sorted(route_rank.items())),
        "failures_by_status_and_host": {f"{s} {h}": c for (s, h), c in fail_by_pub.most_common()},
        "images_total": sum(e["n_images"] for e in ok),
        "images_per_paper": dict(sorted(dist.items())),
        "new_papers_with_images": len(with_images),
        "their_figures": sum(e["n_figures"] for e in with_images),
        "their_public_figures": sum(e["n_figures_public"] for e in with_images),
    }
    print(json.dumps(out, indent=1))


def update_manifest() -> None:
    """collect_v0_dataset.py's convention: papers.json carries n_extracted_images
    (and pdf_status) per paper; prepare_hf_dataset.py keys on n_extracted_images."""
    entries = load_log()["papers"]
    papers = json.loads(PAPERS_PATH.read_text())
    changed = 0
    for p in papers:
        e = entries.get(p["paper_id"])
        if e is None or e["status"] in TRANSIENT:
            continue
        p["pdf_status"] = e["status"]
        if e["status"] == "ok" and e["n_images"]:
            p["n_extracted_images"] = e["n_images"]
        changed += 1
    PAPERS_PATH.write_text(json.dumps(papers, indent=2) + "\n")
    summary = json.loads(SUMMARY_PATH.read_text())
    summary["n_papers_with_images"] = sum(1 for p in papers if p.get("n_extracted_images"))
    summary["n_extracted_images_total"] = sum(p.get("n_extracted_images") or 0 for p in papers)
    summary["refetch_note"] = (
        "n_papers_with_images / n_extracted_images_total include the 2026-10-09 re-fetch over "
        "all Unpaywall OA locations (data/manifest/v0/refetch_log.json, design §7.87); "
        "pdf_status of re-fetched papers is the re-fetch status."
    )
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n")
    log(f"papers.json: {changed} papers updated from the refetch log")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--raw-dir", type=pathlib.Path, default=ROOT / "data/raw",
                    help="gitignored raw area (pdf/ and images/<paper_id>/ go here)")
    ap.add_argument("--delay", type=float, default=1.0, help="min seconds between requests")
    ap.add_argument("--pdf-gap", type=float, default=60.0,
                    help="min seconds between any two publisher/PDF requests (all hosts)")
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--limit", type=int, default=0, help="process at most N papers this run")
    ap.add_argument("--report", action="store_true", help="print counts from the log and exit")
    ap.add_argument("--update-manifest", action="store_true",
                    help="apply the log to papers.json / summary.json and exit")
    args = ap.parse_args()
    if args.delay < 1.0:
        ap.error("--delay must be >= 1.0 s (politeness, design §6.3)")
    if args.report:
        report(load_log())
    elif args.update_manifest:
        update_manifest()
    else:
        run(args)


if __name__ == "__main__":
    main()
