"""OpenAlex-backed implementation of LicenseLookupPort (design §1.1/§1.3).

HTTP access is isolated behind an injectable ``transport`` callable
(URL -> response bytes) so tests never need a live network connection; the
default transport is a thin urllib wrapper (stdlib only, no extra
dependency). Batches DOIs into OpenAlex OR-filter queries (validated in the
Phase 2 pilot, design §7.9: 500 DOIs resolved cleanly in batches of 40).
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence

from real_chart_bench.usecase.license_lookup import LicenseLookupResult

_API_BASE = "https://api.openalex.org/works"
_DEFAULT_BATCH_SIZE = 40
_USER_AGENT = "real-chart-bench/0.0.1 (https://github.com/t29mato/real-chart-bench)"

Transport = Callable[[str], bytes]


def _default_transport(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=30) as response:  # noqa: S310
        return response.read()


def _chunks(items: Sequence[str], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _strip_doi_prefix(doi: str | None) -> str | None:
    if not doi:
        return None
    return doi.removeprefix("https://doi.org/")


def _extract_license(work: dict) -> str | None:
    primary = work.get("primary_location") or {}
    if primary.get("license"):
        return primary["license"]
    for location in work.get("locations") or []:
        if location and location.get("license"):
            return location["license"]
    return None


class OpenAlexLicenseLookupAdapter:
    def __init__(
        self,
        *,
        transport: Transport | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        mailto: str | None = None,
    ) -> None:
        self._transport = transport or _default_transport
        self._batch_size = batch_size
        self._mailto = mailto

    def fetch_many(self, dois: Sequence[str]) -> dict[str, LicenseLookupResult]:
        if not dois:
            return {}

        results: dict[str, LicenseLookupResult] = {}
        for batch in _chunks(list(dois), self._batch_size):
            data = json.loads(self._transport(self._build_url(batch)))
            for work in data.get("results", []):
                doi = _strip_doi_prefix(work.get("doi"))
                if not doi:
                    continue
                open_access = work.get("open_access") or {}
                results[doi] = LicenseLookupResult(
                    doi=doi,
                    is_oa=open_access.get("is_oa"),
                    license_id=_extract_license(work),
                )
        return results

    def _build_url(self, batch: Sequence[str]) -> str:
        params = {
            "filter": "doi:" + "|".join(batch),
            "select": "id,doi,open_access,primary_location,locations",
            "per-page": len(batch),
        }
        if self._mailto:
            params["mailto"] = self._mailto
        return f"{_API_BASE}?{urllib.parse.urlencode(params)}"
