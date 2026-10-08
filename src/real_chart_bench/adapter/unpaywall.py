"""Unpaywall-backed implementation of OaLookupPort
(figure-fetch-distribution.md §8.1: same query and location order as
``scripts/train/starrydata_fetch.py``'s ``pdf_urls()``, which OpenAlex's daily
quota pushed onto Unpaywall on 2026-10-06).

HTTP access is isolated behind an injectable ``transport`` (URL -> bytes) so
tests never touch the network. Any failure of the lookup itself returns None,
which callers record as retryable rather than as "closed access".
"""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from collections.abc import Callable

from real_chart_bench.usecase.oa_lookup import OaLocation, OaRecord

_API_BASE = "https://api.unpaywall.org/v2/"
_USER_AGENT = "real-chart-bench/0.0.1 (https://github.com/t29mato/real-chart-bench)"

Transport = Callable[[str], bytes]


def _default_transport(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
        return response.read()


def _location(raw: dict) -> OaLocation:
    return OaLocation(
        url_for_pdf=raw.get("url_for_pdf"),
        host_type=raw.get("host_type"),
        license=raw.get("license"),
    )


class UnpaywallOaLookupAdapter:
    def __init__(self, *, email: str, transport: Transport | None = None) -> None:
        self._email = email
        self._transport = transport or _default_transport

    def lookup(self, doi: str) -> OaRecord | None:
        url = f"{_API_BASE}{urllib.parse.quote(doi)}?email={self._email}"
        try:
            data = json.loads(self._transport(url))
        except (OSError, ValueError):
            return None
        best = data.get("best_oa_location")
        raw_locations = [best, *(data.get("oa_locations") or [])]
        return OaRecord(
            is_oa=data.get("is_oa"),
            best_license=(best or {}).get("license"),
            locations=tuple(_location(loc) for loc in raw_locations if loc),
        )
