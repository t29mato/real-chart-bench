"""Publisher politeness around a PdfFetchPort (benchmark-architecture.md §7.87).

HQ 2026-10-09: a run spaced 1 s apart still sent 31 requests to pubs.rsc.org
in quick succession (one per paper) and the owner's browser on the same network
then got HTTP 429 from RSC; MDPI kept being asked after answering 403 every time.
Owner instruction the same day: about one publisher request per minute. So:

- at least ``min_gap_s`` (owner default: 60 s) between ANY two PDF requests,
  whatever their hosts (global, not per host);
- after the first 403 or 429 from a host, never contact that host again in this
  run; later URLs on it are answered locally as ``http_error`` with the host and
  the original code in ``detail``.
"""

from __future__ import annotations

import time
import urllib.parse
from collections.abc import Callable

from real_chart_bench.usecase.pdf_fetch import PdfFetchPort, PdfFetchResult, PdfFetchStatus

BLOCKING_DETAILS = frozenset({"HTTP 403", "HTTP 429"})


class HostPoliteFetcher:
    def __init__(
        self,
        inner: PdfFetchPort,
        *,
        min_gap_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        blocked_hosts: dict[str, str] | None = None,
    ) -> None:
        self._inner = inner
        self._min_gap_s = min_gap_s
        self._clock = clock
        self._sleep = sleep
        self._last: float | None = None
        self.blocked_hosts: dict[str, str] = dict(blocked_hosts or {})

    def fetch(self, pdf_url: str | None) -> PdfFetchResult:
        if not pdf_url:
            return self._inner.fetch(pdf_url)
        host = urllib.parse.urlparse(pdf_url).netloc.lower()
        if host in self.blocked_hosts:
            return PdfFetchResult(
                status=PdfFetchStatus.HTTP_ERROR,
                detail=f"host skipped: {host} returned {self.blocked_hosts[host]} earlier",
            )
        if self._last is not None:
            wait = self._last + self._min_gap_s - self._clock()
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()
        result = self._inner.fetch(pdf_url)
        if result.status is PdfFetchStatus.HTTP_ERROR and result.detail in BLOCKING_DETAILS:
            self.blocked_hosts[host] = result.detail
        return result
