"""Re-fetch one licence-cleared paper's PDF over every OA location and extract
its candidate images (scaling-verification.md 第1段 "取得経路の多経路化",
figure-fetch-distribution.md §7.2 / §9 Phase 2).

Per paper: re-check the licence as it reads *today* (§8.5 found 3.0% of the
collection-time ``cc-by`` labels had moved), then try each ``url_for_pdf`` in
Unpaywall order until one returns a real PDF, then extract images with the
same extractor and file naming as the v0 collection. The outcome records every
attempt so failures stay attributable to a route (§8.2: 5 of 6 failures were one
publisher's 403, which a single "failed" bucket would have hidden).
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from real_chart_bench.domain.licensing import LicenseStatus, classify_license
from real_chart_bench.usecase.figure_extraction import FigureExtractionPort, canonical_image_name
from real_chart_bench.usecase.oa_lookup import OaRecord
from real_chart_bench.usecase.pdf_fetch import PdfFetchPort, PdfFetchStatus


class RefetchStatus(Enum):
    OK = "ok"
    LICENCE_CHANGED = "licence_changed"
    NO_URL = "no_url"
    HTTP_ERROR = "http_error"
    NOT_PDF = "not_pdf"
    CONNECTION_ERROR = "connection_error"
    LOOKUP_FAILED = "lookup_failed"
    EXTRACT_FAILED = "extract_failed"


_FROM_FETCH = {
    PdfFetchStatus.OK: RefetchStatus.OK,
    PdfFetchStatus.NO_URL: RefetchStatus.NO_URL,
    PdfFetchStatus.NOT_A_PDF: RefetchStatus.NOT_PDF,
    PdfFetchStatus.HTTP_ERROR: RefetchStatus.HTTP_ERROR,
    PdfFetchStatus.CONNECTION_ERROR: RefetchStatus.CONNECTION_ERROR,
}


@dataclass(frozen=True)
class FetchAttempt:
    url: str
    status: RefetchStatus
    detail: str | None = None


@dataclass(frozen=True)
class NamedImage:
    name: str
    image_bytes: bytes
    sha256: str


@dataclass(frozen=True)
class RefetchOutcome:
    status: RefetchStatus
    licence_today: str | None = None
    attempts: tuple[FetchAttempt, ...] = ()
    route_url: str | None = None
    route_host_type: str | None = None
    pdf_bytes: bytes | None = None
    pdf_sha256: str | None = None
    images: tuple[NamedImage, ...] = ()
    detail: str | None = None

    @property
    def n_images(self) -> int:
        return len(self.images)

    def to_log_entry(self) -> dict:
        return {
            "status": self.status.value,
            "licence_today": self.licence_today,
            "route": (
                {"url": self.route_url, "host_type": self.route_host_type}
                if self.route_url
                else None
            ),
            "pdf_sha256": self.pdf_sha256,
            "n_images": self.n_images,
            "images": [{"name": i.name, "sha256": i.sha256} for i in self.images],
            "attempts": [
                {"url": a.url, "status": a.status.value, "detail": a.detail} for a in self.attempts
            ],
            "detail": self.detail,
        }


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def refetch_paper(
    record: OaRecord | None,
    fetcher: PdfFetchPort,
    extractor: FigureExtractionPort,
    *,
    pause: Callable[[], None] = lambda: None,
) -> RefetchOutcome:
    """``pause`` is called between consecutive PDF requests (the rate-limit
    policy belongs to the caller; this function only guarantees it is consulted)."""
    if record is None:
        return RefetchOutcome(status=RefetchStatus.LOOKUP_FAILED)

    licence = record.best_license
    if classify_license(licence, is_oa=record.is_oa) is not LicenseStatus.REDISTRIBUTABLE:
        return RefetchOutcome(status=RefetchStatus.LICENCE_CHANGED, licence_today=licence)

    urls = record.pdf_urls()
    if not urls:
        return RefetchOutcome(status=RefetchStatus.NO_URL, licence_today=licence)

    attempts: list[FetchAttempt] = []
    for n, url in enumerate(urls):
        if n:
            pause()
        result = fetcher.fetch(url)
        status = _FROM_FETCH[result.status]
        attempts.append(FetchAttempt(url=url, status=status, detail=result.detail))
        if status is RefetchStatus.OK and result.content:
            return _extract(record, url, result.content, licence, tuple(attempts), extractor)

    return RefetchOutcome(
        status=attempts[-1].status, licence_today=licence, attempts=tuple(attempts)
    )


def _extract(
    record: OaRecord,
    url: str,
    pdf: bytes,
    licence: str | None,
    attempts: tuple[FetchAttempt, ...],
    extractor: FigureExtractionPort,
) -> RefetchOutcome:
    location = record.location_for(url)
    common = {
        "licence_today": licence,
        "attempts": attempts,
        "route_url": url,
        "route_host_type": location.host_type if location else None,
        "pdf_bytes": pdf,
        "pdf_sha256": _sha256(pdf),
    }
    try:
        extracted = extractor.extract(pdf)
    except Exception as exc:  # noqa: BLE001 - one malformed PDF must not stop a batch
        return RefetchOutcome(status=RefetchStatus.EXTRACT_FAILED, detail=str(exc), **common)
    images = tuple(
        NamedImage(
            name=canonical_image_name(i, img),
            image_bytes=img.image_bytes,
            sha256=_sha256(img.image_bytes),
        )
        for i, img in enumerate(extracted)
    )
    return RefetchOutcome(status=RefetchStatus.OK, images=images, **common)
