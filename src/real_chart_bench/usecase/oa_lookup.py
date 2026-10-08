"""Port for looking up a paper's open-access locations today
(figure-fetch-distribution.md §8.1/§8.4: Unpaywall ``best_oa_location`` +
``oa_locations``). The v0 collection resolved a single ``pdf_url`` per paper,
which is why 424 of the 603 licence-cleared papers ended up with no images
(§7.2); this record keeps every location so a caller can try them all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class OaLocation:
    url_for_pdf: str | None
    host_type: str | None = None  # "publisher" | "repository"
    license: str | None = None


@dataclass(frozen=True)
class OaRecord:
    is_oa: bool | None
    best_license: str | None  # licence of best_oa_location (what §8.5 re-checked)
    locations: tuple[OaLocation, ...] = ()  # best_oa_location first, then oa_locations

    def pdf_urls(self) -> list[str]:
        urls: list[str] = []
        for location in self.locations:
            if location.url_for_pdf and location.url_for_pdf not in urls:
                urls.append(location.url_for_pdf)
        return urls

    def location_for(self, url: str) -> OaLocation | None:
        return next((loc for loc in self.locations if loc.url_for_pdf == url), None)


class OaLookupPort(Protocol):
    def lookup(self, doi: str) -> OaRecord | None:
        """None means the lookup itself failed (retry later), not "closed access"."""
        ...
