"""Port for looking up a paper's open-access status and license (design §1.3).

Adapters (e.g. adapter/openalex.py) implement LicenseLookupPort against a
real API; the use case layer only depends on this abstraction (DIP).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LicenseLookupResult:
    doi: str
    is_oa: bool | None
    license_id: str | None


class LicenseLookupPort(Protocol):
    def fetch_many(self, dois: Sequence[str]) -> dict[str, LicenseLookupResult]:
        """Look up OA status + license for each DOI. DOIs that couldn't be
        resolved (e.g. not found upstream) are simply absent from the
        result dict — callers must not assume every input DOI is a key."""
        ...
