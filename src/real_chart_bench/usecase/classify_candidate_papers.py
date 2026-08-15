"""Use case: classify a batch of candidate papers (by DOI) for redistribution
eligibility (design §1.2 pipeline step "license が再配布許容リストに一致?").
"""

from __future__ import annotations

from collections.abc import Sequence

from real_chart_bench.domain.licensing import LicenseStatus, classify_license
from real_chart_bench.usecase.license_lookup import LicenseLookupPort


def classify_candidate_papers(
    dois: Sequence[str],
    *,
    license_lookup: LicenseLookupPort,
) -> dict[str, LicenseStatus]:
    resolved = license_lookup.fetch_many(dois)

    statuses: dict[str, LicenseStatus] = {}
    for doi in dois:
        result = resolved.get(doi)
        if result is None:
            # Not resolvable upstream (e.g. DOI unknown to OpenAlex): treat
            # as needing manual review rather than silently dropping it or
            # assuming closed access.
            statuses[doi] = LicenseStatus.NEEDS_REVIEW
            continue
        statuses[doi] = classify_license(result.license_id, is_oa=result.is_oa)

    return statuses
