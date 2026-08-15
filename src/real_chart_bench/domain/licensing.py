"""Paper/figure license classification (design §1.3, decided in §7.2).

Pure function: given already-fetched license identifiers (as reported by
OpenAlex, with an optional Crossref fallback), classify whether the
associated figure may be redistributed. No I/O — fetching the identifiers is
an adapter concern (see adapter/openalex.py).

Empirically validated in the Phase 2 pilot (design §7.9): on a random
500-paper sample of the Thermoelectric Materials corpus, 6.0% were
REDISTRIBUTABLE, 13.0% NEEDS_REVIEW, and 81.0% EXCLUDED.
"""

from __future__ import annotations

from enum import Enum

REDISTRIBUTABLE_LICENSES = frozenset(
    {
        "cc-by",
        "cc-by-4.0",
        "cc-by-3.0",
        "cc-by-2.5",
        "cc-by-2.0",
        "cc0",
        "public-domain",
        "cc-by-sa",
        "cc-by-sa-4.0",
        "cc-by-sa-3.0",
    }
)


class LicenseStatus(Enum):
    REDISTRIBUTABLE = "REDISTRIBUTABLE"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    EXCLUDED = "EXCLUDED"


def _normalize(license_id: str | None) -> str:
    return (license_id or "").strip().lower()


def classify_license(
    license_id: str | None,
    *,
    crossref_license_id: str | None = None,
    is_oa: bool | None = None,
) -> LicenseStatus:
    """Implements the design §1.3 pseudocode, plus one pilot-driven
    refinement (design §7.9): closed-access papers (``is_oa=False``) are
    EXCLUDED immediately, before the license/Crossref-fallback logic runs.
    Without this, a 500-paper pilot sample showed 393 closed-access papers
    reporting no license at all, which the original pseudocode would have
    sent to NEEDS_REVIEW — an unhelpfully large review queue for papers that
    are unambiguously not redistributable.

    1. If ``is_oa`` is explicitly False -> EXCLUDED.
    2. If the (OpenAlex-reported) license is on the allowlist -> REDISTRIBUTABLE.
    3. If no license was reported at all, fall back to the Crossref-reported
       one; allowlisted -> REDISTRIBUTABLE, otherwise NEEDS_REVIEW (never
       EXCLUDED in this branch — there was nothing to positively exclude on).
    4. Otherwise (a *present* but non-allowlisted license) -> EXCLUDED.
    """
    normalized = _normalize(license_id)

    if normalized in REDISTRIBUTABLE_LICENSES:
        return LicenseStatus.REDISTRIBUTABLE

    if is_oa is False:
        return LicenseStatus.EXCLUDED

    if not normalized:
        fallback = _normalize(crossref_license_id)
        if fallback in REDISTRIBUTABLE_LICENSES:
            return LicenseStatus.REDISTRIBUTABLE
        return LicenseStatus.NEEDS_REVIEW

    return LicenseStatus.EXCLUDED
