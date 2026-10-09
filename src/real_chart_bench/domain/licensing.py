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

import re
from dataclasses import dataclass
from enum import Enum

from real_chart_bench.domain.dataset_subset import DatasetSubset

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
    """The ``core``-only gate (design §7.2): NC licences are EXCLUDED here.
    Kept unchanged for callers that admit only ``core`` figures; the
    collection pipeline uses ``classify_figure_license`` (§7.88.1), which
    tags NC as ``nc`` instead of dropping it.

    Implements the design §1.3 pseudocode, plus one pilot-driven
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


# --- design §7.88.1: licence -> dataset subset ----------------------------------


@dataclass(frozen=True)
class LicenseDecision:
    """``subset`` is set iff ``status`` is REDISTRIBUTABLE. ``normalized`` is
    the canonical licence string the decision was made on (for the record)."""

    status: LicenseStatus
    subset: DatasetSubset | None
    normalized: str


_CC_LICENSE_URL = re.compile(r"creativecommons\.org/licenses/([a-z-]+)(?:/(\d+(?:\.\d+)*))?")
_VERSION_SUFFIX = re.compile(r"-\d+(?:\.\d+)*$")
_PUBLIC_DOMAIN = frozenset({"cc0", "cc-zero", "public-domain", "pd"})
_CC_ELEMENTS = frozenset({"nc", "nd", "sa"})


def normalize_license_id(license_id: str | None) -> str:
    """Canonical spelling of a licence string: lower case, ``-`` between
    words, version kept as a suffix (``"CC BY-NC-SA 4.0"`` ->
    ``"cc-by-nc-sa-4.0"``); creativecommons.org URLs map to the same form.
    Empty / missing -> ``""``."""
    text = (license_id or "").strip().lower()
    if not text:
        return ""
    if "creativecommons.org/publicdomain/zero" in text:
        return "cc0"
    if "creativecommons.org/publicdomain/mark" in text:
        return "public-domain"
    url = _CC_LICENSE_URL.search(text)
    if url:
        version = f"-{url.group(2)}" if url.group(2) else ""
        return f"cc-{url.group(1).strip('-')}{version}"
    text = re.sub(r"[\s_]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-")


def license_subset(license_id: str | None) -> LicenseDecision:
    """The subset a licence string puts a figure in, from the string alone
    (design §7.88): BY / BY-SA / CC0 / public domain -> ``core``; BY-NC /
    BY-NC-SA -> ``nc``; anything with ND -> EXCLUDED (we redistribute crops,
    i.e. derivatives); empty or unrecognised -> NEEDS_REVIEW (manual queue).
    An unrecognised CC element (``cc-by-foo``) is not guessed at."""
    normalized = normalize_license_id(license_id)
    base = _VERSION_SUFFIX.sub("", normalized)
    if base in _PUBLIC_DOMAIN:
        return LicenseDecision(LicenseStatus.REDISTRIBUTABLE, DatasetSubset.CORE, normalized)
    if base == "cc-by" or base.startswith("cc-by-"):
        elements = set(base[len("cc-by") :].split("-")) - {""}
        if elements <= _CC_ELEMENTS:
            if "nd" in elements:
                return LicenseDecision(LicenseStatus.EXCLUDED, None, normalized)
            subset = DatasetSubset.NC if "nc" in elements else DatasetSubset.CORE
            return LicenseDecision(LicenseStatus.REDISTRIBUTABLE, subset, normalized)
    return LicenseDecision(LicenseStatus.NEEDS_REVIEW, None, normalized)


def classify_figure_license(
    license_id: str | None,
    *,
    crossref_license_id: str | None = None,
    is_oa: bool | None = None,
) -> LicenseDecision:
    """The §1.3 gate with subsets (design §7.88.1), used by the collection
    pipeline. Same order as ``classify_license``:

    1. A redistributable licence (core or nc) wins outright.
    2. ``is_oa`` explicitly False -> EXCLUDED.
    3. No licence reported -> Crossref fallback; redistributable -> that
       decision, otherwise NEEDS_REVIEW (never EXCLUDED on this branch).
    4. A present licence: ND -> EXCLUDED, unrecognised -> NEEDS_REVIEW
       (the manual queue of §7.2/§7.88; either way the figure is not admitted).
    """
    decision = license_subset(license_id)
    if decision.status is LicenseStatus.REDISTRIBUTABLE:
        return decision
    if is_oa is False:
        return LicenseDecision(LicenseStatus.EXCLUDED, None, decision.normalized)
    if not decision.normalized:
        fallback = license_subset(crossref_license_id)
        if fallback.status is LicenseStatus.REDISTRIBUTABLE:
            return fallback
        return LicenseDecision(LicenseStatus.NEEDS_REVIEW, None, decision.normalized)
    return decision


class DriftKind(Enum):
    UNCHANGED = "unchanged"
    SUBSET_MOVE = "subset_move"  # core <-> nc: report only, a person moves it
    EXCLUDE = "exclude"  # now ND or closed: candidate for exclusion
    REVIEW = "review"  # now empty/unknown, or the recorded one is unrecognised
    LOOKUP_ERROR = "lookup_error"


@dataclass(frozen=True)
class LicenseDrift:
    kind: DriftKind
    recorded_subset: DatasetSubset | None
    current_subset: DatasetSubset | None


_LOOKUP_FAILURES = ("error:", "?")
_CLOSED = "closed"


def license_drift(recorded: str | None, current: str | None) -> LicenseDrift:
    """How a figure's licence today compares with the one recorded at
    collection (recheck_figure_licenses.py). Only classifies -- the caller
    reports; nothing is changed automatically (design §7.88)."""
    recorded_subset = license_subset(recorded).subset
    if current is not None and current.startswith(_LOOKUP_FAILURES):
        return LicenseDrift(DriftKind.LOOKUP_ERROR, recorded_subset, None)
    if current is not None and current.strip().lower() == _CLOSED:
        return LicenseDrift(DriftKind.EXCLUDE, recorded_subset, None)
    now = license_subset(current)
    if now.status is LicenseStatus.EXCLUDED:
        return LicenseDrift(DriftKind.EXCLUDE, recorded_subset, None)
    if now.status is LicenseStatus.NEEDS_REVIEW or recorded_subset is None:
        return LicenseDrift(DriftKind.REVIEW, recorded_subset, now.subset)
    if now.subset is recorded_subset:
        return LicenseDrift(DriftKind.UNCHANGED, recorded_subset, now.subset)
    return LicenseDrift(DriftKind.SUBSET_MOVE, recorded_subset, now.subset)
