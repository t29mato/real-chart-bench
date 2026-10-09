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


_CC_LICENSE_URL = re.compile(
    r"creativecommons\.org/licenses/([a-z-]+)(?:/(\d+(?:\.\d+)*))?(?:/(igo))?"
)
_VERSION_TOKEN = re.compile(r"\d+(?:\.\d+)*")
# typographic hyphens and dashes that turn up in licence strings
_DASHES = re.compile("[‐-―−]")
_PUBLIC_DOMAIN = frozenset({"cc0", "cc-zero", "public-domain", "pd"})
_CC0 = frozenset({"cc0", "cc-zero"})
_CC_ELEMENTS = frozenset({"nc", "nd", "sa"})
# the versions Creative Commons actually published; anything else is a typo or
# a value we do not understand, and goes to review rather than being guessed
_CC_VERSIONS = frozenset({"1.0", "2.0", "2.5", "3.0", "4.0"})
_CC0_VERSIONS = frozenset({"1.0"})
_IGO_VERSION = "3.0"


def _split_suffix(normalized: str) -> tuple[str, str | None, bool]:
    """``"cc-by-nc-3.0-igo"`` -> (``"cc-by-nc"``, ``"3.0"``, True)."""
    tokens = normalized.split("-")
    igo = len(tokens) > 1 and tokens[-1] == "igo"
    if igo:
        tokens.pop()
    version = None
    if len(tokens) > 1 and _VERSION_TOKEN.fullmatch(tokens[-1]):
        version = tokens.pop()
    return "-".join(tokens), version, igo


def _version_is_known(base: str, version: str | None, igo: bool) -> bool:
    if igo and version != _IGO_VERSION:
        return False
    if version is None:
        return True
    if base in _CC0:
        return version in _CC0_VERSIONS
    if base in _PUBLIC_DOMAIN:
        return False
    return version in _CC_VERSIONS


def normalize_license_id(license_id: str | None) -> str:
    """Canonical spelling of a licence string: lower case, ``-`` between
    words, version kept as a suffix (``"CC BY-NC-SA 4.0"`` ->
    ``"cc-by-nc-sa-4.0"``); creativecommons.org URLs map to the same form.
    Empty / missing -> ``""``."""
    text = _DASHES.sub("-", (license_id or "").strip().lower())
    if not text:
        return ""
    if "creativecommons.org/publicdomain/zero" in text:
        return "cc0"
    if "creativecommons.org/publicdomain/mark" in text:
        return "public-domain"
    url = _CC_LICENSE_URL.search(text)
    if url:
        version = f"-{url.group(2)}" if url.group(2) else ""
        igo = "-igo" if url.group(3) else ""
        return f"cc-{url.group(1).strip('-')}{version}{igo}"
    text = re.sub(r"[\s_/]+", "-", text)
    return re.sub(r"-{2,}", "-", text).strip("-")


def license_subset(license_id: str | None) -> LicenseDecision:
    """The subset a licence string puts a figure in, from the string alone
    (design §7.88): BY / BY-SA / CC0 / public domain -> ``core``; BY-NC /
    BY-NC-SA -> ``nc``; anything with ND -> EXCLUDED (we redistribute crops,
    i.e. derivatives); empty or unrecognised -> NEEDS_REVIEW (manual queue).
    An unrecognised CC element (``cc-by-foo``) or a version CC never published
    (``cc-by-99``) is not guessed at: review. The 3.0 IGO ports are accepted."""
    normalized = normalize_license_id(license_id)
    base, version, igo = _split_suffix(normalized)
    review = LicenseDecision(LicenseStatus.NEEDS_REVIEW, None, normalized)
    if base in _PUBLIC_DOMAIN:
        if not _version_is_known(base, version, igo):
            return review
        return LicenseDecision(LicenseStatus.REDISTRIBUTABLE, DatasetSubset.CORE, normalized)
    if base == "cc-by" or base.startswith("cc-by-"):
        elements = set(base[len("cc-by") :].split("-")) - {""}
        if elements <= _CC_ELEMENTS:
            if "nd" in elements:  # out whatever the version says
                return LicenseDecision(LicenseStatus.EXCLUDED, None, normalized)
            if not _version_is_known(base, version, igo):
                return review
            subset = DatasetSubset.NC if "nc" in elements else DatasetSubset.CORE
            return LicenseDecision(LicenseStatus.REDISTRIBUTABLE, subset, normalized)
    return review


def licence_admitted_for(license_id: str | None, subset: DatasetSubset) -> bool:
    """True iff the licence admits a figure to ``subset`` (design §7.88.1) --
    the gate for what is distributed and scored. ND, unknown and missing
    licences admit to no subset. A licence that admits to the *other* subset
    raises: the subset follows the licence and is never set by hand."""
    decided = license_subset(license_id)
    if decided.status is not LicenseStatus.REDISTRIBUTABLE:
        return False
    if decided.subset is not subset:
        raise ValueError(
            f"subset {subset.value!r} contradicts licence {license_id!r} "
            f"(which admits to {decided.subset.value!r})"
        )
    return True


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
