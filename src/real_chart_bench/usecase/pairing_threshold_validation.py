"""Threshold-validation arithmetic for the automatic pairing rule
(docs/design/pairing-automation.md §5 C7, §6, §12;
docs/experiments/2026-10-10-pairing-threshold-validation.md).

The automatic pipeline proposes, per Starrydata figure, one plot frame with a
score S (C7 hit-rate), a margin M and a contrast. Human-verified pairings in
``data/verified_pairs/registry.json`` say which image each figure really is.
This module labels a proposal against that record and computes, for any
(S, M, contrast) threshold triple, how many proposals an auto-adopt lane
would take and how many of those are right. Pure functions only; running the
pipeline lives in scripts/eval/validate_pairing_thresholds.py.
"""

from __future__ import annotations

import itertools
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

CORRECT = "correct"  # the verified image
WRONG = "wrong"  # certainly not the figure's image
AMBIGUOUS = "ambiguous"  # a crop versus its (possible) source image: cannot be told apart
SIBLING = "sibling"  # an unverified Starrydata figure of the same reference as the image's owner
# (the same plot digitized a second time, e.g. log10(sigma) and sigma): the image does show it,
# but adopting it would duplicate the verified figure
UNLABELLED = "unlabelled"  # nothing in the registry decides it

_EPS = 1e-9


def _is_crop(path: str) -> bool:
    return "/crops/" in path


def normalise_reference(reference: str) -> str:
    """"Fig. 8(a)", "8a sigma" -> "8a": drop a leading "Fig", keep the first word
    (Starrydata's figure_name often carries the quantity after the number)."""
    words = re.sub(r"^fig(?:ure)?\.?\s*", "", reference.strip().lower()).split()
    return "".join(ch for ch in (words[0] if words else "") if ch.isalnum())


def label_assignment(
    key: tuple[str, str],
    image: str,
    *,
    verified: Mapping[tuple[str, str], str],
    rejected: Mapping[tuple[str, str], frozenset[str]],
    owners: Mapping[str, frozenset[str]],
    reference: str = "",
    owner_references: Mapping[str, frozenset[str]] | None = None,
) -> str:
    """Label the proposal "figure ``key`` (paper, figure) is drawn in ``image``".

    ``verified`` maps a verified figure to its image, ``rejected`` a rejected
    figure to the images its rejected entries named, ``owners`` an image to the
    figure ids verified on it. An unverified figure claiming an image that other
    figures own is wrong (one frame shows one figure); a crop and a full image of
    the same paper may or may not be the same picture and are not judged.
    ``rejected`` should hold only pairing-category rejections (an image- or
    GT-category rejection says nothing against the pairing). ``reference`` is
    the claimed figure's printed reference and ``owner_references`` the
    normalised references of each image's verified figures: a claim on an owned
    image under the owner's own reference is a sibling digitization, not an error."""
    expected = verified.get(key)
    if expected is not None:
        if image == expected:
            return CORRECT
        return AMBIGUOUS if _is_crop(image) != _is_crop(expected) else WRONG
    if image in rejected.get(key, frozenset()):
        return WRONG
    owned = owners.get(image)
    if owned and key[1] not in owned:
        same = normalise_reference(reference) in (owner_references or {}).get(image, frozenset())
        return SIBLING if reference and same else WRONG
    return UNLABELLED


@dataclass(frozen=True)
class LabelledAssignment:
    outcome: str
    S: float
    M: float
    contrast: float


@dataclass(frozen=True)
class AdoptStats:
    adopted: int
    correct: int
    wrong: int
    ambiguous: int  # crop-vs-full image and sibling digitizations together
    unlabelled: int
    recall: float  # correct / verified figures with an image
    precision: float | None  # correct / (correct + wrong); None if neither
    precision_pessimistic: float | None  # ambiguous counted as wrong
    error_upper95: float | None  # one-sided 95% upper bound on the error rate (pessimistic)


@dataclass(frozen=True)
class RuleResult:
    s_min: float
    m_min: float
    c_min: float
    stats: AdoptStats


def clopper_pearson_upper(errors: int, n: int, confidence: float = 0.95) -> float:
    """One-sided exact upper bound on a binomial error rate (bisection)."""
    if n <= 0 or errors >= n:
        return 1.0
    alpha = 1.0 - confidence

    def cdf(p: float) -> float:  # P(X <= errors | n, p)
        return sum(math.comb(n, i) * p**i * (1 - p) ** (n - i) for i in range(errors + 1))

    lo, hi = 0.0, 1.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if cdf(mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


def adopt_stats(
    items: Iterable[LabelledAssignment],
    s_min: float,
    m_min: float,
    c_min: float,
    *,
    n_verified: int,
) -> AdoptStats:
    taken = [
        a
        for a in items
        if a.S >= s_min - _EPS and a.M >= m_min - _EPS and a.contrast >= c_min - _EPS
    ]
    count = {k: sum(a.outcome == k for a in taken) for k in (CORRECT, WRONG, UNLABELLED)}
    count[AMBIGUOUS] = sum(a.outcome in (AMBIGUOUS, SIBLING) for a in taken)
    judged = count[CORRECT] + count[WRONG]
    judged_pess = judged + count[AMBIGUOUS]
    return AdoptStats(
        adopted=len(taken),
        correct=count[CORRECT],
        wrong=count[WRONG],
        ambiguous=count[AMBIGUOUS],
        unlabelled=count[UNLABELLED],
        recall=count[CORRECT] / n_verified if n_verified else 0.0,
        precision=count[CORRECT] / judged if judged else None,
        precision_pessimistic=count[CORRECT] / judged_pess if judged_pess else None,
        error_upper95=(
            clopper_pearson_upper(count[WRONG] + count[AMBIGUOUS], judged_pess)
            if judged_pess
            else None
        ),
    )


def sweep_adoption_rules(
    items: Sequence[LabelledAssignment],
    *,
    s_grid: Sequence[float],
    m_grid: Sequence[float],
    c_grid: Sequence[float],
    n_verified: int,
) -> list[RuleResult]:
    return [
        RuleResult(s, m, c, adopt_stats(items, s, m, c, n_verified=n_verified))
        for s, m, c in itertools.product(s_grid, m_grid, c_grid)
    ]
