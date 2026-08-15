"""Deterministic public/held-out split assignment (design §7.5).

External submission review is v2+ scope, but the held-out slot must be
reserved from v0 onward so it isn't retrofitted later against a dataset that
was already fully public. Hash-based (not random.random()) so re-running
dataset construction is reproducible: the same paper always lands in the
same split. Pure function — no I/O.
"""

from __future__ import annotations

import hashlib
from enum import Enum


class DatasetSplit(Enum):
    PUBLIC = "public"
    HELD_OUT = "held_out"


def assign_split(key: str, *, held_out_ratio: float) -> DatasetSplit:
    if not 0.0 <= held_out_ratio <= 1.0:
        raise ValueError(f"held_out_ratio must be in [0, 1], got {held_out_ratio!r}")

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    # Use the first 4 bytes as a uniform-ish integer in [0, 2**32).
    bucket = int.from_bytes(digest[:4], "big") / 2**32
    return DatasetSplit.HELD_OUT if bucket < held_out_ratio else DatasetSplit.PUBLIC
