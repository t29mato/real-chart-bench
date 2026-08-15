"""assign_split: deterministic public/held_out assignment (design §7.5 —
external submission support is v2+, but the held-out slot must be reserved
from v0 onward).

Deterministic (hash-based on a stable key, not random-each-run) so that
re-running dataset construction doesn't reshuffle which papers are held out.
"""

from real_chart_bench.domain.dataset_split import DatasetSplit, assign_split


def test_same_key_always_gets_the_same_split():
    results = {assign_split("paper-123", held_out_ratio=0.2) for _ in range(20)}
    assert len(results) == 1


def test_held_out_ratio_zero_puts_everything_in_public():
    keys = [f"paper-{i}" for i in range(200)]
    splits = [assign_split(k, held_out_ratio=0.0) for k in keys]
    assert all(s is DatasetSplit.PUBLIC for s in splits)


def test_held_out_ratio_one_puts_everything_in_held_out():
    keys = [f"paper-{i}" for i in range(200)]
    splits = [assign_split(k, held_out_ratio=1.0) for k in keys]
    assert all(s is DatasetSplit.HELD_OUT for s in splits)


def test_held_out_ratio_is_approximately_respected_over_many_keys():
    keys = [f"paper-{i}" for i in range(2000)]
    held_out = sum(1 for k in keys if assign_split(k, held_out_ratio=0.2) is DatasetSplit.HELD_OUT)
    ratio = held_out / len(keys)
    assert 0.15 <= ratio <= 0.25


def test_out_of_range_ratio_raises():
    import pytest

    with pytest.raises(ValueError, match="held_out_ratio"):
        assign_split("paper-1", held_out_ratio=1.5)
    with pytest.raises(ValueError, match="held_out_ratio"):
        assign_split("paper-1", held_out_ratio=-0.1)
