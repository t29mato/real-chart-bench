"""Review M1 (design §7.88.1): the licence gate in select_verified_pairings
must not change what the committed registry scores -- every VERIFIED,
not-excluded entry today carries a licence admitted for its subset."""

from pathlib import Path

from real_chart_bench.adapter.verified_pairing_registry import load_registry
from real_chart_bench.domain.dataset_subset import DatasetSubset
from real_chart_bench.domain.verified_pairing import VerificationStatus
from real_chart_bench.usecase.real_image_gate import select_verified_pairings

REGISTRY = Path(__file__).resolve().parents[2] / "data/verified_pairs/registry.json"


def test_licence_gate_drops_no_committed_entry():
    registry = load_registry(REGISTRY)
    without_gate = [
        p
        for p in registry
        if p.status is VerificationStatus.VERIFIED
        and p.excluded_reason is None
        and p.subset is DatasetSubset.CORE
    ]
    assert select_verified_pairings(registry) == without_gate
    assert select_verified_pairings(registry, subset=DatasetSubset.NC) == []
