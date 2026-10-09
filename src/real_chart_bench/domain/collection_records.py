"""Dataset collection metadata records (design §1.4 schema).

Plain, immutable value objects — no I/O, no persistence format opinions
(adapters serialize these to JSON Lines/Parquet per design §6).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.dataset_split import DatasetSplit
from real_chart_bench.domain.dataset_subset import DatasetSubset
from real_chart_bench.domain.licensing import LicenseStatus, license_subset


@dataclass(frozen=True)
class PaperRecord:
    paper_id: str  # Starrydata SID
    doi: str
    title: str
    license_status: LicenseStatus
    license_id: str | None
    starrydata_paper_id: str | None = None
    # design §7.88.1: core / nc, decided from license_id (never by hand)
    subset: DatasetSubset = DatasetSubset.CORE

    def __post_init__(self) -> None:
        if self.license_status is not LicenseStatus.REDISTRIBUTABLE:
            return
        decided = license_subset(self.license_id)
        if decided.status is LicenseStatus.REDISTRIBUTABLE and decided.subset is not self.subset:
            raise ValueError(
                f"paper {self.paper_id}: subset {self.subset.value!r} contradicts "
                f"license_id {self.license_id!r} ({decided.subset.value!r}, design §7.88)"
            )


@dataclass(frozen=True)
class FigureRecord:
    figure_id: str  # Starrydata figure_id (composite figure, may hold >=1 panel)
    paper_id: str
    figure_reference: str  # raw Starrydata figure_name, e.g. "2(a)"
    image_uri: str | None = None
    split: DatasetSplit = DatasetSplit.PUBLIC
    subset: DatasetSubset = DatasetSubset.CORE  # the paper's (design §7.88.1)


@dataclass(frozen=True)
class GroundTruthCurve:
    curve_id: str
    figure_id: str
    x_values: tuple[float, ...]
    y_values: tuple[float, ...]
    x_scale: ScaleType
    series_label: str
    license: str = "CC BY 4.0"
    license_source: str = "manual (NIMS MDR)"
    quality_flags: tuple[str, ...] = field(default_factory=tuple)
