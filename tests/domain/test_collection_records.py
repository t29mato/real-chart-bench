from real_chart_bench.domain.collection_records import (
    FigureRecord,
    GroundTruthCurve,
    PaperRecord,
)
from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.dataset_split import DatasetSplit
from real_chart_bench.domain.licensing import LicenseStatus


def test_paper_record_holds_license_status():
    paper = PaperRecord(
        paper_id="6061",
        doi="10.1/example",
        title="Example paper",
        license_status=LicenseStatus.REDISTRIBUTABLE,
        license_id="cc-by",
    )
    assert paper.license_status is LicenseStatus.REDISTRIBUTABLE


def test_figure_record_defaults_split_to_public():
    figure = FigureRecord(figure_id="5190", paper_id="6061", figure_reference="2(a)")
    assert figure.split is DatasetSplit.PUBLIC


def test_ground_truth_curve_carries_starrydata_license():
    curve = GroundTruthCurve(
        curve_id="6061-5190-0",
        figure_id="5190",
        x_values=(1.0, 2.0),
        y_values=(3.0, 4.0),
        x_scale=ScaleType.LINEAR,
        series_label="",
        license="CC BY 4.0",
        license_source="manual (NIMS MDR)",
    )
    assert curve.license == "CC BY 4.0"
    assert len(curve.x_values) == 2


def test_records_are_immutable():
    import pytest

    paper = PaperRecord(
        paper_id="1", doi="10.1/x", title="t",
        license_status=LicenseStatus.NEEDS_REVIEW, license_id=None,
    )
    with pytest.raises(AttributeError):
        paper.title = "mutated"  # type: ignore[misc]
