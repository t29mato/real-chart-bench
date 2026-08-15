from real_chart_bench.domain.collection_records import PaperRecord
from real_chart_bench.domain.dataset_split import DatasetSplit
from real_chart_bench.domain.licensing import LicenseStatus
from real_chart_bench.usecase.build_ground_truth_manifest import build_ground_truth_for_paper
from real_chart_bench.usecase.starrydata_ingestion import ParsedCurveRow


def _paper(paper_id="6061"):
    return PaperRecord(
        paper_id=paper_id,
        doi="10.1000/example",
        title="Example",
        license_status=LicenseStatus.REDISTRIBUTABLE,
        license_id="cc-by",
    )


def _row(figure_id="5190", figure_name="2(a)", sid="6061"):
    return ParsedCurveRow(
        sid=sid,
        doi="10.1000/example",
        figure_id=figure_id,
        figure_name=figure_name,
        series_label="Seebeck coefficient (V*K^-1)",
        x_values=(1.0, 2.0),
        y_values=(3.0, 4.0),
    )


def test_groups_curves_by_figure_id():
    rows = [_row(figure_id="5190"), _row(figure_id="5190"), _row(figure_id="5191")]

    figures, curves = build_ground_truth_for_paper(_paper(), rows, held_out_ratio=0.0)

    assert {f.figure_id for f in figures} == {"5190", "5191"}
    assert len(curves) == 3


def test_curve_ids_are_stable_and_unique():
    rows = [_row(figure_id="5190"), _row(figure_id="5190")]

    _, curves = build_ground_truth_for_paper(_paper(), rows, held_out_ratio=0.0)

    assert len({c.curve_id for c in curves}) == 2


def test_split_is_shared_across_all_figures_of_a_paper():
    rows = [_row(figure_id="5190"), _row(figure_id="5191")]

    figures, _ = build_ground_truth_for_paper(_paper(), rows, held_out_ratio=1.0)

    assert all(f.split is DatasetSplit.HELD_OUT for f in figures)


def test_curves_carry_the_cc_by_4_starrydata_license():
    _, curves = build_ground_truth_for_paper(_paper(), [_row()], held_out_ratio=0.0)

    assert curves[0].license == "CC BY 4.0"
    assert curves[0].license_source == "manual (NIMS MDR)"


def test_non_redistributable_paper_raises():
    import pytest

    paper = PaperRecord(
        paper_id="1", doi="x", title="t",
        license_status=LicenseStatus.NEEDS_REVIEW, license_id=None,
    )
    with pytest.raises(ValueError, match="REDISTRIBUTABLE"):
        build_ground_truth_for_paper(paper, [_row(sid="1")], held_out_ratio=0.0)


def test_empty_curve_rows_yields_no_figures_or_curves():
    figures, curves = build_ground_truth_for_paper(_paper(), [], held_out_ratio=0.0)
    assert figures == ()
    assert curves == ()
