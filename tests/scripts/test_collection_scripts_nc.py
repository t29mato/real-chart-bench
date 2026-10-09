"""Collection scripts and the nc subset (design §7.87, §7.88.1): PDF requests
stay >= 60 s apart, and new manifest rows write ``subset`` only for nc."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from real_chart_bench.domain.collection_records import FigureRecord, PaperRecord
from real_chart_bench.domain.dataset_subset import DatasetSubset
from real_chart_bench.domain.licensing import LicenseStatus

COLLECT = Path(__file__).resolve().parents[2] / "scripts" / "collect"


def load(name):
    spec = importlib.util.spec_from_file_location(name, COLLECT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_collect_spaces_pdf_requests_at_least_60_s():
    assert load("collect_v0_dataset").PDF_FETCH_DELAY_S >= 60


def test_refetch_minimum_pdf_gap_is_60_s():
    assert load("refetch_cc_by_pdfs").MIN_PDF_GAP_S == 60


@pytest.mark.parametrize("gap", ["0", "1", "59.9"])
def test_refetch_refuses_a_pdf_gap_under_60_s(gap, monkeypatch):
    mod = load("refetch_cc_by_pdfs")
    monkeypatch.setattr(sys, "argv", ["refetch", "--pdf-gap", gap])
    monkeypatch.setattr(mod, "run", lambda args: pytest.fail("must not start fetching"))
    with pytest.raises(SystemExit) as exc:
        mod.main()
    assert exc.value.code == 2


def test_refetch_accepts_60_s(monkeypatch):
    mod = load("refetch_cc_by_pdfs")
    seen = []
    monkeypatch.setattr(sys, "argv", ["refetch", "--pdf-gap", "60"])
    monkeypatch.setattr(mod, "run", lambda args: seen.append(args.pdf_gap))
    mod.main()
    assert seen == [60.0]


def _paper(license_id, subset):
    return PaperRecord(
        paper_id="1", doi="10.1/x", title="t",
        license_status=LicenseStatus.REDISTRIBUTABLE, license_id=license_id, subset=subset,
    )


def test_new_core_rows_carry_no_subset_key():
    mod = load("collect_v0_dataset")
    paper = mod.paper_manifest_entry(
        _paper("cc-by", DatasetSubset.CORE),
        n_figures=1, n_curves=2, n_extracted_images=None, pdf_status=None,
    )
    figure = mod.figure_manifest_entry(
        FigureRecord(figure_id="9", paper_id="1", figure_reference="2")
    )
    assert "subset" not in paper
    assert "subset" not in figure
    # same keys, same order as the rows already in papers.json
    assert list(paper) == [
        "paper_id", "doi", "license_id", "n_figures", "n_curves",
        "n_extracted_images", "pdf_status",
    ]


def test_new_nc_rows_say_nc():
    mod = load("collect_v0_dataset")
    paper = mod.paper_manifest_entry(
        _paper("cc-by-nc", DatasetSubset.NC),
        n_figures=1, n_curves=2, n_extracted_images=3, pdf_status="ok",
    )
    figure = mod.figure_manifest_entry(
        FigureRecord(figure_id="9", paper_id="1", figure_reference="2", subset=DatasetSubset.NC)
    )
    assert paper["subset"] == "nc"
    assert figure["subset"] == "nc"
