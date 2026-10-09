"""ATTRIBUTION.md per dataset subset (design §7.33, §7.88.1): core keeps its
file and wording; nc gets its own file in its own directory, with a
non-commercial notice and the share-alike condition of NC-SA figures."""

from real_chart_bench.domain.dataset_subset import DatasetSubset
from real_chart_bench.domain.verified_pairing import VerificationStatus, VerifiedPairing
from real_chart_bench.usecase.build_attribution import (
    CORE_HEADER,
    attribution_rows,
    render_attribution,
)


def _pairing(figure_id, *, license_id="cc-by", image_path=None, status=None, subset=None):
    nc = subset is DatasetSubset.NC
    root = "verified_pairs_nc" if nc else "verified_pairs"
    return VerifiedPairing(
        paper_id="7",
        figure_id=figure_id,
        image_path=image_path or f"data/{root}/crops/7/fig{figure_id}.png",
        panel_label=None,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        status=status or VerificationStatus.VERIFIED,
        verified_at="2026-10-10",
        evidence="t",
        license_id=license_id,
        subset=subset or DatasetSubset.CORE,
    )


PAPERS = {"7": {"paper_id": "7", "doi": "10.1/abc"}}


def _rows(pairings, subset, exists=lambda path: True):
    return attribution_rows(pairings, PAPERS, subset=subset, file_exists=exists)


def test_core_row_format_is_unchanged():
    rows = _rows([_pairing("1")], DatasetSubset.CORE)
    assert rows == [
        "| [10.1/abc](https://doi.org/10.1/abc) | cc-by "
        "| `data/verified_pairs/crops/7/fig1.png` | yes |"
    ]


def test_core_and_nc_rows_never_mix():
    pairings = [
        _pairing("1"),
        _pairing("2", license_id="cc-by-nc", subset=DatasetSubset.NC),
    ]
    core = _rows(pairings, DatasetSubset.CORE)
    nc = _rows(pairings, DatasetSubset.NC)
    assert len(core) == 1 and "verified_pairs/crops" in core[0]
    assert len(nc) == 1 and "verified_pairs_nc/crops" in nc[0]


def test_nc_row_flags_share_alike():
    rows = _rows(
        [
            _pairing("1", license_id="cc-by-nc-sa", subset=DatasetSubset.NC),
            _pairing("2", license_id="cc-by-nc", subset=DatasetSubset.NC),
        ],
        DatasetSubset.NC,
    )
    assert rows == [
        "| [10.1/abc](https://doi.org/10.1/abc) | cc-by-nc "
        "| `data/verified_pairs_nc/crops/7/fig2.png` | yes | no |",
        "| [10.1/abc](https://doi.org/10.1/abc) | cc-by-nc-sa "
        "| `data/verified_pairs_nc/crops/7/fig1.png` | yes | yes |",
    ]


def test_files_not_on_disk_and_unverified_and_raw_names_are_skipped():
    pairings = [
        _pairing("1"),
        _pairing("2", status=VerificationStatus.REJECTED),
        _pairing("3", image_path="p03_embedded_4.jpg"),
    ]
    rows = _rows(pairings, DatasetSubset.CORE, exists=lambda p: not p.endswith("fig1.png"))
    assert rows == []


def test_core_document_is_the_existing_header_plus_rows():
    text = render_attribution(DatasetSubset.CORE, ["| r |"])
    assert text == CORE_HEADER + "| r |\n"


def test_nc_document_states_non_commercial_and_share_alike():
    text = render_attribution(DatasetSubset.NC, ["| r |"])
    assert "non-commercial" in text.lower()
    assert "share" in text.lower() and "alike" in text.lower()
    assert "verified_pairs_nc" in text
    assert text.endswith("| r |\n")
