"""Ground truth = Starrydata's digitization (ground_truth.json) plus series the
figure draws but Starrydata never digitized -- reference curves from other
works, a comparison sample -- digitized for this benchmark and kept in their
own files with their provenance (ground_truth_supplement/). These tests pin
the merge, the provenance tag, and the dataset_version suffix that keeps a
score on the supplemented ground truth from being compared with one on the
original.
"""

import json

import pytest

from real_chart_bench.adapter.ground_truth_store import (
    GroundTruthSupplementError,
    ground_truth_revision,
    load_ground_truth,
)


def _write_base(tmp_path):
    base = tmp_path / "ground_truth.json"
    base.write_text(
        json.dumps({"200": [{"x": [1, 2], "y": [3, 4], "prop_y": "Seebeck coefficient"}]})
    )
    return base


def _write_supplement(directory, name="10-200.json", **overrides):
    directory.mkdir(exist_ok=True)
    record = {
        "paper_id": "10",
        "figure_id": "200",
        "digitized_by": "t29mato",
        "digitized_at": "2026-10-02",
        "tool": "starry-digitizer",
        "reason": "reference curve from another work, drawn in the figure",
        "curves": [{"series_label": "TiCoSb ref", "x": [1, 2], "y": [5, 6]}],
    }
    record.update(overrides)
    (directory / name).write_text(json.dumps(record))


def test_without_a_supplement_it_is_the_starrydata_ground_truth(tmp_path):
    gt = load_ground_truth(_write_base(tmp_path), tmp_path / "missing_dir")

    assert gt == {
        "200": [{"x": [1, 2], "y": [3, 4], "prop_y": "Seebeck coefficient", "source": "starrydata"}]
    }


def test_supplement_curves_are_appended_with_their_provenance(tmp_path):
    sup = tmp_path / "sup"
    _write_supplement(sup)

    gt = load_ground_truth(_write_base(tmp_path), sup)

    assert [row["source"] for row in gt["200"]] == ["starrydata", "real-chart-bench-supplement"]
    added = gt["200"][1]
    assert added["x"] == [1, 2] and added["y"] == [5, 6]
    assert added["prop_y"] == "TiCoSb ref"
    assert added["digitized_by"] == "t29mato"


def test_a_supplement_for_a_figure_without_starrydata_rows_is_an_error(tmp_path):
    sup = tmp_path / "sup"
    _write_supplement(sup, figure_id="999", name="10-999.json")

    with pytest.raises(GroundTruthSupplementError, match="999"):
        load_ground_truth(_write_base(tmp_path), sup)


@pytest.mark.parametrize(
    "overrides",
    [
        {"digitized_by": ""},
        {"reason": ""},
        {"curves": []},
        {"curves": [{"series_label": "a", "x": [1, 2], "y": [1]}]},
        {"curves": [{"series_label": "a", "x": [], "y": []}]},
    ],
)
def test_incomplete_records_are_rejected_not_skipped(tmp_path, overrides):
    sup = tmp_path / "sup"
    _write_supplement(sup, **overrides)

    with pytest.raises(GroundTruthSupplementError):
        load_ground_truth(_write_base(tmp_path), sup)


def test_revision_is_empty_without_a_supplement(tmp_path):
    assert ground_truth_revision(tmp_path / "missing_dir") == ""
    (tmp_path / "empty").mkdir()
    assert ground_truth_revision(tmp_path / "empty") == ""


def test_revision_changes_whenever_a_supplement_changes(tmp_path):
    sup = tmp_path / "sup"
    _write_supplement(sup)
    first = ground_truth_revision(sup)

    _write_supplement(sup, curves=[{"series_label": "TiCoSb ref", "x": [1, 2], "y": [5, 7]}])
    second = ground_truth_revision(sup)

    assert first.startswith("-gtsup1-") and second.startswith("-gtsup1-")
    assert first != second
