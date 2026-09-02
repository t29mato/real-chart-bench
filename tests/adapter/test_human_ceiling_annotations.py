"""Loads data/human_ceiling/annotations/*.json into FigureAnnotation domain
records (see data/human_ceiling/FORMAT.md for the on-disk schema).

annotation_source is a mandatory, strictly-validated field -- this is the
adapter-level half of the project rule that an LLM judgment must never be
presented as a human one (the domain-level half is
domain.human_ceiling.require_human_ceiling()); a record with a missing or
unrecognized annotation_source must fail loudly, never silently default to
"human".
"""

from __future__ import annotations

import json

import pytest

from real_chart_bench.adapter.human_ceiling_annotations import load_annotation_files
from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.human_ceiling import AnnotationSource

_VALID_RECORD = {
    "paper_id": "4173",
    "figure_id": "20120",
    "annotation_source": "human",
    "annotator_id": "annotator-b",
    "annotated_at": "2026-09-10",
    "tool": "WebPlotDigitizer 4.7",
    "curves": [
        {"series_label": "1150 degC", "x": [773.0, 800.0], "y": [76.7, 78.0]},
    ],
}


def _write(tmp_path, name, record):
    directory = tmp_path / "annotations"
    directory.mkdir(exist_ok=True)
    (directory / name).write_text(json.dumps(record))
    return directory


def test_missing_directory_returns_no_annotations(tmp_path):
    assert load_annotation_files(tmp_path / "does-not-exist") == []


def test_empty_directory_returns_no_annotations(tmp_path):
    directory = tmp_path / "annotations"
    directory.mkdir()
    assert load_annotation_files(directory) == []


def test_parses_a_valid_record(tmp_path):
    directory = _write(tmp_path, "4173-20120__annotator-b.json", _VALID_RECORD)

    annotations = load_annotation_files(directory)

    assert len(annotations) == 1
    annotation = annotations[0]
    assert annotation.figure_id == "4173-20120"
    assert annotation.source is AnnotationSource.HUMAN
    assert annotation.annotator_id == "annotator-b"
    assert annotation.annotated_at == "2026-09-10"
    assert len(annotation.curves) == 1
    assert annotation.curves[0].series_label == "1150 degC"
    assert annotation.curves[0].x_values == (773.0, 800.0)


def test_parses_llm_and_automated_sources(tmp_path):
    llm_record = {**_VALID_RECORD, "annotation_source": "llm", "annotator_id": "gpt-digitizer"}
    automated_record = {
        **_VALID_RECORD,
        "annotation_source": "automated",
        "annotator_id": "cv-pipeline-v1",
    }
    directory = tmp_path / "annotations"
    directory.mkdir()
    (directory / "a.json").write_text(json.dumps(llm_record))
    (directory / "b.json").write_text(json.dumps(automated_record))

    annotations = load_annotation_files(directory)

    sources = {a.source for a in annotations}
    assert sources == {AnnotationSource.LLM, AnnotationSource.AUTOMATED}


def test_missing_annotation_source_raises_rather_than_defaulting_to_human(tmp_path):
    record = dict(_VALID_RECORD)
    del record["annotation_source"]
    directory = _write(tmp_path, "bad.json", record)

    with pytest.raises(ValueError, match="annotation_source"):
        load_annotation_files(directory)


def test_unrecognized_annotation_source_value_raises(tmp_path):
    record = {**_VALID_RECORD, "annotation_source": "definitely_human_i_promise"}
    directory = _write(tmp_path, "bad.json", record)

    with pytest.raises(ValueError, match="annotation_source"):
        load_annotation_files(directory)


def test_missing_annotator_id_raises(tmp_path):
    record = dict(_VALID_RECORD)
    del record["annotator_id"]
    directory = _write(tmp_path, "bad.json", record)

    with pytest.raises(ValueError, match="annotator_id"):
        load_annotation_files(directory)


def test_missing_annotated_at_raises(tmp_path):
    record = dict(_VALID_RECORD)
    del record["annotated_at"]
    directory = _write(tmp_path, "bad.json", record)

    with pytest.raises(ValueError, match="annotated_at"):
        load_annotation_files(directory)


def test_empty_curves_list_raises(tmp_path):
    record = {**_VALID_RECORD, "curves": []}
    directory = _write(tmp_path, "bad.json", record)

    with pytest.raises(ValueError, match="curves"):
        load_annotation_files(directory)


def test_mismatched_x_y_lengths_raise(tmp_path):
    record = {
        **_VALID_RECORD,
        "curves": [{"series_label": "s", "x": [1.0, 2.0], "y": [1.0]}],
    }
    directory = _write(tmp_path, "bad.json", record)

    with pytest.raises(ValueError, match="x.*y|y.*x"):
        load_annotation_files(directory)


def test_error_message_names_the_offending_file(tmp_path):
    record = dict(_VALID_RECORD)
    del record["annotation_source"]
    directory = _write(tmp_path, "clearly-named-file.json", record)

    with pytest.raises(ValueError, match="clearly-named-file.json"):
        load_annotation_files(directory)


def test_applies_x_scale_override_by_figure_id(tmp_path):
    directory = _write(tmp_path, "a.json", _VALID_RECORD)

    annotations = load_annotation_files(
        directory, x_scale_by_figure_id={"4173-20120": ScaleType.LOG}
    )

    assert annotations[0].curves[0].x_scale is ScaleType.LOG


def test_defaults_to_linear_x_scale_when_no_override_given(tmp_path):
    directory = _write(tmp_path, "a.json", _VALID_RECORD)

    annotations = load_annotation_files(directory)

    assert annotations[0].curves[0].x_scale is ScaleType.LINEAR


def test_series_label_defaults_to_empty_string_when_absent(tmp_path):
    record = {**_VALID_RECORD, "curves": [{"x": [1.0, 2.0], "y": [1.0, 2.0]}]}
    directory = _write(tmp_path, "a.json", record)

    annotations = load_annotation_files(directory)

    assert annotations[0].curves[0].series_label == ""


def test_loads_multiple_files_in_deterministic_sorted_order(tmp_path):
    directory = tmp_path / "annotations"
    directory.mkdir()
    for name, annotator in [("z.json", "zed"), ("a.json", "amy")]:
        (directory / name).write_text(
            json.dumps({**_VALID_RECORD, "annotator_id": annotator})
        )

    annotations = load_annotation_files(directory)

    assert [a.annotator_id for a in annotations] == ["amy", "zed"]
