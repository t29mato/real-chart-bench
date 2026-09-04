"""TDD for adapter/starry_digitizer_import.py -- converting a starry-digitizer
"Export Project" .zip (project.json + image) into a schema-valid
data/human_ceiling/annotations/*.json record.

Fixture axis set below is deliberately axis-aligned (x1/x2 share a y pixel,
y1/y2 share an x pixel) so the expected values can be hand-computed:
  x: pixel 100 -> 0, pixel 600 -> 10 (linear)
  y: pixel 500 -> 0, pixel 100 -> 100 (linear, inverted like a real chart)
"""

from __future__ import annotations

import io
import json
import zipfile

import pytest

from real_chart_bench.adapter.human_ceiling_annotations import load_annotation_files
from real_chart_bench.adapter.starry_digitizer_import import (
    StarryDigitizerImportError,
    convert_project_to_annotation,
    load_project_from_zip,
)


def _axis(value, x_px, y_px):
    return {"name": "", "value": value, "coord": {"xPx": x_px, "yPx": y_px}}


def _linear_axis_set(axis_set_id=1, *, consider_graph_tilt=False):
    return {
        "id": axis_set_id,
        "name": "axisSet1",
        "x1": _axis(0, 100, 500),
        "x2": _axis(10, 600, 500),
        "y1": _axis(0, 100, 500),
        "y2": _axis(100, 100, 100),
        "xIsLogScale": False,
        "yIsLogScale": False,
        "considerGraphTilt": consider_graph_tilt,
        "pointMode": "auto",
        "isVisible": True,
    }


def _dataset(dataset_id, axis_set_id, points, name="series A"):
    return {
        "id": dataset_id,
        "name": name,
        "axisSetId": axis_set_id,
        "points": [{"id": i, "xPx": px, "yPx": py} for i, (px, py) in enumerate(points)],
        "visiblePointIds": [],
        "manuallyAddedPointIds": [],
    }


def _project(axis_sets, datasets):
    return {
        "version": "1.11.2",
        "timestamp": "2026-09-04T00:00:00.000Z",
        "axisSets": axis_sets,
        "activeAxisSetId": axis_sets[0]["id"] if axis_sets else 1,
        "datasets": datasets,
        "activeDatasetId": datasets[0]["id"] if datasets else 1,
        "canvasHandler": {"scale": 1, "manualMode": "point"},
    }


_BASE_KWARGS = dict(
    paper_id="4173",
    figure_id="20120",
    annotator_id="annotator-b",
    annotated_at="2026-09-04",
)


def test_linear_axes_convert_pixel_corners_to_known_values():
    axis_set = _linear_axis_set()
    dataset = _dataset(1, 1, [(100, 500), (600, 100)])
    project = _project([axis_set], [dataset])

    record = convert_project_to_annotation(project, **_BASE_KWARGS)

    assert record["curves"][0]["x"] == pytest.approx([0.0, 10.0])
    assert record["curves"][0]["y"] == pytest.approx([0.0, 100.0])


def test_linear_axes_convert_midpoint():
    axis_set = _linear_axis_set()
    dataset = _dataset(1, 1, [(350, 300)])  # midpoint of both axes
    project = _project([axis_set], [dataset])

    record = convert_project_to_annotation(project, **_BASE_KWARGS)

    assert record["curves"][0]["x"] == pytest.approx([5.0])
    assert record["curves"][0]["y"] == pytest.approx([50.0])


def test_log_scale_axis_interpolates_in_log_space():
    axis_set = _linear_axis_set()
    axis_set["x1"] = _axis(1, 100, 500)
    axis_set["x2"] = _axis(100, 300, 500)
    axis_set["xIsLogScale"] = True
    dataset = _dataset(1, 1, [(200, 500)])  # halfway in pixel == log10 midpoint
    project = _project([axis_set], [dataset])

    record = convert_project_to_annotation(project, **_BASE_KWARGS)

    assert record["curves"][0]["x"] == pytest.approx([10.0])


def test_graph_tilt_correction_is_identity_for_an_axis_aligned_rectangle():
    axis_set = _linear_axis_set(consider_graph_tilt=True)
    dataset = _dataset(1, 1, [(350, 300)])
    project = _project([axis_set], [dataset])

    record = convert_project_to_annotation(project, **_BASE_KWARGS)

    assert record["curves"][0]["x"] == pytest.approx([5.0])
    assert record["curves"][0]["y"] == pytest.approx([50.0])


def test_multiple_datasets_become_multiple_curves_with_dataset_names():
    axis_set = _linear_axis_set()
    ds1 = _dataset(1, 1, [(100, 500)], name="1150 degC")
    ds2 = _dataset(2, 1, [(600, 100)], name="1200 degC")
    project = _project([axis_set], [ds1, ds2])

    record = convert_project_to_annotation(project, **_BASE_KWARGS)

    labels = [c["series_label"] for c in record["curves"]]
    assert labels == ["1150 degC", "1200 degC"]


def test_empty_dataset_is_skipped_not_emitted_as_empty_curve():
    axis_set = _linear_axis_set()
    ds1 = _dataset(1, 1, [(100, 500)], name="has points")
    ds2 = _dataset(2, 1, [], name="empty")
    project = _project([axis_set], [ds1, ds2])

    record = convert_project_to_annotation(project, **_BASE_KWARGS)

    labels = [c["series_label"] for c in record["curves"]]
    assert labels == ["has points"]


def test_unnamed_dataset_falls_back_to_dataset_id_label():
    axis_set = _linear_axis_set()
    dataset = _dataset(1, 1, [(100, 500)], name="")
    project = _project([axis_set], [dataset])

    record = convert_project_to_annotation(project, **_BASE_KWARGS)

    assert record["curves"][0]["series_label"] == "dataset-1"


def test_all_datasets_empty_raises():
    axis_set = _linear_axis_set()
    dataset = _dataset(1, 1, [], name="empty")
    project = _project([axis_set], [dataset])

    with pytest.raises(StarryDigitizerImportError, match="no dataset"):
        convert_project_to_annotation(project, **_BASE_KWARGS)


def test_dataset_referencing_unknown_axis_set_raises():
    axis_set = _linear_axis_set(axis_set_id=1)
    dataset = _dataset(1, 999, [(100, 500)])
    project = _project([axis_set], [dataset])

    with pytest.raises(StarryDigitizerImportError, match="axisSetId"):
        convert_project_to_annotation(project, **_BASE_KWARGS)


def test_uncalibrated_axis_point_raises():
    axis_set = _linear_axis_set()
    axis_set["y2"] = _axis(100, -999, -999)  # never placed
    dataset = _dataset(1, 1, [(100, 500)])
    project = _project([axis_set], [dataset])

    with pytest.raises(StarryDigitizerImportError, match="never placed"):
        convert_project_to_annotation(project, **_BASE_KWARGS)


def test_degenerate_axis_same_value_on_both_points_raises():
    axis_set = _linear_axis_set()
    axis_set["x2"] = _axis(0, 600, 500)  # same value as x1
    dataset = _dataset(1, 1, [(100, 500)])
    project = _project([axis_set], [dataset])

    with pytest.raises(StarryDigitizerImportError, match="same value"):
        convert_project_to_annotation(project, **_BASE_KWARGS)


def test_notes_included_only_when_given():
    axis_set = _linear_axis_set()
    dataset = _dataset(1, 1, [(100, 500)])
    project = _project([axis_set], [dataset])

    without_notes = convert_project_to_annotation(project, **_BASE_KWARGS)
    with_notes = convert_project_to_annotation(project, notes="ambiguous overlap", **_BASE_KWARGS)

    assert "notes" not in without_notes
    assert with_notes["notes"] == "ambiguous overlap"


def test_default_tool_is_starry_digitizer():
    axis_set = _linear_axis_set()
    dataset = _dataset(1, 1, [(100, 500)])
    project = _project([axis_set], [dataset])

    record = convert_project_to_annotation(project, **_BASE_KWARGS)

    assert record["tool"] == "starry-digitizer"


def test_annotation_source_defaults_to_human_but_is_overridable():
    axis_set = _linear_axis_set()
    dataset = _dataset(1, 1, [(100, 500)])
    project = _project([axis_set], [dataset])

    default_record = convert_project_to_annotation(project, **_BASE_KWARGS)
    llm_record = convert_project_to_annotation(
        project, annotation_source="llm", **_BASE_KWARGS
    )

    assert default_record["annotation_source"] == "human"
    assert llm_record["annotation_source"] == "llm"


def test_converted_record_round_trips_through_the_annotation_loader(tmp_path):
    axis_set = _linear_axis_set()
    dataset = _dataset(1, 1, [(100, 500), (600, 100)], name="1150 degC")
    project = _project([axis_set], [dataset])

    record = convert_project_to_annotation(project, **_BASE_KWARGS)

    directory = tmp_path / "annotations"
    directory.mkdir()
    (directory / "4173-20120__annotator-b.json").write_text(json.dumps(record))

    annotations = load_annotation_files(directory)

    assert len(annotations) == 1
    assert annotations[0].figure_id == "4173-20120"
    assert annotations[0].annotator_id == "annotator-b"
    assert annotations[0].curves[0].series_label == "1150 degC"
    assert annotations[0].curves[0].x_values == pytest.approx((0.0, 10.0))


def test_load_project_from_zip_reads_project_json(tmp_path):
    axis_set = _linear_axis_set()
    dataset = _dataset(1, 1, [(100, 500)])
    project = _project([axis_set], [dataset])

    zip_path = tmp_path / "sd-export.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("project.json", json.dumps(project))
        zf.writestr("image.png", b"not-a-real-png")

    loaded = load_project_from_zip(zip_path)

    assert loaded == project


def test_load_project_from_zip_without_project_json_raises(tmp_path):
    zip_path = tmp_path / "not-a-project.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("readme.txt", "oops")

    with pytest.raises(StarryDigitizerImportError, match="project.json"):
        load_project_from_zip(zip_path)


def test_load_project_from_zip_accepts_file_like_path(tmp_path):
    # zipfile.ZipFile also accepts an in-memory buffer; sanity-check our
    # helper's contract still works when handed a real path (the CLI's
    # actual use case), not just to exercise io.BytesIO.
    axis_set = _linear_axis_set()
    dataset = _dataset(1, 1, [(100, 500)])
    project = _project([axis_set], [dataset])

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("project.json", json.dumps(project))
    zip_path = tmp_path / "sd-export2.zip"
    zip_path.write_bytes(buf.getvalue())

    assert load_project_from_zip(zip_path) == project
