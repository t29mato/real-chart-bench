import json

from real_chart_bench.adapter.verified_pairing_registry import parse_registry
from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.verified_pairing import VerificationStatus


def test_parses_a_verified_entry_with_calibration():
    raw = [
        {
            "paper_id": "18759",
            "figure_id": "12217",
            "image_path": "p04_embedded_4.jpg",
            "panel_label": "a",
            "x_range": [200.0, 500.0],
            "y_range": [25000.0, 135000.0],
            "x_scale": "linear",
            "status": "verified",
            "verified_at": "2026-08-16",
            "evidence": "cross-checked",
        }
    ]

    registry = parse_registry(raw)

    assert len(registry) == 1
    entry = registry[0]
    assert entry.status is VerificationStatus.VERIFIED
    assert entry.x_range == (200.0, 500.0)
    assert entry.x_scale is ScaleType.LINEAR


def test_parses_a_rejected_entry_with_null_calibration():
    raw = [
        {
            "paper_id": "47139",
            "figure_id": "48697",
            "image_path": "p05_embedded_7.jpg",
            "panel_label": "b",
            "x_range": None,
            "y_range": None,
            "status": "rejected",
            "verified_at": "2026-08-16",
            "evidence": "numeric mismatch",
        }
    ]

    registry = parse_registry(raw)

    assert registry[0].status is VerificationStatus.REJECTED
    assert registry[0].x_range is None


def test_parses_log_x_scale():
    raw = [
        {
            "paper_id": "1",
            "figure_id": "1",
            "image_path": "a.jpg",
            "panel_label": None,
            "x_range": [1.0, 100.0],
            "y_range": [0.0, 1.0],
            "x_scale": "log",
            "status": "verified",
            "verified_at": "2026-08-16",
            "evidence": "x",
        }
    ]

    assert parse_registry(raw)[0].x_scale is ScaleType.LOG


def test_parses_log_y_scale():
    raw = [
        {
            "paper_id": "47534",
            "figure_id": "49581",
            "image_path": "p03_embedded_2.jpg",
            "panel_label": None,
            "x_range": [873.15, 1173.15],
            "y_range": [10.0, 1000.0],
            "y_scale": "log",
            "status": "verified",
            "verified_at": "2026-08-19",
            "evidence": "y",
        }
    ]

    assert parse_registry(raw)[0].y_scale is ScaleType.LOG


def test_parses_license_id_when_present():
    raw = [
        {
            "paper_id": "18759",
            "figure_id": "12217",
            "image_path": "p04_embedded_4.jpg",
            "panel_label": "a",
            "x_range": [200.0, 500.0],
            "y_range": [25000.0, 135000.0],
            "status": "verified",
            "verified_at": "2026-08-16",
            "evidence": "x",
            "license_id": "cc-by",
        }
    ]

    assert parse_registry(raw)[0].license_id == "cc-by"


def test_license_id_defaults_to_none_when_absent():
    raw = [
        {
            "paper_id": "1",
            "figure_id": "1",
            "image_path": "a.jpg",
            "panel_label": None,
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
            "status": "verified",
            "verified_at": "2026-08-16",
            "evidence": "x",
        }
    ]

    assert parse_registry(raw)[0].license_id is None


def test_y_scale_defaults_to_linear_when_absent():
    raw = [
        {
            "paper_id": "1",
            "figure_id": "1",
            "image_path": "a.jpg",
            "panel_label": None,
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
            "status": "verified",
            "verified_at": "2026-08-16",
            "evidence": "x",
        }
    ]

    assert parse_registry(raw)[0].y_scale is ScaleType.LINEAR


def test_empty_registry_file_parses_to_empty_list():
    assert parse_registry([]) == []


def test_excluded_reason_defaults_to_none_when_absent():
    raw = [
        {
            "paper_id": "1",
            "figure_id": "1",
            "image_path": "a.jpg",
            "panel_label": None,
            "x_range": [0.0, 1.0],
            "y_range": [0.0, 1.0],
            "x_scale": "linear",
            "status": "verified",
            "verified_at": "2026-08-16",
            "evidence": "x",
        }
    ]

    assert parse_registry(raw)[0].excluded_reason is None


def test_excluded_reason_parses_when_present():
    raw = [
        {
            "paper_id": "47534",
            "figure_id": "49581",
            "image_path": "p03_embedded_2.jpg",
            "panel_label": None,
            "x_range": [873.15, 1173.15],
            "y_range": [10.0, 1000.0],
            "x_scale": "linear",
            "status": "verified",
            "verified_at": "2026-08-19",
            "evidence": "numerically verified, but log-y axis",
            "excluded_reason": "log-y axis chart; ExtractionTask has no y_scale support",
        }
    ]

    assert parse_registry(raw)[0].excluded_reason == (
        "log-y axis chart; ExtractionTask has no y_scale support"
    )


def test_registry_json_round_trip_from_real_file_shape(tmp_path):
    from real_chart_bench.adapter.verified_pairing_registry import load_registry

    path = tmp_path / "registry.json"
    path.write_text(
        json.dumps(
            [
                {
                    "paper_id": "1",
                    "figure_id": "1",
                    "image_path": "a.jpg",
                    "panel_label": None,
                    "x_range": [0.0, 1.0],
                    "y_range": [0.0, 1.0],
                    "x_scale": "linear",
                    "status": "verified",
                    "verified_at": "2026-08-16",
                    "evidence": "x",
                }
            ]
        )
    )

    registry = load_registry(path)

    assert len(registry) == 1
