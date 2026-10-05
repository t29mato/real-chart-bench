import json

import pytest

from real_chart_bench.adapter.local_vlm_run import load_local_vlm_run


def _write(tmp_path, records):
    path = tmp_path / "calibrated.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    return path


def _ok(fig, series=None, **extra):
    return {
        "fig": fig,
        "paper_figure": f"p-{fig}",
        "error": None,
        "raw": "{}",
        "parsed": series if series is not None else [{"label": "a", "x": [1], "y": [2]}],
        "parse_error": None,
        "truncated": False,
        **extra,
    }


def test_parsed_answers_are_keyed_by_task_id(tmp_path):
    series = [{"label": "a", "x": [1, 2], "y": [3, 4]}]
    run = load_local_vlm_run(_write(tmp_path, [_ok("fig_001.png", series)]))

    assert run.answers == {"fig_001.png": series}


def test_parse_failure_is_left_out_of_answers_and_counted(tmp_path):
    truncated = {
        **_ok("fig_002.png"),
        "parsed": None,
        "parse_error": "JSONDecodeError: x",
        "truncated": True,
    }
    run = load_local_vlm_run(_write(tmp_path, [_ok("fig_001.png"), truncated]))

    assert set(run.answers) == {"fig_001.png"}
    assert run.parse_failures == ("fig_002.png",)
    assert run.truncated == ("fig_002.png",)
    assert run.errors == ()


def test_runtime_error_is_left_out_and_counted_separately_from_parse_failures(tmp_path):
    crashed = {"fig": "fig_003.png", "error": "ValueError: too many digits", "parsed": None}
    run = load_local_vlm_run(_write(tmp_path, [crashed]))

    assert run.answers == {}
    assert run.errors == ("fig_003.png",)
    assert run.parse_failures == ()


def test_accepted_key_mismatch_is_an_answer_but_recorded(tmp_path):
    rec = _ok("fig_004.png", parse_error="key mismatch (accepted single key)")
    run = load_local_vlm_run(_write(tmp_path, [rec]))

    assert set(run.answers) == {"fig_004.png"}
    assert run.parse_failures == ()
    assert run.accepted_with_warning == ("fig_004.png",)


def test_an_empty_series_list_is_an_answer_not_a_failure(tmp_path):
    run = load_local_vlm_run(_write(tmp_path, [_ok("fig_005.png", series=[])]))

    assert run.answers == {"fig_005.png": []}
    assert run.parse_failures == ()


def test_n_records_counts_every_line_and_blank_lines_are_ignored(tmp_path):
    path = _write(tmp_path, [_ok("fig_001.png"), _ok("fig_002.png")])
    path.write_text(path.read_text() + "\n")

    assert load_local_vlm_run(path).n_records == 2


def test_a_figure_recorded_twice_is_rejected(tmp_path):
    # the workers append and skip done figures; a duplicate means the file was
    # stitched by hand and which answer counts would be ambiguous
    with pytest.raises(ValueError, match="fig_001.png"):
        load_local_vlm_run(_write(tmp_path, [_ok("fig_001.png"), _ok("fig_001.png")]))


def test_peak_memory_is_the_largest_recorded_and_none_when_absent(tmp_path):
    recs = [
        _ok("fig_001.png", peak_memory_gb=11.2),
        _ok("fig_002.png", peak_memory_gb=13.5),
        {"fig": "fig_003.png", "error": "boom", "parsed": None},
    ]

    assert load_local_vlm_run(_write(tmp_path, recs)).peak_memory_gb_max == 13.5
    assert load_local_vlm_run(_write(tmp_path, [_ok("fig_001.png")])).peak_memory_gb_max is None


def test_seconds_per_figure_are_kept_for_every_record_that_has_them(tmp_path):
    failed = {**_ok("fig_002.png", seconds=200.0), "parsed": None, "truncated": True}
    run = load_local_vlm_run(
        _write(tmp_path, [_ok("fig_001.png", seconds=12.5), failed, _ok("fig_003.png")])
    )

    # failures took time too; a record without timing is left out
    assert run.seconds == {"fig_001.png": 12.5, "fig_002.png": 200.0}
