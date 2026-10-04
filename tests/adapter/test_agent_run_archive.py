import json

import pytest

from real_chart_bench.adapter.agent_run_archive import (
    OFFICIAL_PARTS,
    load_agent_run_parts,
)


def _write(dir_, name, answers):
    dir_.mkdir(parents=True, exist_ok=True)
    (dir_ / f"{name}.predictions.json").write_text(json.dumps(answers))


def _series(v):
    return [{"label": "a", "x": [1, 2], "y": [v, v]}]


def test_official_parts_are_exactly_part1_and_part2():
    assert OFFICIAL_PARTS == ("part1", "part2")


def test_reads_the_named_parts_and_merges_them(tmp_path):
    _write(tmp_path, "part1", {"fig_001.png": _series(1)})
    _write(tmp_path, "part2", {"fig_002.png": _series(2)})

    answers = load_agent_run_parts(tmp_path, OFFICIAL_PARTS)

    assert answers == {"fig_001.png": _series(1), "fig_002.png": _series(2)}


def test_other_prediction_files_in_the_dir_are_ignored(tmp_path):
    # The v3 Sonnet dirs also hold part{1,2}_nopillow (the first attempts).
    # A part*.predictions.json glob would load them, and sorted after part2
    # they would overwrite the official reruns' answers.
    _write(tmp_path, "part1", {"fig_001.png": _series(1)})
    _write(tmp_path, "part2", {"fig_002.png": _series(2)})
    _write(tmp_path, "part1_nopillow", {"fig_001.png": _series(99)})
    _write(tmp_path, "part2_nopillow", {"fig_002.png": _series(98), "fig_003.png": _series(3)})

    answers = load_agent_run_parts(tmp_path, OFFICIAL_PARTS)

    assert answers == {"fig_001.png": _series(1), "fig_002.png": _series(2)}


def test_a_diagnostic_part_list_reads_only_those_files(tmp_path):
    _write(tmp_path, "part1", {"fig_001.png": _series(1)})
    _write(tmp_path, "part1_nopillow", {"fig_001.png": _series(99)})
    _write(tmp_path, "part2_nopillow", {"fig_002.png": _series(98)})

    answers = load_agent_run_parts(tmp_path, ("part1_nopillow", "part2_nopillow"))

    assert answers == {"fig_001.png": _series(99), "fig_002.png": _series(98)}


def test_a_missing_part_is_an_error_not_a_silent_half_run(tmp_path):
    _write(tmp_path, "part1", {"fig_001.png": _series(1)})

    with pytest.raises(FileNotFoundError, match="part2"):
        load_agent_run_parts(tmp_path, OFFICIAL_PARTS)


def test_a_figure_answered_in_two_parts_is_an_error(tmp_path):
    # Batches are disjoint by construction; an overlap means a wrong file.
    _write(tmp_path, "part1", {"fig_001.png": _series(1)})
    _write(tmp_path, "part2", {"fig_001.png": _series(2)})

    with pytest.raises(ValueError, match="fig_001.png"):
        load_agent_run_parts(tmp_path, OFFICIAL_PARTS)


def test_an_empty_part_list_is_rejected(tmp_path):
    with pytest.raises(ValueError):
        load_agent_run_parts(tmp_path, ())
