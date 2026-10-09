"""Leaderboard and the nc subset (design §7.88.1): nc rows are their own
sections, after every core section, and never ranked with core rows."""

import pytest

from real_chart_bench.usecase.build_leaderboard import (
    build_dense_marker_rows,
    build_leaderboard_rows,
    select_shown_results,
)


def _result(model_id, score, *, version, n=10, subset=None):
    out = {
        "model_id": model_id,
        "model_name": model_id,
        "dataset_version": version,
        "run_at": "2026-10-10T00:00:00+00:00",
        "n_figures": n,
        "mean_summary_score": score,
        "per_figure": [],
    }
    if subset:
        out["subset"] = subset
    return out


def test_nc_rows_are_a_separate_group_ranked_on_their_own():
    rows = build_leaderboard_rows(
        [
            _result("core-a", 0.5, version="v0-eval-pilot-n91-noaxis", n=91),
            _result("nc-a", 0.9, version="nc-v0-eval-pilot-n12-noaxis", n=12, subset="nc"),
            _result("core-b", 0.4, version="v0-eval-pilot-n91-noaxis", n=91),
        ]
    )
    assert [(r.model_id, r.rank) for r in rows] == [("core-a", 1), ("core-b", 2), ("nc-a", 1)]


def test_nc_sections_come_after_every_core_section_even_if_larger():
    rows = build_leaderboard_rows(
        [
            _result("nc-a", 0.9, version="nc-v0-eval-pilot-n500-noaxis", n=500, subset="nc"),
            _result("syn", 0.5, version="synthetic-plotqa-dot-line-n100", n=100),
            _result("core-a", 0.5, version="v0-eval-pilot-n91-pixcal", n=91),
        ]
    )
    assert [r.model_id for r in rows] == ["core-a", "syn", "nc-a"]


def test_nc_main_conditions_keep_their_order():
    rows = build_leaderboard_rows(
        [
            _result("p", 0.9, version="nc-v0-eval-pilot-n12-pixcal", n=12, subset="nc"),
            _result("a", 0.9, version="nc-v0-eval-pilot-n12-noaxis", n=12, subset="nc"),
        ]
    )
    assert [r.model_id for r in rows] == ["a", "p"]


def test_a_result_whose_subset_and_version_disagree_is_refused():
    with pytest.raises(ValueError, match="pool"):
        build_leaderboard_rows([_result("x", 0.5, version="v0-eval-pilot-n91", subset="nc")])


def test_dense_table_refuses_the_mismatch_too():
    bad = _result("x", 0.5, version="nc-v0-eval-pilot-n12")
    bad["dense_marker_metrics"] = {"n_figures": 1, "mean_summary_score": 0.5}
    with pytest.raises(ValueError, match="pool"):
        build_dense_marker_rows([bad])


def test_shown_results_filter_applies_to_nc_like_core():
    results = [
        _result("main", 0.5, version="nc-v0-eval-pilot-n12-noaxis", subset="nc"),
        _result("axis-range", 0.5, version="nc-v0-eval-pilot-n12", subset="nc"),
        _result("naive-cv-v0-nc", 0.5, version="nc-v0-eval-pilot-n12-noaxis", subset="nc"),
    ]
    assert [r["model_id"] for r in select_shown_results(results)] == ["main"]
