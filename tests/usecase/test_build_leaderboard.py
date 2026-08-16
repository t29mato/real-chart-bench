from real_chart_bench.usecase.build_leaderboard import build_leaderboard_rows


def _result(model_id, score, n=4):
    return {
        "model_id": model_id,
        "model_name": model_id.title(),
        "dataset_version": "v0-eval-pilot-2026-08-16",
        "run_at": "2026-08-16T00:00:00+00:00",
        "n_figures": n,
        "mean_summary_score": score,
        "per_figure": [],
    }


def test_rows_are_sorted_by_score_descending():
    results = [_result("weak", 0.2), _result("strong", 0.9), _result("mid", 0.5)]

    rows = build_leaderboard_rows(results)

    assert [r.model_id for r in rows] == ["strong", "mid", "weak"]


def test_row_rank_is_1_indexed():
    results = [_result("a", 0.9), _result("b", 0.1)]

    rows = build_leaderboard_rows(results)

    assert [r.rank for r in rows] == [1, 2]


def test_empty_results_yields_empty_leaderboard():
    assert build_leaderboard_rows([]) == []


def test_ties_are_broken_deterministically_by_model_id():
    results = [_result("zeta", 0.5), _result("alpha", 0.5)]

    rows = build_leaderboard_rows(results)

    assert [r.model_id for r in rows] == ["alpha", "zeta"]
