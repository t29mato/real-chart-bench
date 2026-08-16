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


def _pending(model_id, note="awaiting external run"):
    return {
        "model_id": model_id,
        "model_name": model_id.title(),
        "status": "pending_external_run",
        "note": note,
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


def test_scored_row_has_status_scored_and_a_score():
    rows = build_leaderboard_rows([_result("a", 0.9)])

    assert rows[0].status == "scored"
    assert rows[0].mean_summary_score == 0.9


def test_pending_row_has_no_score_and_carries_a_note():
    rows = build_leaderboard_rows([_pending("lineformer", note="run the Colab notebook")])

    assert rows[0].status == "pending_external_run"
    assert rows[0].mean_summary_score is None
    assert rows[0].note == "run the Colab notebook"


def test_pending_rows_always_sort_after_scored_rows_regardless_of_model_id():
    results = [_pending("aaa-pending"), _result("zzz-scored", 0.01)]

    rows = build_leaderboard_rows(results)

    assert [r.model_id for r in rows] == ["zzz-scored", "aaa-pending"]


def test_multiple_pending_rows_are_ordered_by_model_id():
    results = [_pending("z-model"), _pending("a-model")]

    rows = build_leaderboard_rows(results)

    assert [r.model_id for r in rows] == ["a-model", "z-model"]


def test_pending_rows_still_get_sequential_rank_after_scored_rows():
    results = [_result("scored", 0.5), _pending("pending")]

    rows = build_leaderboard_rows(results)

    assert [r.rank for r in rows] == [1, 2]
