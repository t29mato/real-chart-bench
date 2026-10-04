from real_chart_bench.usecase.build_leaderboard import (
    build_dense_marker_rows,
    build_leaderboard_rows,
)


def _result(model_id, score, n=4, dataset_version="v0-eval-pilot-2026-08-16"):
    return {
        "model_id": model_id,
        "model_name": model_id.title(),
        "dataset_version": dataset_version,
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


def test_pending_rows_are_placed_last_and_unranked():
    # A rank on a pending row would imply it was scored and compared against
    # the scored rows above it -- it wasn't. Pending rows carry rank=None so
    # rendering can show a dash instead of a misleading ordinal.
    results = [_result("scored", 0.5), _pending("pending")]

    rows = build_leaderboard_rows(results)

    assert [r.model_id for r in rows] == ["scored", "pending"]
    assert rows[0].rank == 1
    assert rows[1].rank is None


class TestGroupingByDatasetVersion:
    """A rank number must only ever compare runs scored on the same figure
    set. build_leaderboard_rows groups scored results by dataset_version and
    ranks within each group -- each group's best scorer is rank 1, even if
    its raw score is lower than the top score of a larger, unrelated group.
    """

    def test_two_groups_are_each_ranked_from_1(self):
        results = [
            _result("big-set-only", 0.731, n=114, dataset_version="v0-eval-pilot-n111"),
            _result("lineformer", 0.647, n=45, dataset_version="v0-eval-pilot-n42"),
            _result("naive-cv-subset", 0.607, n=45, dataset_version="v0-eval-pilot-n42"),
        ]

        rows = build_leaderboard_rows(results)
        by_id = {r.model_id: r for r in rows}

        # Same-figure-set group: LineFormer (0.647) beats the naive-CV
        # subset (0.607) on identical figures -- LineFormer must be rank 1
        # here, not "beaten" by a bigger, different-figure-set group.
        assert by_id["lineformer"].rank == 1
        assert by_id["naive-cv-subset"].rank == 2
        # The single-member, larger-but-different-figures group is also
        # rank 1 within its own group -- it is never compared against the
        # v0-eval-pilot-n42 rows.
        assert by_id["big-set-only"].rank == 1

    def test_groups_are_ordered_largest_figure_count_first(self):
        results = [
            _result("small-a", 0.9, n=10, dataset_version="v-small"),
            _result("big-a", 0.1, n=100, dataset_version="v-big"),
        ]

        rows = build_leaderboard_rows(results)

        # The larger evaluated set leads even though its score is lower --
        # group order is by representativeness (figure count), never score.
        assert [r.dataset_version for r in rows] == ["v-big", "v-small"]

    def test_group_order_ties_break_by_dataset_version_string(self):
        results = [
            _result("z", 0.5, n=10, dataset_version="v-zeta"),
            _result("a", 0.5, n=10, dataset_version="v-alpha"),
        ]

        rows = build_leaderboard_rows(results)

        assert [r.dataset_version for r in rows] == ["v-alpha", "v-zeta"]

    def test_single_group_ranks_normally_from_1(self):
        results = [
            _result("a", 0.9, dataset_version="only-set"),
            _result("b", 0.5, dataset_version="only-set"),
            _result("c", 0.1, dataset_version="only-set"),
        ]

        rows = build_leaderboard_rows(results)

        assert [r.rank for r in rows] == [1, 2, 3]
        assert [r.model_id for r in rows] == ["a", "b", "c"]

    def test_pending_rows_are_never_grouped_with_scored_rows(self):
        results = [
            _result("a", 0.9, dataset_version="v1"),
            _result("b", 0.1, dataset_version="v2"),
            _pending("p"),
        ]

        rows = build_leaderboard_rows(results)

        assert rows[-1].model_id == "p"
        assert rows[-1].rank is None
        assert rows[-1].status == "pending_external_run"

    def test_result_missing_dataset_version_does_not_crash_and_sorts_last(self):
        # Defensive: a malformed/legacy result without a dataset_version
        # must not crash the leaderboard build. It gets its own group (so
        # it's still ranked internally if more than one such row exists),
        # placed after every labeled group -- an unlabeled figure set is
        # never allowed to visually outrank a traceable, labeled one -- but
        # still before any pending row.
        results = [
            {
                "model_id": "unlabeled",
                "model_name": "Unlabeled",
                "run_at": "2026-08-16T00:00:00+00:00",
                "n_figures": 999,
                "mean_summary_score": 0.99,
                "per_figure": [],
            },
            _result("labeled", 0.01, n=1, dataset_version="v-tiny"),
            _pending("pending-model"),
        ]

        rows = build_leaderboard_rows(results)

        assert [r.model_id for r in rows] == ["labeled", "unlabeled", "pending-model"]
        assert rows[1].rank == 1
        assert rows[1].dataset_version is None


# --- design §7.67: point_f1 is the ranking metric ---------------------------


def _with_points(result, f1, recall=0.5, precision=0.5, loc=0.01):
    cell = {
        "point_recall": recall,
        "point_precision": precision,
        "point_f1": f1,
        "point_loc_error": loc,
        "n_figures_with_matches": 1,
    }
    result["point_metrics"] = {
        "primary_tau": 0.02,
        "norm": "euclidean",
        "n_figures": result["n_figures"],
        "by_tau": {"0.02": {"macro": cell, "micro": dict(cell)}},
    }
    return result


def test_rows_with_point_metrics_rank_by_point_f1_not_summary_score():
    results = [
        _with_points(_result("high-summary", 0.99), f1=0.40),
        _with_points(_result("low-summary", 0.70), f1=0.80),
    ]

    rows = build_leaderboard_rows(results)

    assert [r.model_id for r in rows] == ["low-summary", "high-summary"]
    assert rows[0].point_f1 == 0.80
    assert rows[0].mean_summary_score == 0.70


def test_row_carries_the_primary_point_columns():
    rows = build_leaderboard_rows(
        [_with_points(_result("a", 0.9), f1=0.6, recall=0.7, precision=0.55, loc=0.008)]
    )

    r = rows[0]
    assert (r.point_f1, r.point_recall, r.point_precision, r.point_loc_error) == (
        0.6,
        0.7,
        0.55,
        0.008,
    )
    assert r.point_tau == 0.02


def test_rows_without_point_metrics_rank_after_point_scored_rows_in_their_group():
    results = [_result("legacy", 0.99), _with_points(_result("new", 0.5), f1=0.1)]

    rows = build_leaderboard_rows(results)

    assert [r.model_id for r in rows] == ["new", "legacy"]
    assert rows[1].point_f1 is None


# --- design §7.72: dense-marker figures get their own curve-distance table ---


def _with_dense(result, n_point, n_dense, dense_score, match=1.0, dist=0.1, cov=0.9):
    result["point_metrics"]["n_figures"] = n_point
    result["dense_marker_metrics"] = {
        "criterion": "spacing < 2 tau",
        "n_figures": n_dense,
        "mean_summary_score": dense_score,
        "mean_match_rate": match,
        "mean_curve_distance": dist,
        "mean_coverage_ratio": cov,
    }
    return result


def test_main_row_carries_the_point_figure_count():
    rows = build_leaderboard_rows(
        [_with_dense(_with_points(_result("a", 0.9, n=10), f1=0.6), 8, 2, 0.7)]
    )

    assert rows[0].n_figures == 10
    assert rows[0].point_n_figures == 8


def test_point_figure_count_is_none_without_point_metrics():
    assert build_leaderboard_rows([_result("legacy", 0.9)])[0].point_n_figures is None


def test_dense_rows_rank_by_the_dense_summary_score_not_point_f1():
    results = [
        _with_dense(_with_points(_result("pointy", 0.9, n=10), f1=0.9), 8, 2, 0.5),
        _with_dense(_with_points(_result("curvy", 0.8, n=10), f1=0.1), 8, 2, 0.95),
    ]

    rows = build_dense_marker_rows(results)

    assert [r.model_id for r in rows] == ["curvy", "pointy"]
    assert [r.rank for r in rows] == [1, 2]
    assert rows[0].mean_summary_score == 0.95
    assert rows[0].n_figures == 2
    assert (rows[0].mean_match_rate, rows[0].mean_curve_distance, rows[0].mean_coverage_ratio) == (
        1.0,
        0.1,
        0.9,
    )


def test_dense_rows_rank_within_each_dataset_version():
    results = [
        _with_dense(_with_points(_result("a", 0.9, n=10, dataset_version="big"), 0.5), 8, 2, 0.4),
        _with_dense(_with_points(_result("b", 0.9, n=4, dataset_version="small"), 0.5), 3, 1, 0.9),
        _with_dense(_with_points(_result("c", 0.9, n=10, dataset_version="big"), 0.5), 8, 2, 0.6),
    ]

    rows = build_dense_marker_rows(results)

    assert [(r.dataset_version, r.model_id, r.rank) for r in rows] == [
        ("big", "c", 1),
        ("big", "a", 2),
        ("small", "b", 1),
    ]


def test_dense_rows_skip_results_without_dense_figures_or_the_block():
    results = [
        _with_dense(_with_points(_result("none-dense", 0.9), f1=0.5), 4, 0, None),
        _with_points(_result("pre-7-72", 0.9), f1=0.5),
        _result("legacy", 0.9),
        _pending("lf"),
    ]

    assert build_dense_marker_rows(results) == []
