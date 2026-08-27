from real_chart_bench.domain.curve import ScaleType
from real_chart_bench.domain.verified_pairing import VerificationStatus, VerifiedPairing
from real_chart_bench.usecase.build_leaderboard_breakdown import (
    build_model_breakdown,
    categorize_figure,
)


def _pairing(paper_id, figure_id, x_scale=ScaleType.LINEAR):
    return VerifiedPairing(
        paper_id=paper_id,
        figure_id=figure_id,
        image_path=None,
        panel_label=None,
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        status=VerificationStatus.VERIFIED,
        verified_at="2026-08-27",
        evidence="test fixture",
        x_scale=x_scale,
    )


def _result(per_figure):
    return {
        "model_id": "m",
        "model_name": "M",
        "dataset_version": "v0-eval-pilot-n2",
        "run_at": "2026-08-27T00:00:00+00:00",
        "n_figures": len(per_figure),
        "mean_summary_score": (
            sum(p["summary_score"] for p in per_figure) / len(per_figure) if per_figure else 0.0
        ),
        "per_figure": per_figure,
    }


def _figure(figure_id, score):
    return {
        "figure_id": figure_id,
        "summary_score": score,
        "match_rate": 0.0,
        "mean_curve_distance": 1.0,
        "mean_coverage_ratio": 0.0,
        "error": None,
    }


class TestCategorizeFigure:
    def test_synthetic_prefix_is_its_own_category_regardless_of_registry(self):
        assert categorize_figure("synthetic-log-black-line", {}) == "synthetic"

    def test_real_figure_with_linear_x_scale_pairing(self):
        pairings = {"18759-12217": _pairing("18759", "12217", ScaleType.LINEAR)}
        assert categorize_figure("18759-12217", pairings) == "real-linear-x"

    def test_real_figure_with_log_x_scale_pairing(self):
        pairings = {"5902-15112": _pairing("5902", "15112", ScaleType.LOG)}
        assert categorize_figure("5902-15112", pairings) == "real-log-x"

    def test_real_figure_id_missing_from_registry_falls_back_to_unknown(self):
        # Defensive: a results file could in principle reference a figure_id
        # no longer in the registry (e.g. re-classified since the run). Must
        # not crash the leaderboard build over one stale reference.
        assert categorize_figure("99999-00000", {}) == "real-unknown"


class TestBuildModelBreakdown:
    def test_groups_and_averages_scores_per_category(self):
        pairings = {
            "18759-12217": _pairing("18759", "12217", ScaleType.LINEAR),
            "5902-15112": _pairing("5902", "15112", ScaleType.LOG),
        }
        result = _result(
            [
                _figure("18759-12217", 0.8),
                _figure("5902-15112", 0.4),
                _figure("synthetic-linear-single", 1.0),
                _figure("synthetic-log-black-line", 0.0),
            ]
        )

        breakdown = build_model_breakdown(result, pairings)

        by_category = {b.category: b for b in breakdown}
        assert by_category["real-linear-x"].n_figures == 1
        assert by_category["real-linear-x"].mean_summary_score == 0.8
        assert by_category["real-log-x"].n_figures == 1
        assert by_category["real-log-x"].mean_summary_score == 0.4
        assert by_category["synthetic"].n_figures == 2
        assert by_category["synthetic"].mean_summary_score == 0.5

    def test_multiple_figures_in_the_same_category_are_averaged(self):
        pairings = {
            "a-1": _pairing("a", "1", ScaleType.LINEAR),
            "a-2": _pairing("a", "2", ScaleType.LINEAR),
        }
        result = _result([_figure("a-1", 1.0), _figure("a-2", 0.0)])

        breakdown = build_model_breakdown(result, pairings)

        assert len(breakdown) == 1
        assert breakdown[0].category == "real-linear-x"
        assert breakdown[0].n_figures == 2
        assert breakdown[0].mean_summary_score == 0.5

    def test_categories_are_sorted_deterministically(self):
        pairings = {"a-1": _pairing("a", "1", ScaleType.LOG)}
        result = _result(
            [_figure("a-1", 0.5), _figure("synthetic-x", 0.5)]
        )

        breakdown = build_model_breakdown(result, pairings)

        assert [b.category for b in breakdown] == ["real-log-x", "synthetic"]

    def test_pending_result_with_no_per_figure_key_yields_empty_breakdown(self):
        pending = {"model_id": "m", "model_name": "M", "status": "pending_external_run"}

        assert build_model_breakdown(pending, {}) == []

    def test_empty_per_figure_yields_empty_breakdown(self):
        assert build_model_breakdown(_result([]), {}) == []
