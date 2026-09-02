"""select_human_ceiling_subset(): picks a re-digitization subset that covers
the *distribution* of figure characteristics in the full verified-pairs
population, not just a convenient sample (戦略メモ「柱B」: a biased human-
ceiling subset over-credits methods strong on the over-represented type, the
same argument the memo makes about baseline evaluation subsets).
"""

from __future__ import annotations

from real_chart_bench.usecase.select_human_ceiling_subset import (
    FigureProfile,
    build_figure_profiles,
    coverage_report,
    select_coverage_subset,
)


def _profile(
    figure_id,
    axis_type="linear-linear",
    series_bucket="single",
    points_bucket="medium",
    density_bucket="medium",
    y_quantity="Electrical conductivity",
):
    return FigureProfile(
        figure_id=figure_id,
        paper_id=figure_id.split("-")[0],
        axis_type=axis_type,
        series_bucket=series_bucket,
        points_bucket=points_bucket,
        density_bucket=density_bucket,
        y_quantity=y_quantity,
    )


# --- select_coverage_subset() ----------------------------------------------


def test_selection_is_deterministic_across_repeated_calls():
    profiles = [_profile(f"p{i}-{i}") for i in range(40)]

    first = select_coverage_subset(profiles, target_size=20)
    second = select_coverage_subset(profiles, target_size=20)

    assert first == second


def test_selected_figure_ids_are_unique_and_drawn_from_the_population():
    profiles = [_profile(f"p{i}-{i}") for i in range(40)]

    selected = select_coverage_subset(profiles, target_size=20)

    assert len(selected) == len(set(selected))
    population_ids = {p.figure_id for p in profiles}
    assert set(selected) <= population_ids


def test_selection_covers_every_category_present_when_population_allows():
    profiles = [
        _profile("p1-1", axis_type="linear-linear", series_bucket="single"),
        _profile("p2-2", axis_type="log-x", series_bucket="multi"),
        _profile(
            "p3-3", axis_type="log-y", series_bucket="single", y_quantity="Seebeck coefficient"
        ),
        _profile("p4-4", axis_type="linear-linear", series_bucket="multi", points_bucket="high"),
        _profile("p5-5", axis_type="linear-linear", series_bucket="single", density_bucket="low"),
    ] * 6  # 30 figures, each distinct category repeated 6x

    selected = select_coverage_subset(profiles, target_size=20)
    selected_profiles = [p for p in profiles if p.figure_id in selected]

    assert {p.axis_type for p in selected_profiles} == {"linear-linear", "log-x", "log-y"}
    assert {p.series_bucket for p in selected_profiles} == {"single", "multi"}


def test_target_size_larger_than_population_returns_the_whole_population():
    profiles = [_profile(f"p{i}-{i}") for i in range(5)]

    selected = select_coverage_subset(profiles, target_size=25)

    assert set(selected) == {p.figure_id for p in profiles}


def test_empty_population_returns_empty_selection():
    assert select_coverage_subset([], target_size=25) == []


def test_selection_respects_target_size_when_population_is_larger():
    profiles = [_profile(f"p{i}-{i}") for i in range(111)]

    selected = select_coverage_subset(profiles, target_size=25)

    assert len(selected) == 25


# --- coverage_report() ------------------------------------------------------


def test_coverage_report_reports_full_and_subset_fractions():
    profiles = [
        _profile("p1-1", axis_type="linear-linear"),
        _profile("p2-2", axis_type="linear-linear"),
        _profile("p3-3", axis_type="log-x"),
        _profile("p4-4", axis_type="log-x"),
    ]
    selected = ["p1-1", "p3-3"]

    report = coverage_report(profiles, selected)

    axis_rows = {(r.dimension, r.category): r for r in report if r.dimension == "axis_type"}
    assert axis_rows[("axis_type", "linear-linear")].full_count == 2
    assert axis_rows[("axis_type", "linear-linear")].subset_count == 1
    assert axis_rows[("axis_type", "log-x")].full_fraction == 0.5
    assert axis_rows[("axis_type", "log-x")].subset_fraction == 0.5


def test_coverage_report_covers_all_five_dimensions():
    profiles = [_profile("p1-1"), _profile("p2-2", axis_type="log-x")]

    report = coverage_report(profiles, ["p1-1"])

    assert {r.dimension for r in report} == {
        "axis_type",
        "series_bucket",
        "points_bucket",
        "density_bucket",
        "y_quantity",
    }


# --- build_figure_profiles() ------------------------------------------------


def test_build_figure_profiles_derives_axis_type_from_x_and_y_scale():
    registry_rows = [
        {"paper_id": "1", "figure_id": "1", "x_scale": "linear", "y_scale": "linear"},
        {"paper_id": "2", "figure_id": "2", "x_scale": "log", "y_scale": "linear"},
        {"paper_id": "3", "figure_id": "3", "x_scale": "linear", "y_scale": "log"},
    ]
    ground_truth = {
        "1": [{"x": [1, 2, 3], "y": [1, 2, 3], "prop_y": "Q"}],
        "2": [{"x": [1, 2, 3], "y": [1, 2, 3], "prop_y": "Q"}],
        "3": [{"x": [1, 2, 3], "y": [1, 2, 3], "prop_y": "Q"}],
    }

    profiles = build_figure_profiles(registry_rows, ground_truth)

    by_id = {p.figure_id: p for p in profiles}
    assert by_id["1-1"].axis_type == "linear-linear"
    assert by_id["2-2"].axis_type == "log-x"
    assert by_id["3-3"].axis_type == "log-y"


def test_build_figure_profiles_derives_series_bucket_from_curve_count():
    registry_rows = [
        {"paper_id": "1", "figure_id": "1", "x_scale": "linear", "y_scale": "linear"},
        {"paper_id": "2", "figure_id": "2", "x_scale": "linear", "y_scale": "linear"},
    ]
    ground_truth = {
        "1": [{"x": [1, 2], "y": [1, 2], "prop_y": "Q"}],
        "2": [
            {"x": [1, 2], "y": [1, 2], "prop_y": "Q"},
            {"x": [1, 2], "y": [1, 2], "prop_y": "R"},
        ],
    }

    profiles = build_figure_profiles(registry_rows, ground_truth)

    by_id = {p.figure_id: p for p in profiles}
    assert by_id["1-1"].series_bucket == "single"
    assert by_id["2-2"].series_bucket == "multi"


def test_build_figure_profiles_skips_figures_with_no_ground_truth_curves():
    registry_rows = [
        {"paper_id": "1", "figure_id": "1", "x_scale": "linear", "y_scale": "linear"},
        {"paper_id": "2", "figure_id": "2", "x_scale": "linear", "y_scale": "linear"},
    ]
    ground_truth = {
        "1": [{"x": [1, 2], "y": [1, 2], "prop_y": "Q"}],
        "2": [],
    }

    profiles = build_figure_profiles(registry_rows, ground_truth)

    assert {p.figure_id for p in profiles} == {"1-1"}


def test_build_figure_profiles_uses_the_most_common_prop_y_as_y_quantity():
    registry_rows = [{"paper_id": "1", "figure_id": "1", "x_scale": "linear", "y_scale": "linear"}]
    ground_truth = {
        "1": [
            {"x": [1, 2], "y": [1, 2], "prop_y": "Seebeck coefficient"},
            {"x": [1, 2], "y": [1, 2], "prop_y": "Seebeck coefficient"},
            {"x": [1, 2], "y": [1, 2], "prop_y": "ZT"},
        ],
    }

    profiles = build_figure_profiles(registry_rows, ground_truth)

    assert profiles[0].y_quantity == "Seebeck coefficient"
