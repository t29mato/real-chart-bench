"""Scoring scripts and the nc subset (design §7.88.1): run_baselines --subset nc
writes an nc-labelled payload; the leaderboard labels nc sections and keeps
them out of the version banner."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

from real_chart_bench.adapter.naive_cv_extractor import NaiveCvModelRunner
from real_chart_bench.domain.dataset_subset import DatasetSubset
from real_chart_bench.usecase.build_leaderboard import LeaderboardRow

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def load(rel, name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / rel)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- run_baselines -------------------------------------------------------------------


def _baselines_with_one_real_figure(monkeypatch):
    mod = load("eval/run_baselines.py", "run_baselines")
    seen = []

    def fake_build_dataset(subset=DatasetSubset.CORE):
        seen.append(subset)
        real = dataclasses.replace(mod._synthetic_items()[0], figure_id="1-10")
        return [real, *mod._synthetic_items()], 1

    monkeypatch.setattr(mod, "build_dataset", fake_build_dataset)
    return mod, seen


def test_nc_payload_is_flagged_and_versioned_nc(monkeypatch):
    mod, seen = _baselines_with_one_real_figure(monkeypatch)
    payload = mod.run("naive-cv-v0-nc", "x", NaiveCvModelRunner(), DatasetSubset.NC)
    assert seen == [DatasetSubset.NC]
    assert payload["subset"] == "nc"
    assert payload["dataset_version"].startswith("nc-v0-eval-pilot-n1")
    assert payload["n_figures"] == 1


def test_core_payload_has_no_subset_key(monkeypatch):
    mod, seen = _baselines_with_one_real_figure(monkeypatch)
    payload = mod.run("naive-cv-v0", "x", NaiveCvModelRunner())
    assert seen == [DatasetSubset.CORE]
    assert "subset" not in payload
    assert payload["dataset_version"].startswith("v0-eval-pilot-n1")


# --- leaderboard/generate.py ---------------------------------------------------------


def _row(version, run_at):
    return LeaderboardRow(
        rank=1, model_id=version, model_name=version, status="scored",
        mean_summary_score=0.5, n_figures=10, dataset_version=version, run_at=run_at,
    )


def test_banner_names_the_latest_core_set_even_if_an_nc_run_is_newer():
    gen = load("leaderboard/generate.py", "leaderboard_generate")
    rows = [
        _row("v0-eval-pilot-n91-pixcal", "2026-10-01T00:00:00+00:00"),
        _row("nc-v0-eval-pilot-n12-noaxis", "2026-10-09T00:00:00+00:00"),
    ]
    assert gen.latest_banner(rows) == ("v0-eval-pilot-n91-pixcal", "2026-10-01T00:00:00+00:00")


def test_banner_with_only_nc_rows_says_no_core_runs():
    gen = load("leaderboard/generate.py", "leaderboard_generate")
    assert gen.latest_banner([_row("nc-v0-eval-pilot-n12", "2026-10-09")]) == (
        "(no scored runs yet)",
        "-",
    )


def test_nc_section_heading_is_labelled_and_keeps_its_condition():
    gen = load("leaderboard/generate.py", "leaderboard_generate")
    heading = gen._section_heading(
        "nc-v0-eval-pilot-n12-noaxis", [_row("nc-v0-eval-pilot-n12-noaxis", "t")]
    )
    assert "NC subset" in heading
    assert "non-commercial" in heading
    assert "fully automatic" in heading


def test_core_section_heading_has_no_nc_label():
    gen = load("leaderboard/generate.py", "leaderboard_generate")
    heading = gen._section_heading(
        "v0-eval-pilot-n91-noaxis", [_row("v0-eval-pilot-n91-noaxis", "t")]
    )
    assert "NC subset" not in heading
