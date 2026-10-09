from __future__ import annotations

import gzip
import json

from real_chart_bench.adapter.starrydata_figure_gt import load_figure_gt, sibling_groups_of

HEADER = "SID,DOI,composition,sample_id,figure_id,figure_name,prop_x,prop_y,unit_x,unit_y,x,y\n"


def _row(sid, fid, name, px, py, ux, uy, xs, ys, comp="A"):
    return (
        f'{sid},10.1/x,{comp},1,{fid},{name},{px},{py},{ux},{uy},'
        f'"{json.dumps(xs)}","{json.dumps(ys)}"\n'
    )


def _write(tmp_path, rows):
    path = tmp_path / "curves.csv.gz"
    with gzip.open(path, "wt", encoding="utf-8-sig") as f:
        f.write(HEADER + "".join(rows))
    return path


def test_groups_curves_by_paper_and_figure_and_keeps_manifest_curve_ids(tmp_path):
    path = _write(tmp_path, [
        _row("1", "10", "3(a)", "Temperature", "Seebeck coefficient", "K", "V*K^(-1)",
             [300, 400], [1e-6, 2e-6]),
        _row("2", "20", "1", "Temperature", "ZT", "K", "1", [300, 400], [0.1, 0.2]),
        _row("1", "10", "3(a)", "Temperature", "Seebeck coefficient", "K", "V*K^(-1)",
             [300, 400], [3e-6, 4e-6], comp="B"),
    ])
    out = load_figure_gt(path, {"1", "2"})
    fig = out["1"]["10"]
    assert fig.figure_name == "3(a)"
    assert [c.curve_id for c in fig.curves] == ["1-10-0", "1-10-1"]
    assert out["2"]["20"].curves[0].curve_id == "2-20-0"


def test_paper_not_requested_is_skipped(tmp_path):
    path = _write(tmp_path, [_row("9", "1", "1", "T", "ZT", "K", "1", [1, 2], [1, 2])])
    assert load_figure_gt(path, {"1"}) == {}


def test_dominant_axis_pair_is_kept_and_the_rest_counted(tmp_path):
    path = _write(tmp_path, [
        _row("1", "10", "2", "T", "ZT", "K", "1", [1, 2], [1, 2]),
        _row("1", "10", "2", "T", "ZT", "K", "1", [1, 2], [2, 3]),
        _row("1", "10", "2", "T", "Seebeck", "K", "V*K^(-1)", [1, 2], [5, 6]),
    ])
    fig = load_figure_gt(path, {"1"})["1"]["10"]
    assert (fig.prop_y, fig.unit_y) == ("ZT", "1")
    assert len(fig.curves) == 2
    assert fig.n_other_axis_curves == 1


def test_malformed_or_empty_rows_are_skipped(tmp_path):
    path = _write(tmp_path, [
        _row("1", "10", "2", "T", "ZT", "K", "1", [1, 2], [1]),
        _row("1", "10", "2", "T", "ZT", "K", "1", [], []),
        _row("1", "10", "2", "T", "ZT", "K", "1", [1, 2], [1, 2]),
    ])
    fig = load_figure_gt(path, {"1"})["1"]["10"]
    assert len(fig.curves) == 1


def test_sibling_groups_of_finds_sigma_and_log_sigma_records(tmp_path):
    xs = [0.0009, 0.001, 0.0011]
    sigma = [100.0, 50.0, 10.0]
    log = [100 * __import__("math").log10(v) - 200 for v in sigma]
    path = _write(tmp_path, [
        _row("1", "10", "8a sigma", "Inverse temperature", "Electrical conductivity",
             "K^(-1)", "S*m^(-1)", xs, sigma),
        _row("1", "11", "8a", "Inverse temperature", "log(Electrical conductivity)",
             "K^(-1)", "S*m^(-1)", xs, log),
        _row("1", "12", "9", "Inverse temperature", "Seebeck coefficient", "K^(-1)", "V*K^(-1)",
             xs, [1e-6, 2e-6, 3e-6]),
    ])
    figures = load_figure_gt(path, ["1"])["1"]
    assert sibling_groups_of(figures) == [("10", "11")]
    assert sibling_groups_of({}) == []
