"""Point F1 by figure category (docs/design/local-model.md「方式D」): which
kinds of figure a row loses on, for 方式D against 方式C and the detector.

Categories are fixed once, from the figure and the ground truth's shape
(never from any row's score), and written to
data/local_orchestrator_runs/figure_categories.json:

- 2軸 (twin_y): a second y axis with its own scale is printed on the right
  of the crop (a twin axis, or the next panel's axis inside the crop). Tagged
  by eye on a contact sheet of the 94 figures (2026-10-06); a rule on the
  automatic calibration (labels read on both sides with different lines)
  found only 2 of them.
- 系列の分け方 (series_5plus): the ground truth has 5 or more series.
- 密集 (dense): the scorer's dense-marker rule (design 7.72); these figures
  are outside point F1 and are compared on summary_score.
- 目盛 (tick_hard): the automatic calibration does not agree with the
  person's within 1% of the tick span on both axes (the
  auto_calibration_check rule), no frame and no reading included.

  .venv/bin/python scripts/eval/orchestrator/category_compare.py [--recompute] \\
      results/<row>.json ...
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.auto_axis_calibration import calibrate_image, load_rgb  # noqa: E402
from real_chart_bench.adapter.ground_truth_store import load_ground_truth  # noqa: E402
from real_chart_bench.adapter.tick_plot_areas import load_tick_calibration  # noqa: E402
from real_chart_bench.domain.tick_calibration import axis_agreement  # noqa: E402

OUT = REPO / "data/local_orchestrator_runs/figure_categories.json"
TWIN_Y = {"10939-1528", "10939-1529", "10939-1532", "10939-1534", "10939-1536", "10939-1538",
          "17040-21020", "36305-45818", "44283-38973", "44283-39578"}
NAMES = {"twin_y": "2軸", "series_5plus": "系列の分け方(5系列以上)",
         "tick_hard": "目盛(自動校正が不一致)"}


def _one(item):
    pf, k, cal = item
    rgb = load_rgb(REPO / k["image_path"])
    c = calibrate_image(rgb) if rgb is not None else None
    hard = c is None or max(axis_agreement(c.x_fit, cal["x"], cal["x_scale"]),
                            axis_agreement(c.y_fit, cal["y"], cal["y_scale"])) > 0.01
    return pf, hard


def compute() -> dict:
    key = json.loads((REPO / "data/llm_run_pixcal/_key.json").read_text())
    cals = load_tick_calibration(REPO / "data/verified_pairs/tick_calibration.json")
    gt = load_ground_truth(REPO / "data/verified_pairs/ground_truth.json",
                           REPO / "data/verified_pairs/ground_truth_supplement")
    items = [(f"{k['paper_id']}-{k['figure_id']}", k, cals[(k["paper_id"], k["figure_id"])])
             for k in key.values()]
    with Pool(16) as pool:
        hard = dict(pool.map(_one, items))
    return {pf: {"twin_y": pf in TWIN_Y,
                 "series_5plus": len([c for c in gt[k["figure_id"]] if c.get("x")]) >= 5,
                 "tick_hard": hard[pf]}
            for pf, k, _ in sorted(items)}


def table(paths: list[str], cats: dict) -> list[dict]:
    rows = []
    for path in paths:
        r = json.loads(Path(path).read_text())
        per = {p["figure_id"]: p for p in r["per_figure"]}
        dense = {f for f, p in per.items() if p["marker_density"]["dense"]}
        nondense = [f for f in per if f not in dense]

        def f1(fs, per=per):
            v = [per[f]["point"]["by_tau"]["0.02"]["point_f1"] for f in fs]
            return [round(sum(v) / len(v), 3), len(v)] if v else [None, 0]

        out = {"row": Path(path).stem, "all": f1(nondense)}
        for c, name in NAMES.items():
            out[name] = f1([f for f in nondense if cats[f][c]])
            out[name + "以外"] = f1([f for f in nondense if not cats[f][c]])
        sv = [per[f]["summary_score"] for f in dense]
        out["密集(summary_score)"] = [round(sum(sv) / len(sv), 3), len(sv)]
        rows.append(out)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="*")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args()
    if args.recompute or not OUT.exists():
        OUT.write_text(json.dumps({"rules": __doc__.split("\n\n")[1], "figures": compute()},
                                  indent=1, ensure_ascii=False) + "\n")
    cats = json.loads(OUT.read_text())["figures"]
    for row in table(args.results, cats):
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
