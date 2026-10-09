"""Upper bound on the CHART-Infographics loss that comes from unit
conventions (%, x10^n, E-notation), paper 4.8.2.

For every figure and model, the answer is rescored after multiplying ONE axis
(x or y, all series) by 10^k for k in -9..9, and the best point F1 is kept.
The scoring is the error analysis's point F1 (export_chartinfo_viewer.matches:
one-to-one, series paired, tau 2% of the ground-truth spread), so the baseline
is the same 0.873 as error_analysis.json. The gain is an UPPER bound: a
rescaling that happens to line up a wrong reading also counts.

Needs only the files under data/chartinfo_runs/ (no dataset ZIP).

Usage: python scripts/eval/chartinfo_rescale_gain.py
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RUNS = REPO / "data/chartinfo_runs"
sys.path.insert(0, str(REPO / "scripts/viz"))
sys.path.insert(0, str(REPO / "scripts/eval"))

from export_chartinfo_viewer import MODELS, matches  # noqa: E402

KS = range(-9, 10)


def _scaled(pred: list[dict], axis: str, k: int) -> list[dict]:
    f = 10.0**k
    out = []
    for s in pred:
        data = []
        for p in s.get("data", []):
            q = dict(p)
            if isinstance(q.get(axis), int | float) and not isinstance(q.get(axis), bool):
                q[axis] = q[axis] * f
            data.append(q)
        out.append({**s, "data": data})
    return out


def best_rescale(pred: list[dict], gt: list[dict]) -> dict:
    """Best point F1 over one-axis power-of-ten rescalings of the answer."""
    base = matches(pred, gt)[3]["f1"] if pred else 0.0
    best = {"f1": base, "axis": "", "k": 0, "f1_as_answered": base}
    if not pred:
        return best
    for axis in ("x", "y"):
        for k in KS:
            if k == 0:
                continue
            f1 = matches(_scaled(pred, axis, k), gt)[3]["f1"]
            if f1 > best["f1"] + 1e-12:
                best = {"f1": f1, "axis": axis, "k": k, "f1_as_answered": base}
    return best


def main() -> None:
    rows = []
    for batch in sorted(RUNS.glob("batch*")):
        gt_all = json.loads((batch / "ground_truth.json").read_text())
        analysis = {r["fig"]: r for r in json.loads((batch / "error_analysis.json").read_text())}
        for model in MODELS:
            path = batch / f"{model}.predictions.json"
            if not path.exists():
                continue
            preds = json.loads(path.read_text())
            for fig, gt in gt_all.items():
                pred = [s for s in preds.get(fig, []) if isinstance(s, dict)]
                b = best_rescale(pred, gt)
                a = analysis.get(fig, {})
                rows.append(
                    {
                        "batch": batch.name,
                        "fig": fig,
                        "model": model,
                        "labelled": a.get("labelled", False),
                        "dual_y": a.get("dual_y", False),
                        "dense": a.get("dense", False),
                        **b,
                    }
                )
    print(
        "| model | figures | point F1 as answered | best one-axis 10^k rescale | gain"
        " | figures gaining > 0.5 |"
    )
    print("|---|---|---|---|---|---|")
    for model in MODELS:
        r = [x for x in rows if x["model"] == model]
        n = len(r)
        base = sum(x["f1_as_answered"] for x in r) / n
        best = sum(x["f1"] for x in r) / n
        big = sum(x["f1"] - x["f1_as_answered"] > 0.5 for x in r)
        print(f"| {model} | {n} | {base:.3f} | {best:.3f} | {best - base:+.3f} | {big} |")
    print()
    print("Figures where the rescale gains > 0.5 for at least one model (axis, k per model):")
    figs: dict[tuple[str, str], list[str]] = {}
    for x in rows:
        if x["f1"] - x["f1_as_answered"] > 0.5:
            figs.setdefault((x["batch"], x["fig"]), []).append(
                f"{x['model']} {x['axis']}x10^{x['k']} {x['f1_as_answered']:.2f}->{x['f1']:.2f}"
            )
    for (batch, fig), items in sorted(figs.items()):
        print(f"- {batch} {fig}: " + "; ".join(items))


if __name__ == "__main__":
    main()
