"""Why figures fail (CHART-Infographics batches): per figure, its features and,
per model, a failure category from the scores themselves -- no judgement by
eye. Writes <batch>/error_analysis.{json,md}.

Features (from the ground truth and its annotation):
  dense        median same-series nearest-neighbour spacing < 2 tau (design 7.72),
               in the space normalised by the ground-truth spread
  dual_y       the annotation has a second y axis (task4 "y-axis-2")
  log_x/log_y  a log10 map fits the (pixel, value) pairs better than a linear one
  labelled     at least half the series hold a single point (points labelled in
               the plot rather than through a legend)

Scores per model: point F1 (one-to-one, series paired), the same with all
series pooled into one ("series-agnostic"), and curve F1 (design 7.83).
Category, for point F1 < 0.8, first rule that applies:
  dense            -> 密集(曲線 F1 で見るべき)
  agnostic >= F1 + 0.25 -> 系列の分け方・対応
  dual_y           -> 2軸の読み違い
  n_pred < 0.7 n_gt -> 点の見逃し;  n_pred > 1.4 n_gt -> 余分な点
  otherwise        -> 値のずれ(目盛・位置)

Usage: python scripts/eval/analyze_chartinfo_errors.py <zip> <batch_dir>
"""

from __future__ import annotations

import json
import pathlib
import sys
import zipfile

import numpy as np

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/viz"))
sys.path.insert(0, str(REPO / "scripts/eval"))

from export_chartinfo_viewer import MODELS, fit_axis, matches  # noqa: E402

from real_chart_bench.domain.point_metrics import curve_f1  # noqa: E402

TAU = 0.02


def norm_sets(series, lo, span):
    out = []
    for s in series:
        pts = [
            (p["x"], p["y"])
            for p in s.get("data", [])
            if isinstance(p.get("x"), int | float) and isinstance(p.get("y"), int | float)
        ]
        out.append(
            np.array([[(x - lo[0]) / span[0], (y - lo[1]) / span[1]] for x, y in pts]).reshape(
                -1, 2
            )
        )
    return out


def category(f, m):
    if m["f1"] >= 0.8:
        return ""
    if f["dense"]:
        return "密集(曲線 F1 で見るべき)"
    if m["agnostic_f1"] >= m["f1"] + 0.25:
        return "系列の分け方・対応"
    if f["dual_y"]:
        return "2軸の読み違い"
    if m["n_pred"] < 0.7 * m["n_gt"]:
        return "点の見逃し"
    if m["n_pred"] > 1.4 * m["n_gt"]:
        return "余分な点"
    return "値のずれ(目盛・位置)"


def main() -> None:
    z = zipfile.ZipFile(sys.argv[1])
    batch = pathlib.Path(sys.argv[2])
    key = json.loads((batch / "_key.json").read_text())
    gt_all = json.loads((batch / "ground_truth.json").read_text())
    preds = {
        m: json.loads((batch / f"{m}.predictions.json").read_text())
        for m in MODELS
        if (batch / f"{m}.predictions.json").exists()
    }
    rows = []
    for fig in sorted(key):
        gt = gt_all[fig]
        ann = json.loads(z.read(key[fig]["source"]))
        axes = ((ann.get("task4") or {}).get("output") or {}).get("axes") or {}
        vis = (ann["task6"]["output"].get("visual elements") or {}).get("scatter points") or []
        px, py = [], []
        for si, s in enumerate(gt):
            if si < len(vis) and len(vis[si]) == len(s["data"]):
                px += [(p["x"], v["x"]) for p, v in zip(s["data"], vis[si])]
                py += [(p["y"], v["y"]) for p, v in zip(s["data"], vis[si])]
        fx = fit_axis([a for a, _ in px], [b for _, b in px]) if len(px) >= 2 else None
        fy = fit_axis([a for a, _ in py], [b for _, b in py]) if len(py) >= 2 else None
        flat = [(p["x"], p["y"]) for s in gt for p in s["data"]]
        lo = (min(x for x, _ in flat), min(y for _, y in flat))
        span = ((max(x for x, _ in flat) - lo[0]) or 1.0, (max(y for _, y in flat) - lo[1]) or 1.0)
        g = norm_sets(gt, lo, span)
        nn = []
        for a in g:
            if len(a) >= 2:
                d = np.sqrt(((a[:, None] - a[None]) ** 2).sum(axis=2))
                np.fill_diagonal(d, np.inf)
                nn += d.min(axis=1).tolist()
        f = {
            "fig": fig,
            "source": key[fig]["image"].split("/")[-1],
            "n_series": len(gt),
            "n_points": len(flat),
            "dense": bool(nn) and float(np.median(nn)) < 2 * TAU,
            "dual_y": bool(axes.get("y-axis-2")),
            "log_x": bool(fx and fx[0] == "log"),
            "log_y": bool(fy and fy[0] == "log"),
            "labelled": len(gt) >= 3 and sum(len(s["data"]) == 1 for s in gt) >= len(gt) / 2,
            "models": {},
        }
        for m, P in preds.items():
            pred = [s for s in P.get(fig, []) if isinstance(s, dict)]
            _, _, _, ours = matches(pred, gt)
            pooled = [{"data": [p for s in pred for p in s.get("data", [])]}]
            _, _, _, agn = matches(pooled, [{"data": [p for s in gt for p in s["data"]]}])
            cf = curve_f1(norm_sets(pred, lo, span), g, TAU)
            rec = {
                "f1": ours["f1"],
                "agnostic_f1": agn["f1"],
                "curve_f1": cf["f1"],
                "n_gt": ours["n_gt"],
                "n_pred": ours["n_pred"],
            }
            rec["category"] = category(f, rec)
            f["models"][m] = rec
        rows.append(f)
    (batch / "error_analysis.json").write_text(
        json.dumps(rows, indent=1, ensure_ascii=False) + "\n"
    )

    def mean(xs):
        xs = list(xs)
        return sum(xs) / len(xs) if xs else float("nan")

    lines = [
        "# 失敗の分析(自動分類)",
        "",
        f"{batch.name}、{len(rows)}図。"
        "分類規則は `scripts/eval/analyze_chartinfo_errors.py` の docstring。",
        "",
    ]
    lines += [
        "## 図の特徴",
        "",
        f"- 密集: {sum(r['dense'] for r in rows)}図 / "
        f"2本の y 軸: {sum(r['dual_y'] for r in rows)}図 / "
        f"log 軸: {sum(r['log_x'] or r['log_y'] for r in rows)}図 / "
        f"点ごとのラベル: {sum(r['labelled'] for r in rows)}図",
        "",
    ]
    lines += [
        "## モデル別(点 F1 は密でない図、曲線 F1 は密な図)",
        "",
        "| モデル | 点 F1(密でない) | 系列を無視した点 F1 | 曲線 F1(密) |",
        "|---|---|---|---|",
    ]
    for m in preds:
        nd = [r["models"][m] for r in rows if not r["dense"]]
        de = [r["models"][m] for r in rows if r["dense"]]
        lines.append(
            f"| {m} | {mean(x['f1'] for x in nd):.3f}({len(nd)}図) | "
            f"{mean(x['agnostic_f1'] for x in nd):.3f} | "
            f"{mean(x['curve_f1'] for x in de):.3f}({len(de)}図) |"
        )
    lines += [
        "",
        "## 失敗の分類(点 F1 < 0.8、図 × モデル)",
        "",
        "| 分類 | " + " | ".join(preds) + " |",
        "|---|" + "---|" * len(preds),
    ]
    cats = sorted({r["models"][m]["category"] for r in rows for m in preds} - {""})
    for c in cats:
        lines.append(
            f"| {c} | "
            + " | ".join(str(sum(r["models"][m]["category"] == c for r in rows)) for m in preds)
            + " |"
        )
    lines += [
        "",
        "## 図ごと",
        "",
        "| 図 | 出典 | 特徴 | " + " | ".join(preds) + " |",
        "|---|---|---|" + "---|" * len(preds),
    ]
    for r in rows:
        feats = (
            ", ".join(k for k in ("dense", "dual_y", "log_x", "log_y", "labelled") if r[k]) or "—"
        )
        cells = []
        for m in preds:
            x = r["models"][m]
            cells.append(f"{x['f1']:.2f}" + (f" {x['category']}" if x["category"] else ""))
        lines.append(
            f"| {r['fig']} | {r['source'].split('___')[-1][:28]} | {feats} | "
            + " | ".join(cells)
            + " |"
        )
    (batch / "error_analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines[:30]))


if __name__ == "__main__":
    main()
