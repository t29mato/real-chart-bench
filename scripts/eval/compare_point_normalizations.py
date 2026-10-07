"""正規化の分母に何を使うべきか — 軸レンジ / 図ごとの値幅 / 系列ごとの値幅。

本研究の主指標は τ を**軸レンジ**の割合で定める。Scatteract(2017)は
**正解点の値幅**で定める。どちらが良いかを3つの性質で測る(論文 §3.5.1)。

性質1 等方性 — x と y で τ の実効的な大きさが揃うか。
    デジタイズ誤差はプロット枠上のピクセル距離で決まる。線形軸なら
    値 → ピクセルは一次変換なので「軸レンジの 2%」は「プロット枠の 2%」と
    同じであり、x と y で必ず揃う。対数軸でも log10 空間で同じことが成り立つ。
    値幅で割ると、データが枠のどれだけを占めるかに τ が左右される。
性質2 安定性 — 正解点を1つ増減したとき分母が動くか。
    分母が正解データに依存すると、正解を1点足すだけで他の全点の判定が変わる。
性質3 順位 — モデルの並び順が分母によって変わるか。

実行: PYTHONPATH= python3 scripts/eval/compare_point_normalizations.py
"""

import json
import pathlib
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "src")

from real_chart_bench.adapter.verified_pairing_registry import load_registry
from real_chart_bench.usecase.real_image_gate import select_verified_pairings

TAU = 0.02
CONDITION = "noaxis"
MODELS = ("claude-opus-5-5", "claude-fable-5-1", "claude-sonnet-5-5", "claude-haiku-4-5")
NORMS = (("axis", "軸レンジ"), ("figure", "値幅[図]"), ("series", "値幅[系列]"))

REGISTRY = pathlib.Path("data/verified_pairs/registry.json")
GROUND_TRUTH = pathlib.Path("data/verified_pairs/ground_truth.json")
RUN_ROOT = pathlib.Path("data/llm_run_v3")


def load_figures() -> list[dict]:
    """図ごとに、軸レンジ・図ごとの値幅・系列ごとの値幅をまとめて返す。"""
    registry = {p.figure_id: p for p in select_verified_pairings(load_registry(REGISTRY))}
    ground_truth = json.loads(GROUND_TRUTH.read_text())
    figures = []
    for figure_id, pairing in registry.items():
        curves = [c for c in ground_truth.get(figure_id, []) if c.get("x")]
        if not curves:
            continue
        axis = (
            (pairing.x_range[1] - pairing.x_range[0]) or 1.0,
            (pairing.y_range[1] - pairing.y_range[0]) or 1.0,
        )
        points = [np.array(list(zip(c["x"], c["y"], strict=True)), dtype=float) for c in curves]
        stacked = np.vstack(points)
        figure_span = (
            float(np.ptp(stacked[:, 0])) or axis[0],
            float(np.ptp(stacked[:, 1])) or axis[1],
        )
        figures.append(
            {
                "figure_id": figure_id,
                "axis": axis,
                "figure": figure_span,
                "curves": points,
                "series": [
                    (float(np.ptp(p[:, 0])) or axis[0], float(np.ptp(p[:, 1])) or axis[1])
                    for p in points
                ],
            }
        )
    return figures


def report_isotropy(figures: list[dict]) -> None:
    """実効 tau を軸レンジ比に換算したときの x / y の比。1.0 が理想。"""
    ratios = {key: [] for key, _ in NORMS}
    for fig in figures:
        ax, ay = fig["axis"]
        ratios["axis"].append(1.0)
        fx, fy = fig["figure"]
        ratios["figure"].append((fx / ax) / (fy / ay))
        for sx, sy in fig["series"]:
            ratios["series"].append((sx / ax) / (sy / ay))

    print("性質1 等方性 — 実効 τ の x/y 比(1.0 が理想。大きいほど y だけ厳しい)")
    print(f"  {'分母':<12}{'中央値':>9}{'p90':>9}{'最大':>10}{'2倍以上ずれる割合':>20}")
    for key, label in NORMS:
        a = np.array(ratios[key])
        skew = float((np.maximum(a, 1 / a) >= 2).mean())
        print(
            f"  {label:<12}{np.median(a):>9.2f}{np.percentile(a, 90):>9.2f}"
            f"{a.max():>10.2f}{skew:>19.1%}"
        )


def report_stability(figures: list[dict]) -> None:
    """正解の端点を1つ落としたときに分母が何%動くか。"""
    shifts = {"figure": [], "series": []}
    for fig in figures:
        stacked = np.vstack(fig["curves"])
        for axis_index in (0, 1):
            shifts["figure"].extend(_leave_one_out(stacked[:, axis_index]))
            for points in fig["curves"]:
                shifts["series"].extend(_leave_one_out(points[:, axis_index]))

    print("性質2 安定性 — 正解の端点1つを落としたときの分母の変化率")
    print(f"  {'分母':<12}{'中央値':>9}{'p90':>9}{'最大':>10}{'10%超の割合':>18}")
    print(f"  {'軸レンジ':<12}{'0.00%':>9}{'0.00%':>9}{'0.0%':>10}{'0.0%':>17}  (正解に依存しない)")
    for key in ("figure", "series"):
        a = np.array(shifts[key])
        label = dict(NORMS)[key]
        print(
            f"  {label:<12}{np.median(a):>9.2%}{np.percentile(a, 90):>9.2%}"
            f"{a.max():>10.1%}{float((a > 0.10).mean()):>17.1%}"
        )


def _leave_one_out(values: np.ndarray) -> list[float]:
    span = float(np.ptp(values))
    if span <= 0 or len(values) < 3:
        return []
    ordered = np.sort(values)
    return [abs(float(np.ptp(trimmed)) - span) / span for trimmed in (ordered[1:], ordered[:-1])]


def _matched(pred: np.ndarray, gt: np.ndarray, norm: tuple[float, float]) -> int:
    """本研究の点の対応づけ(円 × ハンガリアン法)を、与えた分母の上で行う。"""
    dx = np.abs(pred[:, None, 0] - gt[None, :, 0]) / norm[0]
    dy = np.abs(pred[:, None, 1] - gt[None, :, 1]) / norm[1]
    dist = np.hypot(dx, dy)
    mask = dist <= TAU
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if not len(rows):
        return 0
    sub_mask = mask[np.ix_(rows, cols)]
    sub_dist = dist[np.ix_(rows, cols)]
    r, c = linear_sum_assignment(np.where(sub_mask, sub_dist, sub_dist.max() + 1.0))
    return int(sub_mask[r, c].sum())


def load_predictions(model: str) -> dict[str, list[dict]] | None:
    root = RUN_ROOT / CONDITION / model
    if not root.exists():
        return None
    key = json.loads((RUN_ROOT / "_key.json").read_text())
    raw: dict[str, list[dict]] = {}
    for part in sorted(root.glob("*.predictions.json")):
        raw.update(json.loads(part.read_text()))
    return {key[k]["figure_id"]: v for k, v in raw.items() if k in key}


def report_ranking(figures: list[dict]) -> None:
    """分母だけを替えて、モデルの並び順が変わるか見る。
    系列の対応づけは「各正解系列について一致点が最も多くなる予測系列」で
    3条件に共通に当てるので、主指標の値(4章)とは一致しない。"""
    print("性質3 順位 — 一致点率(分母以外は本研究のまま)")
    header = "".join(f"{label:>12}" for _, label in NORMS)
    print(f"  {'モデル':<22}{header}")
    rates: dict[str, dict[str, float]] = {}
    for model in MODELS:
        predictions = load_predictions(model)
        if not predictions:
            print(f"  {model:<22}  予測なし")
            continue
        counts = {key: 0 for key, _ in NORMS}
        n_gt = 0
        for fig in figures:
            predicted = predictions.get(fig["figure_id"])
            if not predicted:
                continue
            for index, gt in enumerate(fig["curves"]):
                n_gt += len(gt)
                best = {key: 0 for key, _ in NORMS}
                spans = {
                    "axis": fig["axis"],
                    "figure": fig["figure"],
                    "series": fig["series"][index],
                }
                for curve in predicted:
                    xs, ys = curve.get("x") or [], curve.get("y") or []
                    if not xs:
                        continue
                    pred = np.array(list(zip(xs, ys, strict=True)), dtype=float)
                    for key in best:
                        best[key] = max(best[key], _matched(pred, gt, spans[key]))
                for key in counts:
                    counts[key] += best[key]
        rates[model] = {k: v / n_gt for k, v in counts.items()}
        row = "".join(f"{rates[model][key]:>12.3f}" for key, _ in NORMS)
        print(f"  {model:<22}{row}")

    print()
    for key, label in NORMS:
        order = sorted(rates, key=lambda m: -rates[m][key])
        print(f"  {label:<12} 順位: {' > '.join(m.removeprefix('claude-') for m in order)}")


def main() -> None:
    figures = load_figures()
    n_points = sum(len(c) for fig in figures for c in fig["curves"])
    print(
        f"{len(figures)} 図 / {sum(len(f['curves']) for f in figures)} 系列 / "
        f"{n_points:,} 点、τ = {TAU}、条件 {CONDITION}\n"
    )
    report_isotropy(figures)
    print()
    report_stability(figures)
    print()
    report_ranking(figures)


if __name__ == "__main__":
    main()
