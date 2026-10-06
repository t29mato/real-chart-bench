"""Scatteract の判定条件と本研究の判定条件、どちらが厳しいのか。

Scatteract(Cliche et al., ECML-PKDD 2017)式(2)と本研究の主指標(3.2)は3点が違う。
うち2つは**逆方向に働く**。

(a) 正規化: Scatteract は正解点の値幅(max-min)で割る。本研究は軸レンジで割る。
    正解点は軸いっぱいには広がらないので、値幅のほうが小さく、割った値は大きくなる
    → Scatteract のほうが**厳しい**
(b) 距離の形: Scatteract は x と y を個別に判定する(箱 = チェビシェフ距離)。
    本研究はユークリッド距離(円)。一辺 2tau の箱は半径 tau の円を完全に含む
    → Scatteract のほうが**緩い**
(c) 対応づけ: Scatteract は近い順に確定する貪欲法。本研究はハンガリアン法
    → 本研究のほうがわずかに**緩い**(取りこぼしを塞ぐ分だけ一致が増える)

どちらが勝つかは実データでしか分からないので、94図で条件を1つずつ差し替え、
「一致」と判定される点の数を数える。

正解点の値幅には2つの読み方がある。Scatteract は1系列の合成散布図を扱うので
原論文では区別がつかない。両方測る。

- 系列ごと(series): 各正解系列の自分の値幅で割る
- 図ごと(figure): 図内の全正解点の値幅で割る

系列の対応づけは、本研究の主指標(ハンガリアン法)とは独立に、
「各正解系列について、一致点が最も多くなる予測系列を採る」で揃える。
4条件すべてに同じ対応づけを当てるので、比較としては公平である。
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
MODEL = "claude-opus-5-5"
CONDITION = "noaxis"

# Scatteract から本研究へ、1条件ずつ差し替える経路。
# 各行は直前の行から1つだけ違うので、差分がその条件単独の効果になる。
VARIANTS = (
    ("box_series_greedy", "Scatteract  箱 × 値幅[系列] × 貪欲"),
    ("box_figure_greedy", "  値幅を図ごとに      箱 × 値幅[図] × 貪欲"),
    ("box_axis_greedy", "  軸レンジに          箱 × 軸レンジ × 貪欲"),
    ("circle_axis_greedy", "  円に                円 × 軸レンジ × 貪欲"),
    ("circle_axis_hungarian", "本研究      円 × 軸レンジ × ハンガリアン"),
    ("circle_series_hungarian", "(参考)    円 × 値幅[系列] × ハンガリアン"),
)


def greedy(mask: np.ndarray, dist: np.ndarray) -> int:
    """近い順に確定し、確定した点は双方から除く(Scatteract の手順)。"""
    used_pred: set[int] = set()
    used_gt: set[int] = set()
    matched = 0
    order = np.dstack(np.unravel_index(np.argsort(dist, axis=None), dist.shape))[0]
    for i, j in order:
        if not mask[i, j] or int(i) in used_pred or int(j) in used_gt:
            continue
        used_pred.add(int(i))
        used_gt.add(int(j))
        matched += 1
    return matched


def hungarian(mask: np.ndarray, dist: np.ndarray) -> int:
    """tau 以内の組の数を最大にする 1 対 1 割り当て(本研究の手順)。"""
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if len(rows) == 0:
        return 0
    sub_mask = mask[np.ix_(rows, cols)]
    sub_dist = dist[np.ix_(rows, cols)]
    penalty = sub_dist.max() + 1.0
    r, c = linear_sum_assignment(np.where(sub_mask, sub_dist, penalty))
    return int(sub_mask[r, c].sum())


def load_predictions() -> dict[str, list[dict]]:
    key = json.loads(pathlib.Path("data/llm_run_v3/_key.json").read_text())
    raw: dict[str, list[dict]] = {}
    root = pathlib.Path(f"data/llm_run_v3/{CONDITION}/{MODEL}")
    for part in sorted(root.glob("*.predictions.json")):
        raw.update(json.loads(part.read_text()))
    return {key[k]["figure_id"]: v for k, v in raw.items() if k in key}


def main() -> None:
    registry = {
        pairing.figure_id: pairing
        for pairing in select_verified_pairings(
            load_registry(pathlib.Path("data/verified_pairs/registry.json"))
        )
    }
    ground_truth = json.load(open("data/verified_pairs/ground_truth.json"))
    predictions = load_predictions()

    counts = {key: 0 for key, _ in VARIANTS}
    n_gt = 0
    for figure_id, pairing in registry.items():
        curves = [c for c in ground_truth.get(figure_id, []) if c.get("x")]
        predicted = predictions.get(figure_id)
        if not curves or not predicted:
            continue
        axis_x = (pairing.x_range[1] - pairing.x_range[0]) or 1.0
        axis_y = (pairing.y_range[1] - pairing.y_range[0]) or 1.0
        all_x = np.concatenate([np.array(c["x"], dtype=float) for c in curves])
        all_y = np.concatenate([np.array(c["y"], dtype=float) for c in curves])
        fig_x = (all_x.max() - all_x.min()) or axis_x
        fig_y = (all_y.max() - all_y.min()) or axis_y

        for curve in curves:
            gt = np.array(list(zip(curve["x"], curve["y"], strict=True)), dtype=float)
            n_gt += len(gt)
            span_x = (gt[:, 0].max() - gt[:, 0].min()) or axis_x
            span_y = (gt[:, 1].max() - gt[:, 1].min()) or axis_y
            best = {key: 0 for key, _ in VARIANTS}

            for pred_curve in predicted:
                xs = pred_curve.get("x") or []
                ys = pred_curve.get("y") or []
                if not xs:
                    continue
                pred = np.array(list(zip(xs, ys, strict=True)), dtype=float)
                dx = np.abs(pred[:, None, 0] - gt[None, :, 0])
                dy = np.abs(pred[:, None, 1] - gt[None, :, 1])
                series = (dx / span_x, dy / span_y)
                figure = (dx / fig_x, dy / fig_y)
                axis = (dx / axis_x, dy / axis_y)
                shaped = {
                    "box_series_greedy": (np.maximum(*series), greedy),
                    "box_figure_greedy": (np.maximum(*figure), greedy),
                    "box_axis_greedy": (np.maximum(*axis), greedy),
                    "circle_axis_greedy": (np.hypot(*axis), greedy),
                    "circle_axis_hungarian": (np.hypot(*axis), hungarian),
                    "circle_series_hungarian": (np.hypot(*series), hungarian),
                }
                for key, (dist, match) in shaped.items():
                    best[key] = max(best[key], match(dist <= TAU, dist))

            for key in counts:
                counts[key] += best[key]

    print(f"{MODEL}({CONDITION})、正解点 {n_gt:,} に対する一致点")
    print(f"{'条件':<44}{'一致点':>9}{'率':>8}{'直前との差':>12}")
    previous: int | None = None
    for key, label in VARIANTS:
        matched = counts[key]
        delta = "" if previous is None else f"{matched - previous:+,}"
        print(f"{label:<44}{matched:>9,}{matched / n_gt:>8.3f}{delta:>12}")
        previous = matched if key != "circle_axis_hungarian" else None


if __name__ == "__main__":
    main()
