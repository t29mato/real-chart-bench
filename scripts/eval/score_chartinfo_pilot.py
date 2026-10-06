"""CHART-Infographics の図を、彼らの指標と本研究の指標の両方で採点する。

使い方:
    python3 scripts/eval/score_chartinfo_pilot.py <予測JSON> <正解JSON> [--key <鍵JSON>]

なぜ両方で採点するか(オーナー判断 2026-10-06):

- **彼らの指標**(`third_party/chartinfo/metric6b.py`、無改変)で測らなければ、ICPR 2020 の
  公開スコア(実図 scatter 0.232 / 0.710)と比較できない。
- **本研究の指標**(点 F1、τ = 軸レンジの 2%)で測らなければ、本研究の94図の数字と比較できず、
  再現率と適合率の内訳も見えない。

両者は測っているものが違う。彼らの式は正解点群の共分散で正規化し(図ごとに尺度が変わる)、
系列名を 25% の重みでスコアに入れ、しきい値を持たない。本研究の式は軸レンジで正規化し
(図をまたいで同じ意味)、系列名を使わず、τ でしきい値を切る。詳細は
`scripts/eval/repro/third_party/chartinfo/README.md`。
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics as st
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eval/repro/third_party/chartinfo"))
sys.path.insert(0, str(REPO / "scripts/eval"))

import metric6b  # noqa: E402  (vendored, unmodified)
from chartinfo_axis_range import span_for_scoring  # noqa: E402


def their_score(pred: list, gt: list) -> tuple[float, float, float]:
    """CHART-Infographics Task 6b のスコア。(総合, 名前スコア, データスコア) を返す。

    競技会の既定値 α=1, β₁=0.75, β₂=0.75, γ=1 を使う。これは metric6b.py の __main__ 側に
    書かれた値で、関数シグネチャの既定(0.5)とは違う。公表値と比べるには競技会側の値を使う。

    3つ組を返すのは metric_6b 自身の仕様で、論文 Table 9/10 の
    「Combined / Name / Data」の分解に対応する。先行研究と比べるべきは data-score である
    (名前スコアは凡例テキストの一致を測っており、読み取り精度ではない)。
    """
    # A predicted series with no points (the model listed a legend entry it
    # could not find) makes metric_6b's cost matrix NaN; it carries no data,
    # so it is dropped here, before the unmodified metric, and counted by the
    # caller. A prediction left with no series at all scores 0.
    pred = [s for s in pred if s.get("data")]
    if not pred:
        return 0.0, 0.0, 0.0
    combined, name, data = metric6b.metric_6b(
        pred, gt, "scatter", alpha=1, beta1=0.75, beta2=0.75, gamma=1
    )
    return float(combined), float(name), float(data)


def our_score(pred: list, gt: list, tau: float = 0.02, annotation: dict | None = None) -> dict:
    """本研究の点単位 F1。

    正規化に使う軸レンジは、注釈から復元する(`chartinfo_axis_range`)。
    `task4` の目盛ピクセル位置が `task2` の目盛ラベル文字列に id で紐づいているので、
    ラベルの最小・最大が軸レンジになる。scatter で使える 592 図のうち 93.4% が
    両軸とも復元できる。

    復元できない図では正解点の広がりで代用するが、その場合 τ は実質的に厳しくなる
    (本研究の 94 図で測ると、正解点の広がりは軸レンジの中央値 x 0.920 / y 0.787)。
    どちらを使ったかは戻り値の `span_source` に入れる。
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    def pts(series_list):
        return [
            [(float(p["x"]), float(p["y"])) for p in s.get("data", [])
             if isinstance(p.get("x"), (int, float)) and isinstance(p.get("y"), (int, float))]
            for s in series_list
        ]

    gt_sets, pred_sets = pts(gt), pts(pred)
    flat = [p for s in gt_sets for p in s]
    if not flat:
        return {"f1": float("nan"), "recall": float("nan"), "precision": float("nan"),
                "n_gt": 0, "span_source": "none"}
    (x_lo, x_hi), (y_lo, y_hi), span_source = span_for_scoring(annotation, flat)
    span_x = (x_hi - x_lo) or 1.0
    span_y = (y_hi - y_lo) or 1.0

    def norm(s):
        return np.array([[(x - x_lo) / span_x, (y - y_lo) / span_y] for x, y in s])

    # 系列の対応づけ: 組ごとの点 F1 を最大化する 1 対 1 割当(本研究 3.2 と同じ)
    n_p, n_g = len(pred_sets), len(gt_sets)
    if n_p == 0 or n_g == 0:
        return {"f1": 0.0, "recall": 0.0, "precision": 0.0, "n_gt": len(flat),
                "span_source": span_source}
    cost = np.ones((n_p, n_g))
    cache: dict[tuple[int, int], int] = {}
    for i, ps in enumerate(pred_sets):
        for j, gs in enumerate(gt_sets):
            if not ps or not gs:
                continue
            d = np.linalg.norm(norm(ps)[:, None, :] - norm(gs)[None, :, :], axis=2)
            ri, ci = linear_sum_assignment(d)
            m = int(sum(1 for r, c in zip(ri, ci) if d[r, c] <= tau))
            cache[(i, j)] = m
            f1 = 2 * m / (len(ps) + len(gs)) if (len(ps) + len(gs)) else 0.0
            cost[i, j] = 1 - f1
    ri, ci = linear_sum_assignment(cost)
    matched = sum(cache.get((r, c), 0) for r, c in zip(ri, ci))
    n_pred = sum(len(s) for s in pred_sets)
    n_gt = len(flat)
    recall = matched / n_gt if n_gt else 0.0
    precision = matched / n_pred if n_pred else 0.0
    f1 = 2 * recall * precision / (recall + precision) if (recall + precision) else 0.0
    return {"f1": f1, "recall": recall, "precision": precision, "n_gt": n_gt,
            "n_pred": n_pred, "span_source": span_source}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("predictions")
    ap.add_argument("ground_truth")
    ap.add_argument("--key")
    args = ap.parse_args()

    preds = json.loads(pathlib.Path(args.predictions).read_text())
    gts = json.loads(pathlib.Path(args.ground_truth).read_text())
    key = json.loads(pathlib.Path(args.key).read_text()) if args.key else {}

    rows = []
    header = (f"{'図':<12}{'正解':>5}{'予測':>5}{'総合':>8}{'名前':>7}{'データ':>8}"
              f"{'点F1':>7}{'再現':>7}{'適合':>7}  出典")
    print(header)
    for name in sorted(gts):
        gt = gts[name]
        pred = preds.get(name, [])
        combined, nm, data = their_score(pred, gt) if pred else (0.0, 0.0, 0.0)
        ours = our_score(pred, gt)
        src = key.get(name, {}).get("source", "").split("/")[-1][:-5].split("___")[-1][:28]
        rows.append((name, combined, nm, data, ours))
        print(
            f"{name:<12}{ours['n_gt']:>5}{ours.get('n_pred', 0):>5}{combined:>8.3f}"
            f"{nm:>7.3f}{data:>8.3f}{ours['f1']:>7.3f}{ours['recall']:>7.3f}"
            f"{ours['precision']:>7.3f}  {src}"
        )
    answered = [r for r in rows if r[4].get("n_pred", 0) > 0]
    print(
        f"\n全{len(rows)}図      総合 {st.mean(r[1] for r in rows):.4f}  "
        f"名前 {st.mean(r[2] for r in rows):.4f}  データ {st.mean(r[3] for r in rows):.4f}  |  "
        f"点F1 {st.mean(r[4]['f1'] for r in rows):.3f}  "
        f"再現 {st.mean(r[4]['recall'] for r in rows):.3f}  "
        f"適合 {st.mean(r[4]['precision'] for r in rows):.3f}"
    )
    if len(answered) != len(rows):
        print(
            f"回答した{len(answered)}図のみ  総合 {st.mean(r[1] for r in answered):.4f}  "
            f"名前 {st.mean(r[2] for r in answered):.4f}  "
            f"データ {st.mean(r[3] for r in answered):.4f}  |  "
            f"点F1 {st.mean(r[4]['f1'] for r in answered):.3f}  "
            f"再現 {st.mean(r[4]['recall'] for r in answered):.3f}  "
            f"適合 {st.mean(r[4]['precision'] for r in answered):.3f}"
        )
    print("\n参考: ICPR 2020 の実図 scatter(Task 6b、上流タスクの正解を与えた条件)")
    print("      総合 DeepBlueAI 0.232 / IntSig-SCUT-Lenovo 0.710")
    print("      ※ 先行研究の数値は全チャート型の平均で name/data の分解は scatter 単独では未公開")


if __name__ == "__main__":
    main()
