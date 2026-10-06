"""ハンガリアン法に変えたことで、スコアは実際に変わるのか。

Scatteract は「近い順に確定して双方から除く」貪欲法で点を対応づける。本研究はこれを
ハンガリアン法に替えた。その判断を主張ではなく実測で裏づける。

同じ τ・同じ正規化のまま、点の対応づけだけを貪欲法に差し替えて全図を採点し直す。
"""
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, "src")

from real_chart_bench.adapter.verified_pairing_registry import load_registry
from real_chart_bench.usecase.real_image_gate import select_verified_pairings

TAU = 0.02


def greedy_match(d, tau):
    """Scatteract の手順: 全ペアの距離から近い順に確定し、確定した点は双方から除く。"""
    n_p, n_g = d.shape
    used_p, used_g = set(), set()
    pairs = []
    order = np.dstack(np.unravel_index(np.argsort(d, axis=None), d.shape))[0]
    for i, j in order:
        if d[i, j] > tau:
            break
        if i in used_p or j in used_g:
            continue
        used_p.add(int(i))
        used_g.add(int(j))
        pairs.append((int(i), int(j)))
    return pairs


def hungarian_match(d, tau):
    """本研究の手順: τ 以内の組の数を最大にし、その中で距離の和を最小にする。"""
    from scipy.optimize import linear_sum_assignment
    within = d <= tau
    rows = np.flatnonzero(within.any(axis=1))
    cols = np.flatnonzero(within.any(axis=0))
    if len(rows) == 0:
        return []
    sub = d[np.ix_(rows, cols)]
    sw = sub <= tau
    penalty = tau * (min(len(rows), len(cols)) + 1) + 1.0
    r, c = linear_sum_assignment(np.where(sw, sub, penalty))
    return [(int(rows[i]), int(cols[j])) for i, j in zip(r, c) if sw[i, j]]


reg = {p.figure_id: p for p in select_verified_pairings(
    load_registry(pathlib.Path("data/verified_pairs/registry.json")))}
gt_all = json.load(open("data/verified_pairs/ground_truth.json"))

results = {}
for model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"):
    preds = {}
    for part in sorted(pathlib.Path(f"data/llm_run_v3/noaxis/{model}").glob("*.predictions.json")):
        preds.update(json.loads(part.read_text()))
    if not preds:
        continue
    key = json.loads(pathlib.Path("data/llm_run_v3/_key.json").read_text())
    # fig_NNN.png -> figure_id に読み替える(鍵は封印ディレクトリの外にある)
    preds = {key[k]["figure_id"]: v for k, v in preds.items() if k in key}
    gcnt = hcnt = 0
    diff_figs = 0
    for fid, p in reg.items():
        curves = [c for c in gt_all.get(fid, []) if c.get("x")]
        pr = preds.get(fid)
        if not curves or not pr:
            continue
        xr = p.x_range[1] - p.x_range[0] or 1.0
        yr = p.y_range[1] - p.y_range[0] or 1.0
        fig_g = fig_h = 0
        for gc in curves:
            g = np.array([[(x - p.x_range[0]) / xr, (y - p.y_range[0]) / yr]
                          for x, y in zip(gc["x"], gc["y"])])
            best_g = best_h = 0
            for pc in pr:
                xs, ys = pc.get("x") or [], pc.get("y") or []
                if not xs:
                    continue
                a = np.array([[(x - p.x_range[0]) / xr, (y - p.y_range[0]) / yr]
                              for x, y in zip(xs, ys)])
                d = np.linalg.norm(a[:, None, :] - g[None, :, :], axis=2)
                best_g = max(best_g, len(greedy_match(d, TAU)))
                best_h = max(best_h, len(hungarian_match(d, TAU)))
            fig_g += best_g
            fig_h += best_h
        gcnt += fig_g
        hcnt += fig_h
        if fig_g != fig_h:
            diff_figs += 1
    results[model] = (gcnt, hcnt, diff_figs)

print(f"{'model':<20}{'貪欲で一致した点':>18}{'ハンガリアンで一致':>20}{'差':>8}{'差が出た図':>12}")
for m, (g, h, df) in results.items():
    print(f"{m:<20}{g:>18}{h:>20}{h-g:>8}{df:>12}")
print("\n(系列の対応づけは両方とも同じ。点の対応づけだけを差し替えた比較)")
