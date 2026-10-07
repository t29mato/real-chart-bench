"""チャート→表の指標で、対応づけの取り方が順位を変えるか(論文 4.9)。

RMS は見出し(x の値 + 系列名)の文字列が最も近い三つ組どうしを組にする。
`rms_f1_value_only` は見出しを**採点から**外すが、**対応づけからは**外していない。
本コーパスには行見出しに使えるものがないので(3.5.3)、見出し駆動の対応づけは
ほぼ任意の組を作ってしまう。

値の得点(軸レンジ正規化)はそのままに、対応づけだけを見出し駆動と値駆動で
入れ替えて、順位が変わるかを測る。変わるなら、順位の入れ替わりはモデルの
性質ではなく指標の性質である。

実行: PYTHONPATH= python3 scripts/eval/compare_table_assignment.py
"""

import json
import pathlib
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "src")
sys.path.insert(0, "scripts/eval")

from real_chart_bench.adapter.verified_pairing_registry import load_registry
from real_chart_bench.domain.curve import Curve
from real_chart_bench.domain.point_metrics import AxisFrame
from real_chart_bench.domain.table_metrics import (
    NUMBER_THETA,
    TEXT_THETA,
    VALUE_NORM_AXIS_RANGE,
    _value_score,
    anls,
    entries_from_curves,
)
from real_chart_bench.usecase.real_image_gate import select_verified_pairings

reg = {
    p.figure_id: p
    for p in select_verified_pairings(
        load_registry(pathlib.Path("data/verified_pairs/registry.json"))
    )
}
gt_all = json.loads(pathlib.Path("data/verified_pairs/ground_truth.json").read_text())
key = json.loads(pathlib.Path("data/llm_run_v3/_key.json").read_text())


def preds_for(model, root="data/llm_run_v3/noaxis"):
    raw = {}
    for part in sorted(pathlib.Path(f"{root}/{model}").glob("*.predictions.json")):
        raw.update(json.loads(part.read_text()))
    return {key[k]["figure_id"]: v for k, v in raw.items() if k in key}


def table_score(pred_curves, gt_curves, frame, assign):
    g = entries_from_curves(gt_curves)
    p = entries_from_curves(pred_curves)
    if not g or not p:
        return 0.0
    header = np.array([[anls(a.key, b.key, TEXT_THETA) for b in p] for a in g])
    value = np.array(
        [
            [_value_score(a.value, b.value, VALUE_NORM_AXIS_RANGE, frame, NUMBER_THETA) for b in p]
            for a in g
        ]
    )
    cost = (1.0 - header) - 1e-9 * value if assign == "header" else 1.0 - value
    r, c = linear_sum_assignment(cost)
    score = float(value[r, c].sum())
    if score <= 0:
        return 0.0
    prec, rec = score / len(p), score / len(g)
    return 2 * prec * rec / (prec + rec)


models = {
    "Opus 5.5": "claude-opus-5-5",
    "GPT-6.1-Sol": None,
    "Fable 5.1": "claude-fable-5-1",
    "Sonnet 5.5": "claude-sonnet-5-5",
}
preds = {}
for label, m in models.items():
    if m:
        preds[label] = preds_for(m)
preds["GPT-6.1-Sol"] = preds_for("gpt-6.1-sol", "data/llm_run_codex/noaxis")

print(f"{'モデル':<14}{'見出し駆動':>12}{'値駆動':>12}")
out = {}
for label, pr in preds.items():
    tot = {"header": [], "value": []}
    for fid, pairing in reg.items():
        gc = [c for c in gt_all.get(fid, []) if c.get("x")]
        pc = pr.get(fid)
        if not gc or not pc:
            continue
        frame = AxisFrame(
            x_range=pairing.x_range,
            y_range=pairing.y_range,
            x_scale=pairing.x_scale,
            y_scale=pairing.y_scale,
        )
        gcur = [
            Curve(
                x_values=c["x"],
                y_values=c["y"],
                x_scale=pairing.x_scale,
                series_label=c.get("series_label") or "",
            )
            for c in gc
        ]
        pcur = []
        for c in pc:
            xs, ys = c.get("x") or [], c.get("y") or []
            if xs:
                pcur.append(
                    Curve(
                        x_values=xs,
                        y_values=ys,
                        x_scale=pairing.x_scale,
                        series_label=c.get("label") or "",
                    )
                )
        if not pcur:
            continue
        for mode in ("header", "value"):
            tot[mode].append(table_score(pcur, gcur, frame, mode))
    out[label] = (float(np.mean(tot["header"])), float(np.mean(tot["value"])))
    print(f"{label:<14}{out[label][0]:>12.4f}{out[label][1]:>12.4f}")
print()
for i, mode in enumerate(("見出し駆動", "値駆動")):
    print(f"  {mode}: {' > '.join(sorted(out, key=lambda m: -out[m][i]))}")
