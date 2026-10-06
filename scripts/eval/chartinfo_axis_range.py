"""CHART-Infographics の注釈から、軸の値の範囲を復元する。

本研究の点 F1 は τ =「軸レンジの 2%」で一致を判定する(論文 3.2)。CHART-Infographics の
注釈には軸レンジを直接表す欄がないが、**復元できる**:

- `task4.output.axes.{x,y}-axis` が目盛の**ピクセル位置**と `id` を持つ
- その `id` が `task2.output.text_blocks` の**目盛ラベルの文字列**を指す
- 文字列を数値に直せば「ピクセル位置 ↔ 値」の対応が得られ、最小・最大が軸レンジになる

実測(ICPR 2024 の scatter で使える 592 図)では 93.4% が両軸とも復元でき、
31 図が片軸のみ、8 図が復元不可だった。

これを使わずに正解点の広がりで代用すると、τ が実質的に厳しくなる。本研究の 94 図で
測ると、正解点の広がりは軸レンジの中央値 x 0.920 / y 0.787 しかないので、
τ = 2% のつもりが 1.57% 相当になっていた。
"""

from __future__ import annotations

import re

_NUMBER = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _as_number(text: str | None) -> float | None:
    """目盛ラベルの文字列を数値にする。読めないものは None。

    注釈のラベルには Unicode のマイナス、桁区切りのコンマ、単位の付記などが混ざる。
    数値として完全に読めるものだけを採用し、曖昧なものは落とす — 軸の校正に使うので、
    推測で埋めるより欠けているほうが安全である。
    """
    if not text:
        return None
    cleaned = text.strip().replace("−", "-").replace(",", "")
    match = _NUMBER.fullmatch(cleaned)
    return float(match.group()) if match else None


def axis_ranges(annotation: dict) -> dict[str, tuple[float, float]]:
    """注釈 1 件から {"x": (lo, hi), "y": (lo, hi)} を返す。復元できない軸は入らない。"""
    text_blocks = {
        block["id"]: block.get("text")
        for block in (annotation.get("task2") or {}).get("output", {}).get("text_blocks", [])
    }
    axes = (annotation.get("task4") or {}).get("output", {}).get("axes", {})

    out: dict[str, tuple[float, float]] = {}
    for key, name in (("x-axis", "x"), ("y-axis", "y")):
        values = [_as_number(text_blocks.get(tick["id"])) for tick in axes.get(key, [])]
        values = [v for v in values if v is not None]
        # 2 点以上ないと範囲にならない
        if len(values) >= 2 and max(values) > min(values):
            out[name] = (min(values), max(values))
    return out


def span_for_scoring(
    annotation: dict | None, gt_points: list[tuple[float, float]]
) -> tuple[tuple[float, float], tuple[float, float], str]:
    """正規化に使う (x幅, y幅) と、その出所を返す。

    軸レンジが復元できればそれを使い、できなければ正解点の広がりで代用する。
    どちらを使ったかを返すのは、スコアを読むときに必要だからである — 代用した図は
    τ が実質的に厳しくなっており、復元できた図と同じ土俵にない。
    """
    xs = [p[0] for p in gt_points]
    ys = [p[1] for p in gt_points]
    fallback_x = (min(xs), max(xs))
    fallback_y = (min(ys), max(ys))

    ranges = axis_ranges(annotation) if annotation else {}
    x = ranges.get("x", fallback_x)
    y = ranges.get("y", fallback_y)
    if "x" in ranges and "y" in ranges:
        source = "axis"
    elif ranges:
        source = "partial"
    else:
        source = "gt-spread"
    return x, y, source
