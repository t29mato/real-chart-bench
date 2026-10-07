"""正解データの曲線に、Starrydata の試料名(sample_name)を付ける。

これまで `ground_truth.json` の曲線は `prop_y` しか持っておらず、1図内の全曲線が
同じ文字列になっていた(論文 3.6 / 7.2)。原因は取り込みの欠落である。

**名前は `*_samples.csv` の `sample_name` を使う。** `*_curves.csv` の `composition` では
足りない。採点対象 94 図のうち 45 図は全曲線が同じ公称組成で、違うのは合成条件だけである
(例 1527: Pb 系が4本、違いはホットプレス温度)。`composition` では区別できず、
`sample_id` は DB の内部番号なので図のどこにも現れない。`sample_name` は
`HP680` `SPS760` のように**図の凡例に印字されている文字列**で、45 図すべてを区別できた。

対応づけは figure_id の中で行う。2段階:

1. x 値の完全一致。正解データは同じ CSV から作られているので、普通はここで付く。
2. 付かなければ**形で照合する**。2.6 の単位移行で、正解側の値には図の印字に合わせた
   係数が掛かっている(例 8724: K → °C)。単位の変換は一次変換なので、x と y を
   それぞれ 0〜1 に正規化すると形は変わらない。正規化した点列が一致する組を
   ハンガリアン法で1対1に割り当てる。

どちらでも付かない曲線は印を付けて残し、黙って捨てない。

**採点値は動かない。** 採点は x / y しか見ないので、このスクリプトは曲線に
フィールドを足すだけである。足した後に採点を回して確かめること。

使い方:
    python3 scripts/eval/attach_series_labels.py <curves.csv> <samples.csv>
    python3 scripts/eval/attach_series_labels.py <curves.csv> <samples.csv> --apply

CSV は公開データセットのもの(github.com/starrydata/starrydata_datasets、release latest):
  ThermoelectricMaterials_curves.csv.gz / ThermoelectricMaterials_samples.csv.gz
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import pathlib
import sys

sys.path.insert(0, "src")

from real_chart_bench.adapter.starrydata_csv import parse_curve_row

REPO = pathlib.Path(__file__).resolve().parents[2]
GROUND_TRUTH = REPO / "data/verified_pairs/ground_truth.json"
DRAFT = REPO / "data/verified_pairs/series_labels_draft.json"

csv.field_size_limit(10**8)


def read_rows(path: pathlib.Path, figure_ids: set[str]) -> dict[str, list]:
    """CSV から、欲しい figure_id の行だけを拾う。95MB あるので一度の走査で済ませる。"""
    opener = gzip.open if path.suffix == ".gz" else open
    found: dict[str, list] = {}
    with opener(path, mode="rt", newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            figure_id = (raw.get("figure_id") or "").strip()
            if figure_id not in figure_ids:
                continue
            try:
                found.setdefault(figure_id, []).append(parse_curve_row(raw))
            except ValueError:
                continue
    return found


def _shape(values: tuple[float, ...]) -> tuple[float, ...]:
    """0〜1 に正規化した形。単位の一次変換(K ↔ °C、×10^4 など)で変わらない。"""
    low, high = min(values), max(values)
    if high == low:
        return tuple(0.0 for _ in values)
    return tuple((v - low) / (high - low) for v in values)


def _ambiguous(curve: dict, siblings: list, chosen) -> bool:
    """選んだ行以外にも形が一致する行があるか。あれば割り当ては任意で、人の確認が要る。"""
    others = [
        r for r in siblings if r is not chosen and _shape_distance(curve, r) <= SHAPE_TOLERANCE
    ]
    return bool(others)


def _max_shape_gap(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    return max((abs(p - q) for p, q in zip(_shape(a), _shape(b), strict=True)), default=0.0)


def _shape_distance(curve: dict, row) -> float:
    """正規化した x と y の形の差。点数が違えば比較しない。

    y は、正解側が軸の印字に合わせて log10 に移してある図がある
    (例 51437: unit_y が "log10 of the quantity, as printed on the axis")。
    log10 は一次変換ではないので形が変わる。生の y と log10(y) の両方を試し、
    近いほうを採る。
    """
    gx = tuple(float(v) for v in curve.get("x") or ())
    gy = tuple(float(v) for v in curve.get("y") or ())
    if not gx or len(gx) != len(row.x_values) or len(gy) != len(row.y_values):
        return float("inf")
    dx = _max_shape_gap(gx, row.x_values)
    dy = _max_shape_gap(gy, row.y_values)
    if all(v > 0 for v in row.y_values):
        logged = tuple(math.log10(v) for v in row.y_values)
        dy = min(dy, _max_shape_gap(gy, logged))
    return max(dx, dy)


# 形の一致とみなす上限。94 図で実測した: 正しい組の形距離は中央値 1e-16、p90 が 4e-16 で、
# 次に大きいのは 0.24 まで飛ぶ。間に桁の空白があるので、正解側の丸め
# (29160 の y は 6 桁に丸められていて 2.4e-6 ずれる)を吸収できる 1e-5 に置く。
SHAPE_TOLERANCE = 1e-5


def match_curves(curves: list[dict], rows: list) -> list:
    """図の中で、正解曲線 -> CSV 行の 1 対 1 対応を返す(付かなければ None)。"""
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    # x だけを鍵にすると、同じ温度点で測った複数試料(1図内でよくある)が
    # 1件に潰れる。y も鍵に入れる。
    by_xy: dict[tuple, list] = {}
    for r in rows:
        by_xy.setdefault((r.x_values, r.y_values), []).append(r)
    matched: list = [None] * len(curves)
    taken: set[int] = set()
    for i, curve in enumerate(curves):
        key = (
            tuple(float(v) for v in curve.get("x") or ()),
            tuple(float(v) for v in curve.get("y") or ()),
        )
        for row in by_xy.get(key, []):
            if id(row) not in taken:
                matched[i] = row
                taken.add(id(row))
                break

    pending = [i for i, m in enumerate(matched) if m is None]
    free = [r for r in rows if id(r) not in taken]
    if not pending or not free:
        return matched

    cost = np.array([[_shape_distance(curves[i], r) for r in free] for i in pending], dtype=float)
    finite = np.isfinite(cost) & (cost <= SHAPE_TOLERANCE)
    if not finite.any():
        return matched
    big = 1e6
    for r, c in zip(*linear_sum_assignment(np.where(finite, cost, big))):
        if finite[r, c]:
            matched[pending[r]] = free[c]
    return matched


def read_samples(path: pathlib.Path) -> dict[str, dict[str, str]]:
    """sample_id -> その試料の名前と組成(*_samples.csv)。"""
    opener = gzip.open if path.suffix == ".gz" else open
    out: dict[str, dict[str, str]] = {}
    with opener(path, mode="rt", newline="", encoding="utf-8-sig") as handle:
        for raw in csv.DictReader(handle):
            sample_id = (raw.get("sample_id") or "").strip()
            if sample_id:
                out[sample_id] = {
                    "sample_name": (raw.get("sample_name") or "").strip(),
                    "composition": (raw.get("composition") or "").strip(),
                }
    return out


def label_for(row, samples: dict[str, dict[str, str]]) -> tuple[str, str]:
    """表示用のラベルと、その出どころ。

    sample_name を第一候補にする。図の凡例に印字されている文字列なので、
    図を読んでいる抽出手法の出力と突き合わせられる唯一の名前である。
    無い場合だけ composition に落ちる。
    """
    sample = samples.get(row.sample_id, {})
    name = sample.get("sample_name") or ""
    if name:
        return name, "sample_name"
    if row.composition:
        return row.composition, "composition"
    if row.sample_id:
        return f"sample {row.sample_id}", "sample_id"
    return "", "none"


def build(csv_path: pathlib.Path, samples_path: pathlib.Path) -> dict:
    ground_truth = json.loads(GROUND_TRUTH.read_text())
    rows = read_rows(csv_path, set(ground_truth))
    samples = read_samples(samples_path)

    out: dict[str, list] = {}
    unmatched = 0
    for figure_id, curves in ground_truth.items():
        siblings = rows.get(figure_id, [])
        labelled = []
        for curve, row in zip(curves, match_curves(curves, siblings), strict=True):
            if row is None:
                unmatched += 1
                labelled.append({"label": None, "source": "unmatched"})
                continue
            label, source = label_for(row, samples)
            labelled.append(
                {
                    "label": label or None,
                    "source": source,
                    "composition": samples.get(row.sample_id, {}).get(
                        "composition", row.composition
                    ),
                    "sample_id": row.sample_id,
                    # 形だけで決めた組なので、2番目の候補も許容内なら人が見るべき
                    "ambiguous": _ambiguous(curve, siblings, row),
                }
            )
        out[figure_id] = labelled
    return {"figures": out, "unmatched_curves": unmatched}


def report(draft: dict) -> None:
    figures = draft["figures"]
    curves = [c for cs in figures.values() for c in cs]
    sources: dict[str, int] = {}
    for c in curves:
        sources[c["source"]] = sources.get(c["source"], 0) + 1
    print(f"{len(figures)} 図 / {len(curves)} 曲線")
    for source, n in sorted(sources.items(), key=lambda kv: -kv[1]):
        print(f"  {source:<22}{n:>6}  ({n / len(curves):.1%})")

    ambiguous = sum(1 for c in curves if c.get("ambiguous"))
    print(f"  {'(形が曖昧で人の確認が要る)':<22}{ambiguous:>6}")
    multi = {f: cs for f, cs in figures.items() if len(cs) > 1}
    unique = sum(1 for cs in multi.values() if len({c["label"] for c in cs}) == len(cs))
    share = unique / max(len(multi), 1)
    print(f"\n複数曲線の図 {len(multi)} のうち、ラベルが全曲線で異なる: {unique} ({share:.1%})")
    short = [f for f, cs in multi.items() if len({c["label"] for c in cs}) < len(cs)]
    if short:
        print(f"  重複が残る図: {', '.join(sorted(short)[:12])}")


def apply_to_ground_truth(draft: dict) -> None:
    ground_truth = json.loads(GROUND_TRUTH.read_text())
    for figure_id, labels in draft["figures"].items():
        for curve, info in zip(ground_truth[figure_id], labels, strict=True):
            if info["label"]:
                curve["series_label"] = info["label"]
                curve["composition"] = info["composition"]
                curve["sample_id"] = info["sample_id"]
                curve["series_label_source"] = info["source"]
    GROUND_TRUTH.write_text(json.dumps(ground_truth, ensure_ascii=False, indent=2) + "\n")
    print(f"更新: {GROUND_TRUTH}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=pathlib.Path, help="Starrydata の *_curves.csv(.gz)")
    parser.add_argument("samples_path", type=pathlib.Path, help="Starrydata の *_samples.csv(.gz)")
    parser.add_argument("--apply", action="store_true", help="ground_truth.json を更新する")
    args = parser.parse_args()

    draft = build(args.csv_path, args.samples_path)
    report(draft)
    DRAFT.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n")
    print(f"\n下書き: {DRAFT}")
    if args.apply:
        apply_to_ground_truth(draft)


if __name__ == "__main__":
    main()
