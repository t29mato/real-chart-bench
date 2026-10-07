"""正解曲線の色と線種を、元の図の画像から読み取る。

確認ページで正解データを描き直すとき、色が図と無関係だと凡例の突き合わせができない。
軸校正(目盛2本ずつのピクセル位置と値)があるので、正解点が画像のどこに来るかは分かる。
その位置の画素を見て、その曲線が図で何色で描かれているかを決める。

線種も同じ考えで見る。隣り合うマーカーの間を歩いて、その色のインクがどれだけ
連続しているかを数える。全部あれば実線、切れていれば破線か点線、ほとんど無ければ線なし。

使い方(確認ページから呼ばれる。単体でも動く):
    python3 scripts/eval/sample_curve_styles.py [figure_id ...]
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
from collections import Counter

import numpy as np
from PIL import Image
from scipy import ndimage

sys.path.insert(0, "src")

from real_chart_bench.domain.curve import ScaleType  # noqa: E402
from real_chart_bench.domain.pixel_calibration import PixelCalibration  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
TICKS = REPO / "data/verified_pairs/tick_calibration.json"
GROUND_TRUTH = REPO / "data/verified_pairs/ground_truth.json"

# マーカーを探す半径。図の線幅とマーカー径を考えるとこの程度で足りる。
PATCH = 5
# 線種を見るとき、マーカーの影響を避けて区間の中央寄りだけを歩く
SEGMENT_SAMPLES = 24
SEGMENT_MARGIN = 0.18
# 線は細くて滲むので、画素はマーカーより薄く出る(28498 の赤い線は (197,62,92)、
# マーカーは (204,19,42))。生の RGB 距離だと落ちるので、色みの向きで見る。
HUE_SIMILARITY = 0.85
INK_FLOOR = 40.0
GREY_CHROMA = 30.0
DARK_CEILING = 170.0
SOLID_DUTY = 0.85
DASHED_DUTY = 0.30


def calibration_for(entry: dict) -> PixelCalibration | None:
    """目盛2本ずつから、データ空間 <-> 画素の対応を作る。"""
    x_ticks, y_ticks = entry.get("x") or [], entry.get("y") or []
    if len(x_ticks) < 2 or len(y_ticks) < 2:
        return None
    (xa, xb) = sorted(x_ticks, key=lambda t: t["px"])[:2]
    y_sorted = sorted(y_ticks, key=lambda t: t["value"])
    (y_low, y_high) = y_sorted[0], y_sorted[-1]
    if xa["value"] == xb["value"] or y_low["value"] == y_high["value"]:
        return None
    return PixelCalibration(
        pixel_bbox=(xa["px"], y_high["px"], xb["px"], y_low["px"]),
        x_range=(xa["value"], xb["value"]),
        y_range=(y_low["value"], y_high["value"]),
        x_scale=ScaleType(entry.get("x_scale", "linear")),
        y_scale=ScaleType(entry.get("y_scale", "linear")),
    )


def _inkiness(patch: np.ndarray) -> np.ndarray:
    """画素ごとの「インクらしさ」。彩度が高いか暗いほど大きい。

    図の地は白か薄い灰色で、マーカーは色が付いているか黒い。どちらも拾うために
    彩度(最大 - 最小)と暗さ(255 - 最大)を足す。
    """
    high = patch.max(axis=-1).astype(float)
    low = patch.min(axis=-1).astype(float)
    return (high - low) + (255.0 - high)


def sample_point_color(image: np.ndarray, px: float, py: float) -> tuple[int, int, int] | None:
    """その位置の近傍で最もインクらしい画素の色。"""
    h, w = image.shape[:2]
    cx, cy = int(round(px)), int(round(py))
    x0, x1 = max(cx - PATCH, 0), min(cx + PATCH + 1, w)
    y0, y1 = max(cy - PATCH, 0), min(cy + PATCH + 1, h)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = image[y0:y1, x0:x1]
    score = _inkiness(patch)
    if float(score.max()) < 40.0:  # 地しかない
        return None
    index = np.unravel_index(int(np.argmax(score)), score.shape)
    return tuple(int(v) for v in patch[index])


def _modal_color(colors: list[tuple[int, int, int]]) -> tuple[int, int, int] | None:
    """量子化して最も多い色を採る。中央値より、混ざった画素に強い。"""
    if not colors:
        return None
    buckets: dict[tuple[int, int, int], list] = {}
    for c in colors:
        buckets.setdefault(tuple(v // 24 for v in c), []).append(c)
    best = max(buckets.values(), key=len)
    return tuple(int(round(sum(c[i] for c in best) / len(best))) for i in range(3))


def _is_color(pixel: np.ndarray, target: tuple[int, int, int]) -> bool:
    """その画素が、この曲線のインクか。

    明るさは比べない。線の画素は地と混ざって薄くなるが、色みの向きは変わらない。
    目標が無彩色(黒い線)のときは向きが定まらないので、暗さで見る。
    """
    p = pixel.astype(float)
    if (p.max() - p.min()) + (255.0 - p.max()) < INK_FLOOR:
        return False  # 地
    t = np.array(target, dtype=float)
    if t.max() - t.min() < GREY_CHROMA:
        return bool(p.max() < DARK_CEILING and p.max() - p.min() < GREY_CHROMA * 2)
    pv, tv = p - p.mean(), t - t.mean()
    norms = float(np.linalg.norm(pv) * np.linalg.norm(tv))
    if norms < 1e-9:
        return False
    return bool(float(pv @ tv) / norms >= HUE_SIMILARITY)


# マーカーの形を見る窓。図のマーカーはおおむね 6〜16 px。
# 窓を広げれば大きいマーカーも拾えるが、tests/fixtures/curve_style_ground_truth.json
# (12図57曲線)で測ると窓を広げるほど正解率が下がる(隣のマーカーや線を
# 拾いやすくなるため)。11px が実測で最も良かったので変えていない。
MARKER_WINDOW = 11
# 形の分類のしきい値。同じ fixture で測って決めた値(元は 0.88 / 0.66 の勘)。
# 丸と三角は外接箱に対する塗りの割合が重なる(丸 0.5〜0.9、三角 0.6〜0.7)ので、
# 重心のずれ(三角)を塗り割合より先に見る必要がある。
FILL_SQUARE = 0.88
FILL_CIRCLE = 0.48
CENTROID_OFFSET = 0.10
# マーカーの塊を線や隣のマーカーから切り離す開き処理の構造要素(8近傍、半径1)。
_BLOB_STRUCT = np.ones((3, 3), dtype=bool)


def _isolate_marker_blob(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray | None:
    """窓内の色マスクから、中心のマーカー1個だけの塊を取り出す。

    窓内の色が合う画素を全部使うと、隣のマーカー・文字・別の曲線が同じ窓に
    入ったときに紛れ込んで塗りの割合や外接箱が壊れる(21283 や 15452 の密な
    データで顕著)。まず中心画素を含む連結成分だけに絞り、次に開き処理
    (収縮してから膨張)で線の太さ(2〜3px)だけの細い部分を削ぎ、マーカー
    本体の塊だけを残す。線が太くてマーカーごと消えてしまったとき
    (小さいマーカーや細い菱形)は、削る前の連結成分に戻す。中空(縁取りだけ)
    のマーカーは穴埋めして、塗りつぶしマーカーと同じ基準で形を測れるようにする。
    """
    if mask.sum() < 6:
        return None
    labeled, _ = ndimage.label(mask, structure=_BLOB_STRUCT)
    seed_r = min(max(seed[0], 0), mask.shape[0] - 1)
    seed_c = min(max(seed[1], 0), mask.shape[1] - 1)
    seed_label = labeled[seed_r, seed_c]
    if seed_label == 0:
        rows, cols = np.nonzero(mask)
        nearest = int(np.argmin((rows - seed_r) ** 2 + (cols - seed_c) ** 2))
        seed_label = labeled[rows[nearest], cols[nearest]]
    blob = labeled == seed_label

    opened = ndimage.binary_opening(blob, structure=_BLOB_STRUCT)
    if opened.sum() >= max(6, 0.25 * blob.sum()):
        labeled2, _ = ndimage.label(opened, structure=_BLOB_STRUCT)
        if labeled2[seed_r, seed_c] != 0:
            blob = labeled2 == labeled2[seed_r, seed_c]
        else:
            rows, cols = np.nonzero(opened)
            nearest = int(np.argmin((rows - seed_r) ** 2 + (cols - seed_c) ** 2))
            blob = labeled2 == labeled2[rows[nearest], cols[nearest]]
    return ndimage.binary_fill_holes(blob)


def marker_shape(image: np.ndarray, px: float, py: float, color: tuple[int, int, int]) -> str:
    """マーカー1個の形。重心の偏りと、塗りの割合で分ける。

    中心の連結成分だけを取り出し(_isolate_marker_blob)、三角は重心が外接箱の
    縦の中心から大きくずれる性質を先に見る。残りは外接箱に対する塗りの割合で
    四角(高い)・丸(中)・菱形(低い)に分ける。
    """
    h, w = image.shape[:2]
    cx, cy = int(round(px)), int(round(py))
    x0, x1 = max(cx - MARKER_WINDOW, 0), min(cx + MARKER_WINDOW + 1, w)
    y0, y1 = max(cy - MARKER_WINDOW, 0), min(cy + MARKER_WINDOW + 1, h)
    if x1 - x0 < 5 or y1 - y0 < 5:
        return "unknown"
    patch = image[y0:y1, x0:x1]
    mask = np.array(
        [
            [_is_color(patch[r, c], color) for c in range(patch.shape[1])]
            for r in range(patch.shape[0])
        ]
    )
    blob = _isolate_marker_blob(mask, (cy - y0, cx - x0))
    if blob is None or blob.sum() < 6:
        return "unknown"
    rows, cols = np.nonzero(blob)
    height, width = int(np.ptp(rows)) + 1, int(np.ptp(cols)) + 1
    if height < 3 or width < 3:
        return "unknown"
    # 重心が外接箱の縦の中心からどれだけずれているか(0 = 中心、0.5 = 端)
    offset = abs((rows.mean() - rows.min()) / max(height - 1, 1) - 0.5)
    if offset >= CENTROID_OFFSET:
        return "triangle-down" if rows.mean() - rows.min() < (height - 1) / 2 else "triangle-up"
    fill = blob.sum() / float(height * width)
    if fill >= FILL_SQUARE:
        return "square"
    if fill >= FILL_CIRCLE:
        return "circle"
    return "diamond"


def segment_duty(
    image: np.ndarray, a: tuple[float, float], b: tuple[float, float], color: tuple[int, int, int]
) -> float | None:
    """2つのマーカーの間に、その色のインクがどれだけ乗っているか(0〜1)。"""
    h, w = image.shape[:2]
    if math.dist(a, b) < 2 * PATCH:
        return None
    hits = 0
    checked = 0
    for i in range(SEGMENT_SAMPLES):
        t = SEGMENT_MARGIN + (1 - 2 * SEGMENT_MARGIN) * i / max(SEGMENT_SAMPLES - 1, 1)
        px, py = a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t
        cx, cy = int(round(px)), int(round(py))
        x0, x1 = max(cx - 2, 0), min(cx + 3, w)
        y0, y1 = max(cy - 2, 0), min(cy + 3, h)
        if x1 <= x0 or y1 <= y0:
            continue
        checked += 1
        window = image[y0:y1, x0:x1].reshape(-1, 3)
        if any(_is_color(p, color) for p in window):
            hits += 1
    return hits / checked if checked else None


def line_style(duties: list[float]) -> str:
    if not duties:
        return "unknown"
    duty = float(np.median(duties))
    if duty >= SOLID_DUTY:
        return "solid"
    if duty >= DASHED_DUTY:
        return "dashed"
    return "none"


def majority_marker(shapes: list[str]) -> str:
    """曲線沿いの点ごとの形から、曲線全体の形を多数決で決める。

    1点1点は滲みで形が揺れるので多数決を採る。同率のときは set() 経由の
    max だとハッシュ乱択のせいで実行ごとに答えが変わりうる(再現性がない)。
    Counter.most_common は同率のとき曲線に沿って最初に出てきた形を安定して
    返す。
    """
    known = [s for s in shapes if s != "unknown"]
    if not known:
        return "unknown"
    return Counter(known).most_common(1)[0][0]


def styles_for_figure(entry: dict, curves: list[dict]) -> list[dict]:
    """図の1枚について、曲線ごとの色と線種。"""
    calibration = calibration_for(entry)
    image_path = REPO / entry["image_path"]
    if calibration is None or not image_path.exists():
        # 校正か画像が無い図。呼び出し側が key の有無を気にしなくて済むよう、
        # 採れたときと同じ形で返す。
        return [
            {
                "color": None,
                "style": "unknown",
                "marker": "unknown",
                "sampled": 0,
                "of": len(curve.get("x") or ()),
            }
            for curve in curves
        ]
    image = np.asarray(Image.open(image_path).convert("RGB"))

    out = []
    for curve in curves:
        pixels = [
            calibration.to_pixel(float(x), float(y))
            for x, y in zip(curve.get("x") or (), curve.get("y") or (), strict=True)
        ]
        colors = [c for c in (sample_point_color(image, px, py) for px, py in pixels) if c]
        color = _modal_color(colors)
        duties: list[float] = []
        shapes: list[str] = []
        if color:
            for a, b in zip(pixels, pixels[1:], strict=False):
                duty = segment_duty(image, a, b, color)
                if duty is not None:
                    duties.append(duty)
            shapes = [marker_shape(image, px, py, color) for px, py in pixels]
        out.append(
            {
                "color": f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}" if color else None,
                "style": line_style(duties) if color else "unknown",
                "marker": majority_marker(shapes),
                "sampled": len(colors),
                "of": len(pixels),
            }
        )
    return out


def load_entries() -> dict[str, dict]:
    data = json.loads(TICKS.read_text())
    return {e["figure_id"]: e for e in data["figures"]}


def main() -> None:
    entries = load_entries()
    ground_truth = json.loads(GROUND_TRUTH.read_text())
    wanted = sys.argv[1:] or sorted(entries)
    for figure_id in wanted:
        entry = entries.get(figure_id)
        curves = [c for c in ground_truth.get(figure_id, []) if c.get("x")]
        if not entry or not curves:
            print(f"{figure_id}: 校正または正解データがない")
            continue
        for curve, style in zip(curves, styles_for_figure(entry, curves), strict=True):
            print(
                f"{figure_id}  {curve.get('series_label', '?'):<18}"
                f"{style['color'] or '—':<10}{style['style']:<9}{style['marker']:<15}"
                f"{style['sampled']}/{style['of']} 点で採取"
            )


if __name__ == "__main__":
    main()
