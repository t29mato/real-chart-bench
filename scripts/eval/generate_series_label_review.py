"""系列名(composition)の目視確認ページを作る。

`attach_series_labels.py` が正解データの各曲線に Starrydata の試料名を付けた。
付け方は figure_id の中での形の一致なので、**どの曲線がどの試料かは機械が決めている**。
図の凡例と突き合わせて人が確かめる必要がある。

1図につき、元の図と、系列名で色分けした正解データのプロットを並べる。
オーナーは曲線ごとに ok / 違う / 判断できない を付け、JSON で書き出す。

出力は1枚の HTML(画像は data: URI で埋め込む)。スマホで見られるようにしている。

使い方:
    python3 scripts/eval/generate_series_label_review.py [出力先.html]
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, "src")

from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sample_curve_styles import (  # noqa: E402
    calibration_for,
    load_entries,
    styles_for_figure,
)

REPO = pathlib.Path(__file__).resolve().parents[2]
GROUND_TRUTH = REPO / "data/verified_pairs/ground_truth.json"
REGISTRY = REPO / "data/verified_pairs/registry.json"
DRAFT = REPO / "data/verified_pairs/series_labels_draft.json"
DEFAULT_OUT = REPO / "build/series_label_review.html"

# 図から色が採れなかったときだけ使う控えの並び
FALLBACK_COLORS = [
    "#0d7a6f",
    "#d14310",
    "#6b4de6",
    "#b45309",
    "#1f6fd0",
    "#b3261e",
    "#2e7d32",
    "#8e24aa",
    "#00838f",
    "#ef6c00",
]
# 図から読んだマーカーの形を matplotlib の記号に移す
MARKERS = {
    "circle": "o",
    "square": "s",
    "diamond": "D",
    "triangle-up": "^",
    "triangle-down": "v",
    "unknown": "o",
}
LINE_STYLES = {"solid": "-", "dashed": "--", "none": "", "unknown": "-"}


def data_uri(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


# ページ全体を1枚の HTML に埋め込むので、元画像はそのままだと大きすぎる
# (94図で 21MB)。凡例の文字が読めれば足りるので、長辺を抑えて JPEG にする。
MAX_EDGE = 1000
JPEG_QUALITY = 74
# ラベルを書き込んだ図は元図より小さくてよい(凡例ではなく位置関係を見るため)
ANNOTATED_EDGE = 820
# 名前の札を置くために画像の右に足す余白(画像幅に対する割合)
LABEL_MARGIN = 0.22


def source_image(entry) -> str | None:
    path = REPO / entry.image_path
    if not path.exists():
        return None
    image = Image.open(path)
    if max(image.size) > MAX_EDGE:
        scale = MAX_EDGE / max(image.size)
        image = image.resize(
            (round(image.width * scale), round(image.height * scale)), Image.LANCZOS
        )
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return data_uri(buf.getvalue(), "image/jpeg")


def plot_curves(entry, curves: list[dict], labels: list[dict], styles: list[dict]) -> str:
    """正解データを、元の図から読んだ色・マーカー・線種で描き直す。

    正解は既に図の印字単位なので値の変換はしない。色が図と揃っていれば、
    凡例を見比べるだけで対応が確かめられる。
    """
    fig, ax = plt.subplots(figsize=(5.0, 3.7), dpi=110)
    x_log = entry.x_scale.value == "log"
    y_log = entry.y_scale.value == "log"
    for i, (curve, info) in enumerate(zip(curves, labels, strict=True)):
        xs = list(curve["x"])
        ys = list(curve["y"])
        if y_log:
            kept = [(x, y) for x, y in zip(xs, ys, strict=True) if y > 0]
            xs, ys = [p[0] for p in kept], [p[1] for p in kept]
        if x_log:
            kept = [(x, y) for x, y in zip(xs, ys, strict=True) if x > 0]
            xs, ys = [p[0] for p in kept], [p[1] for p in kept]
        style = styles[i] if i < len(styles) else {}
        line = LINE_STYLES.get(style.get("style"), "-")
        ax.plot(
            xs,
            ys,
            marker=MARKERS.get(style.get("marker"), "o"),
            markersize=4.5,
            linewidth=1.3,
            linestyle=line or "None",
            color=style.get("color") or FALLBACK_COLORS[i % len(FALLBACK_COLORS)],
            label=info.get("label") or f"(unnamed #{i + 1})",
        )
    ax.set_xscale(entry.x_scale.value)
    ax.set_yscale(entry.y_scale.value)
    try:
        ax.set_xlim(entry.x_range)
        ax.set_ylim(entry.y_range)
    except ValueError:
        pass
    first = curves[0] if curves else {}
    ax.set_xlabel(f"{first.get('prop_x', 'x')} ({first.get('unit_x', '')})", fontsize=8)
    ax.set_ylabel(f"{first.get('prop_y', 'y')} ({first.get('unit_y', '')})", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6.5, loc="best")
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return data_uri(buf.getvalue(), "image/png")


def annotated_figure(
    entry, pairing, curves: list[dict], labels: list[dict], styles: list[dict]
) -> str | None:
    """元の図の上に、曲線ごとの試料名をその曲線の位置に書き込む。

    凡例と色で突き合わせる手間を省く。どの曲線がどの名前かが図の上で直接読める。
    """
    calibration = calibration_for(entry)
    path = REPO / entry["image_path"]
    if calibration is None or not path.exists():
        return None
    image = Image.open(path).convert("RGB")
    scale = min(1.0, ANNOTATED_EDGE / max(image.size))
    width, height = round(image.width * scale), round(image.height * scale)
    # 名前は曲線の右端に置くので、画像の右に札の分の余白を作る
    margin = round(width * LABEL_MARGIN)
    fig, ax = plt.subplots(figsize=((width + margin) / 110, height / 110), dpi=110)
    ax.imshow(image.resize((width, height), Image.LANCZOS))
    ax.set_xlim(0, width + margin)
    ax.set_ylim(height, 0)
    ax.axis("off")

    for i, (curve, info) in enumerate(zip(curves, labels, strict=True)):
        style = styles[i] if i < len(styles) else {}
        color = style.get("color") or FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
        points = [
            calibration.to_pixel(float(x), float(y))
            for x, y in zip(curve.get("x") or (), curve.get("y") or (), strict=True)
        ]
        if not points:
            continue
        ax.plot(
            [p[0] * scale for p in points],
            [p[1] * scale for p in points],
            marker="o",
            markersize=3.0,
            linewidth=0.0,
            color=color,
            markerfacecolor="none",
            markeredgewidth=1.0,
        )
        anchor = max(points, key=lambda p: p[0])
        ax.annotate(
            info.get("label") or f"#{i + 1}",
            xy=(anchor[0] * scale, anchor[1] * scale),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=6.5,
            color="white",
            va="center",
            bbox={
                "boxstyle": "round,pad=0.22",
                "facecolor": color,
                "edgecolor": "none",
                "alpha": 0.92,
            },
            clip_on=False,
        )
    fig.tight_layout(pad=0.1)
    buf = io.BytesIO()
    fig.savefig(
        buf,
        format="jpeg",
        pil_kwargs={"quality": JPEG_QUALITY},
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close(fig)
    return data_uri(buf.getvalue(), "image/jpeg")


def shown_hash(entry, curves: list[dict], labels: list[dict], styles: list[dict]) -> str:
    """この図で人に見せた判断材料のハッシュ(design scaling-verification §4.4)。

    判定を「何を見せたか」に紐付けるためのもの。規則や採点が変わっても絵が
    変わらなければハッシュは動かず、**再レビューに回す必要がない**。
    逆に単位移行や軸校正のやり直しでハッシュが変われば、その図だけが再キューに入る。
    設計の見積りでは、過去7回の全件見直しのうち人が見る必要があったのは実質3回・
    計55図だった。

    材料に入れるのは、人が実際に判断に使うものだけである。点の値、付けた名前、
    図から読んだ色とマーカー、そして元画像。採点の仕様や指標は入れない — それらが
    変わっても絵は1画素も動かないからである。
    """
    parts: list[str] = [entry["image_path"]]
    image = REPO / entry["image_path"]
    if image.exists():
        parts.append(hashlib.sha256(image.read_bytes()).hexdigest())
    for axis in ("x", "y"):
        parts += [f"{t['px']}:{t['value']}" for t in entry.get(axis) or []]
    for curve, info, style in zip(curves, labels, styles, strict=True):
        parts.append(str(info.get("label")))
        parts.append(f"{style.get('color')}/{style.get('marker')}/{style.get('style')}")
        parts.append(repr([round(float(v), 6) for v in curve.get("x") or ()]))
        parts.append(repr([round(float(v), 6) for v in curve.get("y") or ()]))
    return hashlib.sha256("\u0000".join(parts).encode("utf-8")).hexdigest()[:16]


def build_entries() -> list[dict]:
    ground_truth = json.loads(GROUND_TRUTH.read_text())
    draft = json.loads(DRAFT.read_text())["figures"]
    pairings = {p.figure_id: p for p in select_verified_pairings(load_registry(REGISTRY))}
    tick_entries = load_entries()

    entries = []
    for figure_id, entry in sorted(pairings.items()):
        curves = [c for c in ground_truth.get(figure_id, []) if c.get("x")]
        labels = draft.get(figure_id)
        if not curves or not labels:
            continue
        labels = labels[: len(curves)]
        names = [info.get("label") for info in labels]
        tick_entry = tick_entries.get(figure_id)
        styles = (
            styles_for_figure(tick_entry, curves)
            if tick_entry
            else [{"color": None, "style": "unknown", "marker": "unknown"} for _ in curves]
        )
        entries.append(
            {
                "id": f"{entry.paper_id}-{figure_id}",
                "figure_id": figure_id,
                "paper_id": entry.paper_id,
                "panel_label": entry.panel_label or "",
                "shown_sha": shown_hash(
                    tick_entry or {"image_path": entry.image_path}, curves, labels, styles
                ),
                "image": source_image(entry),
                "annotated": (
                    annotated_figure(tick_entry, entry, curves, labels, styles)
                    if tick_entry
                    else None
                ),
                "plot": plot_curves(entry, curves, labels, styles),
                "curves": [
                    {
                        "label": info.get("label"),
                        "source": info.get("source"),
                        "ambiguous": bool(info.get("ambiguous")),
                        "composition": info.get("composition", ""),
                        "sample_id": info.get("sample_id", ""),
                        "n_points": len(curve["x"]),
                        "color": (
                            styles[i].get("color") or FALLBACK_COLORS[i % len(FALLBACK_COLORS)]
                        ),
                        "color_from_figure": bool(styles[i].get("color")),
                        "marker": styles[i].get("marker", "unknown"),
                        "line": styles[i].get("style", "unknown"),
                    }
                    for i, (curve, info) in enumerate(zip(curves, labels, strict=True))
                ],
                # 人が先に見るべき図: 名前が付かなかった / 名前が重複している
                "needs_attention": (any(n is None for n in names) or len(set(names)) != len(names)),
            }
        )
    return entries


def main() -> None:
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    entries = build_entries()
    template = (pathlib.Path(__file__).parent / "_series_label_review_template.html").read_text()
    page = template.replace("__ENTRIES_JSON__", json.dumps(entries, ensure_ascii=False))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    attention = sum(1 for e in entries if e["needs_attention"])
    print(f"{out}  ({out.stat().st_size / 1e6:.1f} MB, {len(entries)} 図)")
    print(f"  要確認(名前なし or 重複): {attention} 図")


if __name__ == "__main__":
    main()
