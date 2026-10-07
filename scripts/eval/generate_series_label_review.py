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

REPO = pathlib.Path(__file__).resolve().parents[2]
GROUND_TRUTH = REPO / "data/verified_pairs/ground_truth.json"
REGISTRY = REPO / "data/verified_pairs/registry.json"
DRAFT = REPO / "data/verified_pairs/series_labels_draft.json"
DEFAULT_OUT = REPO / "build/series_label_review.html"

# 図のプロットと凡例で同じ色を使う。色覚に配慮した並び。
COLORS = [
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


def data_uri(raw: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


# ページ全体を1枚の HTML に埋め込むので、元画像はそのままだと大きすぎる
# (94図で 21MB)。凡例の文字が読めれば足りるので、長辺を抑えて JPEG にする。
MAX_EDGE = 1100
JPEG_QUALITY = 82


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


def plot_curves(entry, curves: list[dict], labels: list[dict]) -> str:
    """正解データを系列名で色分けして描く。正解は既に図の印字単位なので変換しない。"""
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
        ax.plot(
            xs,
            ys,
            marker="o",
            markersize=3.5,
            linewidth=1.1,
            color=COLORS[i % len(COLORS)],
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


def build_entries() -> list[dict]:
    ground_truth = json.loads(GROUND_TRUTH.read_text())
    draft = json.loads(DRAFT.read_text())["figures"]
    pairings = {p.figure_id: p for p in select_verified_pairings(load_registry(REGISTRY))}

    entries = []
    for figure_id, entry in sorted(pairings.items()):
        curves = [c for c in ground_truth.get(figure_id, []) if c.get("x")]
        labels = draft.get(figure_id)
        if not curves or not labels:
            continue
        labels = labels[: len(curves)]
        names = [info.get("label") for info in labels]
        entries.append(
            {
                "id": f"{entry.paper_id}-{figure_id}",
                "figure_id": figure_id,
                "paper_id": entry.paper_id,
                "panel_label": entry.panel_label or "",
                "image": source_image(entry),
                "plot": plot_curves(entry, curves, labels),
                "curves": [
                    {
                        "label": info.get("label"),
                        "source": info.get("source"),
                        "ambiguous": bool(info.get("ambiguous")),
                        "composition": info.get("composition", ""),
                        "sample_id": info.get("sample_id", ""),
                        "n_points": len(curve["x"]),
                        "color": COLORS[i % len(COLORS)],
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
