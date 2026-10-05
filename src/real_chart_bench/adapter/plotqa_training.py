"""PlotQA dot_line annotation -> one training label (docs/design/local-model.md).

PlotQA (Methani et al., WACV 2020, CC-BY-4.0) annotates every marker of a
dot_line plot with its pixel box and its data value, and every axis tick with
its box and printed label. Quirks (as in scripts/eval/repro/prepare_plotqa.py):
tick/label lists are the true list repeated twice, and a dot_line's model.x
is an index 0..n-1 whose printed label (a year) is on the x axis.

Pure conversion: no I/O. The caller reads the annotation and the image size.
"""

from __future__ import annotations

LICENSE = "CC-BY-4.0"


def _half(seq: list) -> list:
    n = len(seq)
    return seq[: n // 2] if n % 2 == 0 and seq[: n // 2] == seq[n // 2 :] else seq


def _num(s) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def _centre(b: dict) -> tuple[float, float]:
    return b["x"] + b["w"] / 2, b["y"] + b["h"] / 2


def _ticks(axis: dict, along: int) -> list[dict]:
    """(px, printed value) per major tick whose printed label is a number."""
    boxes = _half(axis["major_ticks"]["bboxes"])
    labels = _half(axis["major_labels"]["values"])
    out = []
    for box, text in zip(boxes, labels, strict=False):
        value = _num(text)
        if value is not None:
            out.append({"px": float(_centre(box)[along]), "value": value})
    return out


def _evenly_stepped(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    steps = [b - a for a, b in zip(values, values[1:], strict=False)]
    tol = 1e-9 * max(1.0, abs(steps[0]))
    return steps[0] != 0 and all(abs(st - steps[0]) <= tol for st in steps)


def plotqa_label(ann: dict, *, image: str, width: int, height: int) -> dict:
    if ann.get("type") != "dot_line":
        raise ValueError(f"only dot_line plots carry marker boxes, not {ann.get('type')}")
    g = ann["general_figure_info"]
    x_ticks = _ticks(g["x_axis"], along=0)
    y_ticks = _ticks(g["y_axis"], along=1)
    # The x axis is categorical: labels sit at equal steps whatever the years
    # are. Only evenly stepped years make it a true linear axis; otherwise the
    # x ticks are unknown (null) and only the y ticks are kept.
    x_even = _evenly_stepped([t["value"] for t in x_ticks])
    axes = None
    if x_even and len(y_ticks) >= 2:
        axes = {"x": {"scale": "linear", "ticks": x_ticks},
                "y": {"scale": "linear", "ticks": y_ticks}}

    # model.x indexes the printed x labels
    x_labels = [_num(v) for v in _half(g["x_axis"]["major_labels"]["values"])]
    series = []
    for m in ann["models"]:
        if len(m["bboxes"]) != len(m["y"]) or len(m["x"]) != len(m["y"]):
            raise ValueError(
                f"series {m.get('name')!r}: {len(m['bboxes'])} bboxes for {len(m['y'])} values"
            )
        points_px = [[float(c) for c in _centre(b)] for b in m["bboxes"]]
        xs = [x_labels[i] if 0 <= i < len(x_labels) else None for i in m["x"]]
        values = None
        if all(x is not None for x in xs) and all(_num(y) is not None for y in m["y"]):
            values = [[x, float(y)] for x, y in zip(xs, m["y"], strict=True)]
        series.append({
            "label": m.get("label") or m.get("name"),
            "marker": "circle",  # PlotQA dot_line draws filled dots only
            "filled": True,
            "color": m.get("color"),
            "points_px": points_px,
            "points_value": values,
        })

    p = g["plot_info"]["bbox"]
    return {
        "image": image,
        "width": width,
        "height": height,
        "source": "plotqa",
        "license": LICENSE,
        "paper_id": None,
        "plotqa_image_index": ann.get("image_index"),
        "axes": axes,
        "x_categorical": not x_even,
        "y_ticks": y_ticks if axes is None and len(y_ticks) >= 2 else None,
        "plot_bbox": [p["x"], p["y"], p["x"] + p["w"], p["y"] + p["h"]],
        "series": series,
    }
