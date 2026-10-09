"""CHART-Infographics 2024 annotation -> one training label (docs/design/local-model.md).

CHART-Info 2024 Train (UB-PMC, CC BY-NC-SA 4.0) annotates PubMed Central
figures: task 2 text blocks, task 4 plot box and tick points (each tick
names the text block of its printed label), task 5 legend marker boxes, and
task 6 the data series plus their pixel "visual elements" (scatter points,
or line vertices). Line vertices are not necessarily drawn markers, so a
line chart's label says so (extra.vertices_only) and the trainer decides.

Every figure of a paper (PMCID) that the CHART-Info measurement uses is
excluded from training (excluded_pmcids + assert_no_benchmark_leak).

Pure conversion: no I/O. The caller reads the annotation and the image size.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import PurePosixPath

from real_chart_bench.domain.tick_calibration import parse_tick_label

LICENSE = "CC-BY-NC-SA-4.0"
SOURCE = "chartinfo"


def pmcid_of(name: str) -> str:
    """'…/PMC1173100___1471-2156-6-24-1.json' -> 'PMC1173100'."""
    return PurePosixPath(name).name.split("___")[0]


def excluded_pmcids(keys: Iterable[dict]) -> set[str]:
    """PMCIDs of every figure listed in the measurement keys (_key.json)."""
    out = set()
    for key in keys:
        for v in key.values():
            refs = v.values() if isinstance(v, dict) else [v]
            for ref in refs:
                if isinstance(ref, str) and "___" in ref:
                    out.add(pmcid_of(ref))
    return out


def _value(text: str) -> float | None:
    readings = parse_tick_label(text.replace(",", ""))
    plain = [r for r in readings if r.kind == "plain"] or readings
    v = plain[0].value if plain else None
    return v if v is not None and math.isfinite(v) else None


def _scale(values: list[float]) -> str:
    if len(values) >= 3 and all(v > 0 for v in values):
        diffs = [b - a for a, b in zip(values, values[1:], strict=False)]
        ratios = [b / a for a, b in zip(values, values[1:], strict=False)]
        lin = max(diffs) - min(diffs) <= 1e-6 * max(1.0, max(abs(d) for d in diffs))
        geo = max(ratios) - min(ratios) <= 1e-6 * max(ratios) and abs(ratios[0] - 1) > 0.5
        if geo and not lin:
            return "log"
    return "linear"


def _axes(t2: dict, t4: dict) -> dict | None:
    text = {b.get("id"): b.get("text", "") for b in t2.get("text_blocks") or []}
    axes = {}
    for name, key in (("x", "x-axis"), ("y", "y-axis")):
        ticks = []
        for t in (t4.get("axes") or {}).get(key) or []:
            v = _value(text.get(t.get("id"), ""))
            if v is not None:
                ticks.append({"px": float(t["tick_pt"][name]), "value": v})
        if len(ticks) < 2:
            return None
        ticks.sort(key=lambda t: t["px"])
        values = [t["value"] for t in ticks]
        if name == "y":
            values = values[::-1]  # pixel y grows downwards
        axes[name] = {"scale": _scale(values), "ticks": ticks}
    return axes


def chartinfo_label(ann: dict, *, image: str, width: int, height: int, stem: str,
                    kind: str) -> dict | None:
    """One labels.jsonl line, or None when the figure has no usable points."""
    t2 = (ann.get("task2") or {}).get("output") or {}
    t4 = (ann.get("task4") or {}).get("output") or {}
    t5 = (ann.get("task5") or {}).get("output") or {}
    t6 = (ann.get("task6") or {}).get("output") or {}
    elements = (t6.get("visual elements") or {}).get(
        "scatter points" if kind == "scatter" else "lines") or []
    data = [s for s in t6.get("data series") or [] if isinstance(s, dict)]
    series = []
    for i, el in enumerate(elements):
        pts = [[float(p["x"]), float(p["y"])] for p in el
               if isinstance(p, dict) and 0 <= p.get("x", -1) <= width
               and 0 <= p.get("y", -1) <= height]
        if not pts:
            continue
        s: dict = {"label": None, "marker": None, "filled": None, "color": None,
                   "points_px": pts}
        if i < len(data):
            s["label"] = data[i].get("name")
            vals = [p for p in data[i].get("data") or [] if isinstance(p, dict)]
            if (kind == "scatter" and len(vals) == len(pts) == len(el)
                    and all(isinstance(p.get(c), int | float) and not isinstance(p.get(c), bool)
                            for p in vals for c in ("x", "y"))):
                s["points_value"] = [[float(p["x"]), float(p["y"])] for p in vals]
        series.append(s)
    if not series:
        return None
    bb = t4.get("_plot_bb")
    legend = [[b["bb"]["x0"], b["bb"]["y0"], b["bb"]["x0"] + b["bb"]["width"],
               b["bb"]["y0"] + b["bb"]["height"]]
              for b in t5.get("legend_pairs") or [] if isinstance(b, dict) and b.get("bb")]
    return {
        "image": image, "width": width, "height": height,
        "source": SOURCE, "license": LICENSE, "paper_id": pmcid_of(stem),
        "axes": _axes(t2, t4),
        "plot_bbox": ([bb["x0"], bb["y0"], bb["x0"] + bb["width"], bb["y0"] + bb["height"]]
                      if bb else None),
        "series": series,
        "extra": {"chartinfo": stem, "chart_type": kind,
                  "vertices_only": kind != "scatter", "legend_bboxes": legend},
    }
