"""Real development figures for 方式D v3 (docs/design/local-model.md
「v3: 検証の誤検知と道具の追加」): CHART-Infographics 2024 *line* charts
(PubMed Central figures, annotated by the challenge; CC BY-NC-SA 4.0, used
here for development only and never redistributed). The scatter figures
of data/chartinfo_runs/ and data/chartinfo_pilot/ are not line charts; any
name listed there is excluded all the same.

Kept: charts with axes (task 4: tick points + tick label text) whose x and y
labels are numeric (>= 3 each), and line vertices in pixels (task 6
"visual elements"). A fixed seed picks N of them. Writes, outside the repo,

  ~/.cache/real-chart-bench/orchestrator/dev-v3/chartinfo/{images/, labels.jsonl}

with labels in the labels.jsonl form (plot_bbox, series[].points_px,
axes.{x,y}.{scale, ticks[{px, value}]}).

  .venv/bin/python scripts/eval/orchestrator/chartinfo_dev.py [--n 400]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.domain.tick_calibration import parse_tick_label  # noqa: E402

ZIP = Path.home() / ".cache/real-chart-bench/chartinfo/CHARTINFO_2024_Train.zip"
OUT = Path.home() / ".cache/real-chart-bench/orchestrator/dev-v3/chartinfo"
SEED = 20261008


def excluded() -> set[str]:
    names = set()
    for key in [*REPO.glob("data/chartinfo_runs/*/_key.json"),
                REPO / "data/chartinfo_pilot/_key.json"]:
        if key.exists():
            for v in json.loads(key.read_text()).values():
                for k in ("annotation", "json", "image", "name", "source"):
                    if isinstance(v, dict) and isinstance(v.get(k), str):
                        names.add(Path(v[k]).stem)
                if isinstance(v, str):
                    names.add(Path(v).stem)
    return names


def _value(text: str) -> float | None:
    rs = [r for r in parse_tick_label(text.replace(",", "")) if r.kind == "plain"] or \
        parse_tick_label(text.replace(",", ""))
    return rs[0].value if rs else None


def _scale(ticks: list[dict]) -> str:
    vals = [t["value"] for t in sorted(ticks, key=lambda t: t["px"])]
    if len(vals) >= 3 and all(v > 0 for v in vals):
        diffs = [b - a for a, b in zip(vals, vals[1:], strict=False)]
        ratios = [b / a for a, b in zip(vals, vals[1:], strict=False)]
        lin = max(diffs) - min(diffs) <= 1e-6 * max(1.0, max(abs(d) for d in diffs))
        geo = max(ratios) - min(ratios) <= 1e-6 * max(ratios) and abs(ratios[0] - 1) > 0.5
        if geo and not lin:
            return "log"
    return "linear"


def label_of(ann: dict) -> dict | None:
    t2 = (ann.get("task2") or {}).get("output") or {}
    t4 = (ann.get("task4") or {}).get("output") or {}
    t6 = (ann.get("task6") or {}).get("output") or {}
    text = {b["id"]: b["text"] for b in t2.get("text_blocks") or []}
    axes = {}
    for ax, key, coord in (("x", "x-axis", "x"), ("y", "y-axis", "y")):
        ticks = []
        for t in (t4.get("axes") or {}).get(key) or []:
            v = _value(text.get(t.get("id"), ""))
            if v is not None and math.isfinite(v):
                ticks.append({"px": float(t["tick_pt"][coord]), "value": v})
        if len(ticks) < 3:
            return None
        axes[ax] = {"scale": _scale(ticks), "ticks": sorted(ticks, key=lambda t: t["px"])}
    lines = (t6.get("visual elements") or {}).get("lines") or []
    series = [{"points_px": [[round(p["x"], 2), round(p["y"], 2)] for p in ln]}
              for ln in lines if len(ln) >= 2]
    bb = t4.get("_plot_bb")
    if not series or not bb:
        return None
    return {"plot_bbox": [bb["x0"], bb["y0"], bb["x0"] + bb["width"], bb["y0"] + bb["height"]],
            "series": series, "axes": axes}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    z = zipfile.ZipFile(ZIP)
    skip = excluded()
    names = sorted(n for n in z.namelist()
                   if n.startswith("CHARTINFO_2024_Train/annotations_JSON/line/")
                   and n.endswith(".json"))
    random.Random(SEED).shuffle(names)
    imgs = {Path(n).stem: n for n in z.namelist()
            if n.startswith("CHARTINFO_2024_Train/images/line/")}
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    rows = []
    for n in names:
        stem = Path(n).stem
        if stem in skip or stem not in imgs:
            continue
        ann = json.loads(z.read(n))
        if not isinstance(ann, dict):
            continue
        lab = label_of(ann)
        if lab is None:
            continue
        src = imgs[stem]
        dst = OUT / "images" / Path(src).name
        dst.write_bytes(z.read(src))
        rows.append({"image": f"images/{dst.name}", "chartinfo": stem, **lab})
        if len(rows) >= args.n:
            break
    (OUT / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"{len(rows)} line charts -> {OUT}")


if __name__ == "__main__":
    main()
