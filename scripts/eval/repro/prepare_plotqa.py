"""Prepare PlotQA test-split selections (DePlot repro set + dot_line point-level set).

Downloads (gdown, Google Drive) the PlotQA *test* annotations and images, extracts only the
selected images, and writes:
  data/cache/repro/plotqa/deplot_repro.jsonl   500 plots, 125 per type (seed 20261005)
  data/cache/repro/plotqa/dot_line_100.jsonl   100 dot_line plots, stratified by #series
  data/synthetic/plotqa_dot_line/{images/,ground_truth.json,ATTRIBUTION.md}

PlotQA: Methani et al., WACV 2020 (https://github.com/NiteshMethani/PlotQA).
Annotation quirks handled here (verified on the test split):
  * axis tick/label lists are duplicated (list repeated twice) -> first half is kept.
  * vbar/line/dot_line: model.x = category labels (bar) or integer index 0..n-1 (line/dot_line;
    real labels are in x_axis.major_labels); model.y = values.
  * hbar: model.x = values, model.y = category labels (swapped); categories are on the y axis.
"""
from __future__ import annotations

import json
import random
import shutil
import subprocess
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

SEED = 20261005
ROOT = Path(__file__).resolve().parents[3]
CACHE = Path.home() / ".cache" / "real-chart-bench" / "plotqa"
OUT = ROOT / "data" / "cache" / "repro" / "plotqa"
SYN = ROOT / "data" / "synthetic" / "plotqa_dot_line"
ANN_ID = "1ikiPqkDgxNilYsU5hbK03T4_x2eDmopP"
IMG_ID = "1D_WPUy91vOrFl6cJUkE55n3ZuB6Qrc4u"
TYPES = ["vbar_categorical", "hbar_categorical", "line", "dot_line"]
NL = " <0x0A> "


def fetch(file_id: str, dest: Path) -> None:
    if dest.exists():
        return
    CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run([shutil.which("gdown") or "gdown", file_id, "-O", str(dest)], check=True)


def half(seq: list) -> list:
    """Tick/label lists in the annotations are the true list repeated twice."""
    n = len(seq)
    return seq[: n // 2] if n % 2 == 0 and seq[: n // 2] == seq[n // 2 :] else seq


def num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def cell(v) -> str:
    if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return str(v)


def categories_and_series(p: dict) -> tuple[list, list[tuple[str, list]], str]:
    """Return (category labels, [(series name, values)], category-axis label)."""
    g = p["general_figure_info"]
    ms = p["models"]
    if p["type"] == "hbar_categorical":
        cats = list(ms[0]["y"])
        series = [(m["name"], list(m["x"])) for m in ms]
        return cats, series, g["y_axis"]["label"]["text"]
    n = len(ms[0]["x"])
    if p["type"] == "vbar_categorical":
        cats = list(ms[0]["x"])
    else:  # line / dot_line: x is an index, labels in the x axis
        cats = half(g["x_axis"]["major_labels"]["values"])[:n]
    series = [(m["name"], list(m["y"])) for m in ms]
    return cats, series, g["x_axis"]["label"]["text"]


def linearise(p: dict) -> str:
    cats, series, axis_label = categories_and_series(p)
    title = (p["general_figure_info"].get("title") or {}).get("text")
    lines = []
    if title:
        lines.append(f"TITLE | {title}")
    lines.append(" | ".join([axis_label] + [s for s, _ in series]))
    for i, c in enumerate(cats):
        lines.append(" | ".join([cell(c)] + [cell(v[i]) for _, v in series]))
    return NL.join(lines)


def stratified_by_series(ps: list[dict], k: int, rng: random.Random) -> list[dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for p in ps:
        groups[len(p["models"])].append(p)
    keys = sorted(groups)
    quota = {key: 0 for key in keys}
    left = k
    while left:  # even split, spilling over when a group is exhausted
        open_ = [key for key in keys if quota[key] < len(groups[key])]
        for key in open_:
            if left and quota[key] < len(groups[key]):
                quota[key] += 1
                left -= 1
    out = []
    for key in keys:
        out += rng.sample(groups[key], quota[key])
    return sorted(out, key=lambda p: p["image_index"])


def ticks(axis: dict) -> dict:
    mt = axis["major_ticks"]
    return {
        "label": axis["label"]["text"],
        "major_tick_values": half(mt["values"]),
        "major_tick_labels": half(axis["major_labels"]["values"]),
        "major_tick_bboxes_px": half(mt["bboxes"]),
    }


def main() -> None:
    ann_path = CACHE / "test_annotations.json"
    img_arc = CACHE / "test_images.tar.gz"
    fetch(ANN_ID, ann_path)
    data = json.loads(ann_path.read_text())
    by_type = defaultdict(list)
    for p in data:
        by_type[p["type"]].append(p)
    counts = {t: len(by_type[t]) for t in TYPES}
    print("test counts:", counts, "total", len(data))

    rng = random.Random(SEED)
    deplot = []
    for t in TYPES:
        deplot += rng.sample(by_type[t], 125)
    rng = random.Random(SEED)
    dl100 = stratified_by_series(by_type["dot_line"], 100, rng)

    need = {p["image_index"] for p in deplot + dl100}
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    missing = {i for i in need if not (img_dir / f"{i}.png").exists()}
    if missing:
        fetch(IMG_ID, img_arc)
        want = {f"png/{i}.png" for i in missing}
        with tarfile.open(img_arc, "r:gz") as tf:
            for m in tf:
                if m.name in want:
                    m.name = Path(m.name).name
                    tf.extract(m, img_dir)
                    want.discard(f"png/{m.name}")
                    if not want:
                        break
        img_arc.unlink()  # archive no longer needed

    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "deplot_repro.jsonl").open("w") as f:
        for p in deplot:
            i = p["image_index"]
            f.write(json.dumps({
                "image_index": i, "image": str((img_dir / f"{i}.png").relative_to(ROOT)),
                "plot_type": p["type"], "n_series": len(p["models"]),
                "target": linearise(p),
            }, ensure_ascii=False) + "\n")

    SYN_IMG = SYN / "images"
    SYN_IMG.mkdir(parents=True, exist_ok=True)
    gt, rows = [], []
    for p in dl100:
        i = p["image_index"]
        g = p["general_figure_info"]
        shutil.copyfile(img_dir / f"{i}.png", SYN_IMG / f"{i}.png")
        cats, series, _ = categories_and_series(p)
        xnum = [num(c) for c in cats]
        rec = {
            "image_index": i,
            "image": f"images/{i}.png",
            "plot_type": "dot_line",
            "title": (g.get("title") or {}).get("text"),
            "x_is_categorical_labels": True,  # model.x is an index; real labels are strings
            "x_labels_raw": cats,
            "x_labels_numeric": xnum if all(v is not None for v in xnum) else None,
            "x_axis": ticks(g["x_axis"]),
            "y_axis": ticks(g["y_axis"]),
            "plot_bbox_px": g["plot_info"]["bbox"],
            "figure_bbox_px": g["figure_info"]["bbox"]["bbox"],
            "series": [
                {
                    "name": m["name"], "color": m["color"],
                    "x_raw": m["x"],  # as annotated (index 0..n-1)
                    "x_labels": cats,
                    "x_numeric": xnum if all(v is not None for v in xnum) else None,
                    "y": m["y"],
                    "marker_bboxes_px": m["bboxes"],
                }
                for m in p["models"]
            ],
        }
        gt.append(rec)
        rows.append({"image": f"data/synthetic/plotqa_dot_line/images/{i}.png",
                     "image_index": i, "n_series": len(p["models"])})
    (SYN / "ground_truth.json").write_text(json.dumps(gt, ensure_ascii=False, indent=1))
    with (OUT / "dot_line_100.jsonl").open("w") as f:
        for rec, row in zip(gt, rows):
            f.write(json.dumps({**row, **rec, "image": row["image"]}, ensure_ascii=False) + "\n")
    (SYN / "ATTRIBUTION.md").write_text(
        "# Attribution\n\n"
        "- Dataset: PlotQA (test split)\n"
        "- Authors: Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, Pratyush Kumar\n"
        "- Paper: \"PlotQA: Reasoning over Scientific Plots\", WACV 2020 (arXiv:1909.00997)\n"
        "- URL: https://github.com/NiteshMethani/PlotQA\n"
        "- Licence: CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)\n"
        "- Modified: selected subset (100 dot_line plots, seed 20261005), "
        "ground truth converted to JSON (ground_truth.json).\n")
    n_series = defaultdict(int)
    for p in dl100:
        n_series[len(p["models"])] += 1
    print("dot_line_100 by series:", dict(sorted(n_series.items())))


if __name__ == "__main__":
    sys.exit(main())
