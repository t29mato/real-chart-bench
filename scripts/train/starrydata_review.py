"""Review sheet for the Starrydata real-figure labels: overlay PNGs of a
random sample, for the owner to spot-check that the projected ground truth
sits on the drawn markers (docs/design/local-model.md, "データ: 実図").

Each overlay shows the crop with: the plot frame (blue), the OCR'd ticks
used for calibration (orange marks + value), and the projected Starrydata
points (red rings, one hue per series). A side panel lists the confidence.

Usage:
    python scripts/train/starrydata_review.py [--n 30] [--seed 0] [--labels FILE]
"""

from __future__ import annotations

import argparse
import html
import json
import pathlib
import random

from PIL import Image, ImageDraw

OUT = pathlib.Path.home() / ".cache/real-chart-bench/train-data/starrydata"
HUES = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6",
        "#9a6324", "#800000", "#008080"]


def overlay(label: dict, root: pathlib.Path, width: int = 900) -> Image.Image:
    im = Image.open(root / label["image"]).convert("RGB")
    scale = width / im.width
    im = im.resize((width, max(1, round(im.height * scale))), Image.LANCZOS)
    d = ImageDraw.Draw(im)
    x0, y0, x1, y1 = (v * scale for v in label["plot_bbox"])
    d.rectangle([x0, y0, x1, y1], outline="#1f77b4", width=1)
    for t in label["axes"]["x"]["ticks"]:
        px = t["px"] * scale
        d.line([px, y1, px, y1 + 14], fill="#ff8c00", width=2)
        d.text((px + 2, y1 + 14), f"{t['value']:g}", fill="#ff8c00")
    for t in label["axes"]["y"]["ticks"]:
        py = t["px"] * scale
        d.line([x0 - 14, py, x0, py], fill="#ff8c00", width=2)
        d.text((x0 + 3, py - 12), f"{t['value']:g}", fill="#ff8c00")
    for i, s in enumerate(label["series"]):
        c = HUES[i % len(HUES)]
        for x, y in s["points_px"]:
            x, y = x * scale, y * scale
            d.ellipse([x - 5, y - 5, x + 5, y + 5], outline=c, width=2)
    return im


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--labels", type=pathlib.Path, default=OUT / "labels.jsonl")
    ap.add_argument("--out", type=pathlib.Path, default=OUT / "review")
    args = ap.parse_args()
    labels = [json.loads(line) for line in args.labels.read_text().splitlines() if line]
    random.Random(args.seed).shuffle(labels)
    sample = labels[: args.n]
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for lab in sample:
        name = pathlib.Path(lab["image"]).stem + ".png"
        overlay(lab, args.labels.parent).save(args.out / name)
        e = lab["extra"]
        conf = e["confidence"]
        rows.append(
            f"<tr><td><img src='{name}' width=520></td><td><b>{html.escape(name)}</b><br>"
            f"doi: {html.escape(str(e['doi']))}<br>Fig. {html.escape(e['figure_name'])}<br>"
            f"y: {html.escape(e['prop_y'])} [{lab['axes']['y']['transform_from_starrydata']}]<br>"
            f"x: {html.escape(e['prop_x'])} [{lab['axes']['x']['transform_from_starrydata']}]<br>"
            f"hit {conf['hit']} / null {conf['null']} / margin {conf['margin']}<br>"
            f"series {len(lab['series'])}, points {sum(len(s['points_px']) for s in lab['series'])}"
            f"<br>verdict: [ ] on markers [ ] off [ ] unsure</td></tr>"
        )
    (args.out / "index.html").write_text(
        "<html><meta charset='utf-8'><body><h2>Starrydata real-figure labels: "
        f"random {len(sample)} of {len(labels)}</h2><table border=1>{''.join(rows)}</table>"
        "</body></html>"
    )
    print(args.out / "index.html")


if __name__ == "__main__":
    main()
