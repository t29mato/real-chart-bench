"""PlotQA train/val dot_line plots -> labels.jsonl for the local extractor
(docs/design/local-model.md, データ: 合成).

Inputs (fetched beforehand into ~/.cache/real-chart-bench/plotqa/):
  train_annotations.json  Google Drive 1VzWwxBVrlep17BGZU17GpLuGpwjyWbzq
  train_images.tar.gz     Google Drive 1AYuaPX-Lx7T0GZvnsPgN11Twq2FZbWXL (quota-limited);
                          when Drive refuses (quota), the mirror
                          https://huggingface.co/datasets/Dodon/plotqa-dataset
                          png_train.tar.gz (4,105,223,922 B); every image's size
                          is checked against its annotation below
  val_annotations.json    Google Drive 1CCp1tvMd62LfBrWa6pRdb1lpg68A2yFw
  val_images.tar.gz       Google Drive 1i74NRCEb-x44xqzAovuglex5d583qeiF
The PlotQA TEST split is never read here: its dot_line plots are the
synthetic benchmark (data/synthetic/plotqa_dot_line/).

Output (outside the repo): ~/.cache/real-chart-bench/train-data/plotqa/
  images/{train,val}/<image_index>.png, labels.jsonl (train), labels_val.jsonl,
  LICENSE, ATTRIBUTION.md

Every label passes validate_label, the benchmark-leak check, and a self-check:
its marker centres agree with its values through its own ticks within
MAX_RESIDUAL_PX (plots with a categorical x axis are kept, with axes null).

Usage: .venv/bin/python scripts/train/prepare_plotqa.py [--delete-archives]
"""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

from PIL import Image

from real_chart_bench.adapter.plotqa_training import plotqa_label
from real_chart_bench.domain.training_data import (
    assert_no_benchmark_leak,
    max_axis_residual_px,
    validate_label,
)

CACHE = Path.home() / ".cache" / "real-chart-bench"
SRC = CACHE / "plotqa"
OUT = CACHE / "train-data" / "plotqa"
SPLITS = {"train": "labels.jsonl", "val": "labels_val.jsonl"}
MAX_RESIDUAL_PX = 3.0

ATTRIBUTION = """# Attribution

- Dataset: PlotQA (train and validation splits; the test split is not used)
- Authors: Nitesh Methani, Pritha Ganguly, Mitesh M. Khapra, Pratyush Kumar
- Paper: "PlotQA: Reasoning over Scientific Plots", WACV 2020 (arXiv:1909.00997)
- URL: https://github.com/NiteshMethani/PlotQA
- Licence: CC-BY-4.0 (https://creativecommons.org/licenses/by/4.0/)
- Modified: dot_line plots only; annotations converted to the real-chart-bench
  training label format (labels.jsonl): marker box centres as pixel points,
  printed tick labels as axis ticks. Plots whose annotation is inconsistent
  (missing marker boxes, or centres > 3 px off their own ticks) are dropped.
"""


def convert(split: str) -> list[dict]:
    anns = json.loads((SRC / f"{split}_annotations.json").read_text())
    labels, dropped = [], {"error": 0, "invalid": 0, "residual": 0}
    for ann in anns:
        if ann["type"] != "dot_line":
            continue
        fb = ann["general_figure_info"]["figure_info"]["bbox"]["bbox"]
        i = ann["image_index"]
        try:
            lab = plotqa_label(ann, image=f"images/{split}/{i}.png",
                               width=int(fb["w"]), height=int(fb["h"]))
        except ValueError:
            dropped["error"] += 1
            continue
        if validate_label(lab):
            dropped["invalid"] += 1
            continue
        r = max_axis_residual_px(lab)
        if r is not None and r > MAX_RESIDUAL_PX:
            dropped["residual"] += 1
            continue
        lab["split"] = split
        labels.append(lab)
    print(f"{split}: {len(labels)} dot_line labels, dropped {dropped}", flush=True)
    return labels


def extract(split: str, labels: list[dict]) -> None:
    dest = OUT / "images" / split
    dest.mkdir(parents=True, exist_ok=True)
    want = {f"png/{Path(lab['image']).name}" for lab in labels}
    want = {m for m in want if not (dest / Path(m).name).exists()}
    if not want:
        return
    with tarfile.open(SRC / f"{split}_images.tar.gz", "r|gz") as tf:
        for m in tf:
            if m.name in want:
                f = tf.extractfile(m)
                (dest / Path(m.name).name).write_bytes(f.read())
                want.discard(m.name)
                if not want:
                    break
    if want:
        print(f"{split}: {len(want)} images missing from the archive", flush=True)


def finish(split: str, labels: list[dict]) -> list[dict]:
    """Keep labels whose image exists and whose size matches the annotation."""
    kept, mismatched = [], 0
    for lab in labels:
        path = OUT / lab["image"]
        if not path.exists():
            continue
        with Image.open(path) as im:
            if im.size != (lab["width"], lab["height"]):
                mismatched += 1
                path.unlink()
                continue
        kept.append(lab)
    if mismatched:
        print(f"{split}: dropped {mismatched} images whose size differs from the annotation")
    assert_no_benchmark_leak(kept, benchmark_paper_ids=set())  # no papers here
    with (OUT / SPLITS[split]).open("w") as f:
        for lab in kept:
            f.write(json.dumps(lab, ensure_ascii=False) + "\n")
    print(f"{split}: wrote {len(kept)} labels", flush=True)
    return kept


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--delete-archives", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        if not (SRC / f"{split}_annotations.json").exists():
            print(f"{split}: no annotations in {SRC}, skipped", flush=True)
            continue
        labels = convert(split)
        extract(split, labels)
        finish(split, labels)
    (OUT / "ATTRIBUTION.md").write_text(ATTRIBUTION)
    (OUT / "LICENSE").write_text(
        "PlotQA images and annotations: Creative Commons Attribution 4.0 International\n"
        "(CC-BY-4.0), https://creativecommons.org/licenses/by/4.0/legalcode\n"
        "See ATTRIBUTION.md for the source and the modifications.\n")
    if args.delete_archives:
        for split in SPLITS:
            arc = SRC / f"{split}_images.tar.gz"
            if arc.exists():
                arc.unlink()
    print("disk:", sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()), "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
