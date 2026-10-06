"""A stress validation set for the detector's post-processing (方式C,
docs/design/local-model.md): synthetic charts drawn by the training
generator (scripts/train/gen_synth_materials.py) with the failure modes the
method-A analysis found on real figures, so the settings that address them
are chosen on generated data, never on the benchmark:

- large markers (1.5-2.5x), half-filled (fillstyle left/right/top/bottom),
  cut into quarters by a white cross,
  or with a cross drawn inside (a patterned marker that fires several peaks);
- high-resolution renders (2-3x dpi, up to 2600 px): tick labels, titles and
  legend text at sizes the detector rarely saw.

Images use seeds 5,000,000+ (the training set uses 0-19,999), go to
~/.cache/real-chart-bench/train-data/synth-stress-val/ and are never trained on.

  .venv/bin/python scripts/train/marker_detector/gen_stress_val.py --n 300
"""

from __future__ import annotations

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts/train"))
sys.path.insert(0, str(REPO / "src"))

import gen_synth_materials as gen  # noqa: E402

from real_chart_bench.domain.training_data import validate_label  # noqa: E402

OUT = Path.home() / ".cache/real-chart-bench/train-data/synth-stress-val"
FIRST = 5_000_000


def stress_for(j: int) -> dict:
    rng = np.random.default_rng(777_000 + j)
    force: dict = {"marker_scale": float(rng.uniform(1.5, 2.5))}
    r = rng.random()
    if r < 0.2:
        force["overlay"] = str(rng.choice(["+", "x"]))
    elif r < 0.45:
        # a white cross cutting the marker into quarters (four small squares,
        # a circle with a white cross), as in several real figures
        force["overlay"] = str(rng.choice(["+", "x"]))
        force["overlay_color"] = "white"
        force["overlay_width_frac"] = float(rng.uniform(0.12, 0.25))
    elif r < 0.7:
        force["fillstyle"] = str(rng.choice(["left", "right", "top", "bottom"]))
    if rng.random() < 0.35:
        force["dpi_scale"] = float(rng.uniform(2.0, 3.0))
        force["max_px"] = (2600, 2000)
    return force


def write_one(j: int):
    force = stress_for(j)
    i, label, img, quality = gen.render(FIRST + j, force=force)
    label["source"] = "synth-stress-val"
    label["synth"]["stress"] = force
    if validate_label(label) or not label["series"]:
        return None
    path = OUT / label["image"]
    if quality:
        img.save(path, "JPEG", quality=quality)
    else:
        img.save(path, "PNG", optimize=True)
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    with Pool(16) as pool:
        labels = [lab for lab in pool.map(write_one, range(args.n)) if lab]
    with (OUT / "labels.jsonl").open("w") as f:
        for lab in labels:
            f.write(json.dumps(lab, ensure_ascii=False) + "\n")
    (OUT / "LICENSE").write_text(
        "Synthetic charts from real-chart-bench scripts/train/marker_detector/gen_stress_val.py "
        "(random data). CC BY 4.0. Validation only, never trained on.\n")
    print(f"wrote {len(labels)} labels to {OUT}")


if __name__ == "__main__":
    main()
