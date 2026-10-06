"""Prepare one 30-figure batch of the CHART-Infographics scatter measurement
(docs/handoff/2026-10-06-chartinfo-next-steps.md, owner: "30図ずつ").

Eligible figures: the CHART-Info 2024 training set's scatter charts whose
task6 has data points and a numeric x (592 per the handoff), minus the ten
pilot figures (data/chartinfo_pilot/_key.json). They are shuffled once with a
fixed seed; batch k is items [30(k-1), 30k). The order never depends on what
was run before, so batches are disjoint and reproducible.

Writes, for batch k:
  data/chartinfo_runs/batchNN/{_key.json, ground_truth.json, INSTRUCTIONS.md}
      (ground truth derived from the dataset: CC BY-NC-SA 4.0, see LICENSE.md)
  <work>/batchNN/<model>/{images/fig_01.png..., INSTRUCTIONS.md}
      one sealed directory per model, outside the repository

The prompt is data/chartinfo_pilot/prompt_v2.md (the version that states the
series-name convention) with the figure count and the directory filled in.

Usage: python scripts/eval/prepare_chartinfo_batch.py <zip> <work_dir> <k> <model> [<model> ...]
"""

from __future__ import annotations

import io
import json
import pathlib
import random
import sys
import zipfile

REPO = pathlib.Path(__file__).resolve().parents[2]
PILOT = REPO / "data/chartinfo_pilot"
OUT = REPO / "data/chartinfo_runs"
SEED = 20261006
BATCH = 30


def eligible(z: zipfile.ZipFile) -> list[str]:
    names = []
    for name in sorted(z.namelist()):
        if not (name.endswith(".json") and "/annotations_JSON/scatter/" in name):
            continue
        output = (json.loads(z.read(name)).get("task6") or {}).get("output") or {}
        pts = [
            p
            for s in output.get("data series") or []
            if isinstance(s, dict)
            for p in s.get("data") or []
        ]
        if pts and isinstance(pts[0].get("x"), int | float) and not isinstance(pts[0]["x"], bool):
            names.append(name)
    return names


def instructions(n: int, directory: pathlib.Path) -> str:
    text = (PILOT / "prompt_v2.md").read_text()
    text = text.replace("10枚", f"{n}枚").replace("キーは必ず10個", f"キーは必ず{n}個")
    return f"作業ディレクトリ: `{directory}`(この中だけで作業してください)\n\n" + text


def main() -> None:
    zpath, work, k, models = (
        pathlib.Path(sys.argv[1]),
        pathlib.Path(sys.argv[2]),
        int(sys.argv[3]),
        sys.argv[4:],
    )
    if REPO in work.resolve().parents:
        raise SystemExit("work dir must be outside the repository")
    z = zipfile.ZipFile(zpath)
    pilot = {v["source"] for v in json.loads((PILOT / "_key.json").read_text()).values()}
    pool = [n for n in eligible(z) if n not in pilot]
    random.Random(SEED).shuffle(pool)
    chosen = pool[BATCH * (k - 1) : BATCH * k]
    if not chosen:
        raise SystemExit(f"batch {k} is past the end of the {len(pool)} eligible figures")
    tag = f"batch{k:02d}"
    key, gt = {}, {}
    from PIL import Image

    images = {}
    for i, src in enumerate(chosen, 1):
        fig = f"fig_{i:02d}.png"
        img = src.replace("/annotations_JSON/", "/images/").rsplit(".", 1)[0]
        img = next(n for n in z.namelist() if n.startswith(img + ".") and "/images/" in n)
        series = json.loads(z.read(src))["task6"]["output"]["data series"]
        key[fig] = {
            "source": src,
            "image": img,
            "n_series": len(series),
            "n_points": sum(len(s.get("data") or []) for s in series),
        }
        gt[fig] = series
        buf = io.BytesIO()
        Image.open(io.BytesIO(z.read(img))).convert("RGB").save(buf, format="PNG")
        images[fig] = buf.getvalue()
    out = OUT / tag
    out.mkdir(parents=True, exist_ok=True)
    (out / "_key.json").write_text(json.dumps(key, indent=2, ensure_ascii=False) + "\n")
    (out / "ground_truth.json").write_text(json.dumps(gt, indent=2, ensure_ascii=False) + "\n")
    for model in models:
        d = (work / tag / model).resolve()
        if d.exists():
            raise SystemExit(f"{d} exists -- refusing to overwrite a run in progress")
        (d / "images").mkdir(parents=True)
        for fig, data in images.items():
            (d / "images" / fig).write_bytes(data)
        (d / "INSTRUCTIONS.md").write_text(instructions(len(chosen), d))
    (out / "INSTRUCTIONS.md").write_text(
        instructions(len(chosen), pathlib.Path("<work>/" + tag + "/<model>"))
    )
    print(f"{tag}: {len(chosen)} of {len(pool)} eligible -> {out}; models {models}")


if __name__ == "__main__":
    main()
