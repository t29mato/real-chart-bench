"""CHART-Infographics 2024 Train scatter and line figures -> labels.jsonl for the
local extractor (docs/design/local-model.md, 方式A v2, データ: chartinfo).

Input: CHARTINFO_2024_Train.zip (UB-PMC, CC BY-NC-SA 4.0; --zip, default
~/Downloads/CHARTINFO_2024_Train.zip). Every annotation under
annotations_JSON/scatter/ (kind scatter) and annotations_JSON/line/ (kind line)
is converted by adapter/chartinfo_training.chartinfo_label. Line vertices are
not necessarily drawn markers (extra.vertices_only); the trainer decides.

Every figure of a paper (PMCID) listed in data/chartinfo_runs/*/_key.json or
data/chartinfo_pilot/_key.json is excluded, whatever its class, and
assert_no_benchmark_leak runs before anything is written.

Output (outside the repo), training use only, never redistributed:
  ~/.cache/real-chart-bench/train-data/chartinfo/       scatter
  ~/.cache/real-chart-bench/train-data/chartinfo-line/  line
    images/<stem>.<ext> (RGB), labels.jsonl, LICENSE, ATTRIBUTION.md
and data/train_manifest/chartinfo.json.

Usage: .venv/bin/python scripts/train/prepare_chartinfo.py [--zip PATH]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from PIL import Image

from real_chart_bench.adapter.chartinfo_training import (
    LICENSE,
    chartinfo_label,
    excluded_pmcids,
    pmcid_of,
)
from real_chart_bench.domain.training_data import assert_no_benchmark_leak, validate_label

REPO = Path(__file__).resolve().parents[2]
ZIP = Path.home() / "Downloads" / "CHARTINFO_2024_Train.zip"
TRAIN_DATA = Path.home() / ".cache" / "real-chart-bench" / "train-data"
OUTS = {"scatter": TRAIN_DATA / "chartinfo", "line": TRAIN_DATA / "chartinfo-line"}
MANIFEST = REPO / "data" / "train_manifest" / "chartinfo.json"
ROOT = "CHARTINFO_2024_Train"

ATTRIBUTION = """# Attribution

- Dataset: CHART-Infographics 2024 training dataset (PubMedCentral Training Dataset
  v4.0, CHART-Info 2024), classes scatter and line
- Authors: CHART-Infographics Competition Organizers (DePaul University,
  University at Buffalo, IIIT Hyderabad, UNITEC), https://chartinfo.github.io/
- Licence: CC BY-NC-SA 4.0 (http://creativecommons.org/licenses/by-nc-sa/4.0/)
- Source of each figure: the PubMedCentral article identified by the PMCID in its
  image name; those articles were released under Creative Commons licences.
- Modified: annotations converted to the real-chart-bench training label format
  (labels.jsonl); images converted to RGB. Figures of papers used by the
  real-chart-bench measurement are removed.
- Use: training only (non-commercial). These files are not redistributed; weights
  trained on them are published under CC BY-NC-SA 4.0.
"""


def excluded_set() -> set[str]:
    keys = [json.loads(p.read_text()) for p in sorted(
        [*REPO.glob("data/chartinfo_runs/*/_key.json"), REPO / "data/chartinfo_pilot/_key.json"])
        if p.exists()]
    return excluded_pmcids(keys)


def convert(zf: zipfile.ZipFile, kind: str, excluded: set[str]) -> tuple[list[dict], dict]:
    out = OUTS[kind]
    (out / "images").mkdir(parents=True, exist_ok=True)
    images = {}
    for n in zf.namelist():
        p = PurePosixPath(n)
        if n.startswith(f"{ROOT}/images/{kind}/") and p.suffix:
            images[p.stem] = n
    anns = sorted(n for n in zf.namelist()
                  if n.startswith(f"{ROOT}/annotations_JSON/{kind}/") and n.endswith(".json"))
    stat = Counter()
    labels, ex_papers = [], set()
    for n in anns:
        stat["annotations"] += 1
        stem = PurePosixPath(n).stem
        if pmcid_of(stem) in excluded:
            stat["excluded_figures"] += 1
            ex_papers.add(pmcid_of(stem))
            continue
        try:
            ann = json.loads(zf.read(n))
        except ValueError:
            ann = None
        if not isinstance(ann, dict):
            stat["not_dict"] += 1
            continue
        if stem not in images:
            stat["no_image"] += 1
            continue
        raw = zf.read(images[stem])
        try:
            with Image.open(io.BytesIO(raw)) as im:
                width, height = im.size
                rgb = im.convert("RGB")
        except Exception:
            stat["bad_image"] += 1
            continue
        ext = PurePosixPath(images[stem]).suffix.lower()
        rel = f"images/{stem}{ext}"
        try:
            lab = chartinfo_label(ann, image=rel, width=width, height=height,
                                  stem=stem, kind=kind)
        except (KeyError, TypeError, ValueError):
            lab = None
        if lab is None:
            stat["no_label"] += 1
            continue
        if validate_label(lab):
            stat["invalid"] += 1
            continue
        if ext in (".jpg", ".jpeg"):
            rgb.save(out / rel, "JPEG", quality=95)
        else:
            rgb.save(out / rel)
        labels.append(lab)
    stat["excluded_papers"] = len(ex_papers)
    return labels, dict(stat)


def summarize(kind: str, labels: list[dict], stat: dict) -> dict:
    scales = Counter()
    for lab in labels:
        a = lab["axes"]
        scales[str((a["x"]["scale"], a["y"]["scale"])) if a else "None"] += 1
    return {
        "figures": len(labels),
        "points": sum(len(s["points_px"]) for lab in labels for s in lab["series"]),
        "papers": len({lab["paper_id"] for lab in labels}),
        "series_per_image": dict(sorted(Counter(len(lab["series"]) for lab in labels).items())),
        "axes_known": sum(lab["axes"] is not None for lab in labels),
        "axis_scales_xy": dict(scales),
        "points_value_series": sum("points_value" in s for lab in labels for s in lab["series"]),
        "dropped": stat,
    }


def disk(path: Path) -> int:
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=Path, default=ZIP)
    args = ap.parse_args()
    excluded = excluded_set()
    print(f"excluded measurement papers: {len(excluded)}", flush=True)
    manifest = {
        "source": "chartinfo",
        "path": {k: f"~/.cache/real-chart-bench/train-data/{v.name}/" for k, v in OUTS.items()},
        "license": LICENSE,
        "attribution": "CHART-Infographics 2024 (UB-PMC); https://chartinfo.github.io/ ; "
                       "training only, not redistributed",
        "made_by": "scripts/train/prepare_chartinfo.py",
        "params": {"zip": args.zip.name, "excluded_papers_from_keys": len(excluded),
                   "line_vertices_only": True},
        "kinds": {},
    }
    with zipfile.ZipFile(args.zip) as zf:
        for kind in OUTS:
            labels, stat = convert(zf, kind, excluded)
            assert_no_benchmark_leak(labels, excluded)
            out = OUTS[kind]
            with (out / "labels.jsonl").open("w") as f:
                for lab in labels:
                    f.write(json.dumps(lab, ensure_ascii=False) + "\n")
            (out / "ATTRIBUTION.md").write_text(ATTRIBUTION)
            (out / "LICENSE").write_text(
                "CHART-Infographics 2024 images and annotations: Creative Commons "
                "Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0),\n"
                "http://creativecommons.org/licenses/by-nc-sa/4.0/ . "
                "Training only; not redistributed. See ATTRIBUTION.md.\n")
            manifest["kinds"][kind] = {**summarize(kind, labels, stat), "disk_bytes": disk(out)}
            print(kind, json.dumps(manifest["kinds"][kind], ensure_ascii=False), flush=True)
    manifest["disk_bytes"] = sum(k["disk_bytes"] for k in manifest["kinds"].values())
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=1, ensure_ascii=False) + "\n")
    print("disk:", manifest["disk_bytes"], "bytes")
    return 0


if __name__ == "__main__":
    sys.exit(main())
