"""Prepare TinyChart's ChartQA chart-to-table test items (published RMS-F1 repro).

Selection mirrors TinyChart's eval: items of mPLUG/TinyChartData test.json whose id
starts with "chartqa2table-" (run_eval.py dispatches on 'chartqa2table-').
1509 items > 1000 -> seeded random sample of 500 (seed 20261005).
Images come from ChartQA's GitHub raw test/png (the HF repo only has a 24 GB tar),
matched by basename of the item's "image" path.
"""

from __future__ import annotations

import json
import random
import urllib.parse
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

SEED = 20261005
SAMPLE_N = 500
FULL_LIMIT = 1000
RAW = "https://raw.githubusercontent.com/vis-nlp/ChartQA/main/ChartQA%20Dataset/test/png/"
OUT = Path("data/cache/repro/tinychart_chartqa")


def fetch(name: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        return
    data = urllib.request.urlopen(RAW + urllib.parse.quote(name), timeout=60).read()
    if not data.startswith(b"\x89PNG"):
        raise ValueError(f"not a PNG: {name}")
    dest.write_bytes(data)


def main() -> None:
    path = hf_hub_download("mPLUG/TinyChartData", "test.json", repo_type="dataset")
    items = [x for x in json.load(open(path)) if x["id"].startswith("chartqa2table-")]
    print(f"chartqa2table items: {len(items)}")
    if len(items) > FULL_LIMIT:
        items = random.Random(SEED).sample(items, SAMPLE_N)
        print(f"sampled {SAMPLE_N} (seed {SEED})")
    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    with open(OUT / "items.jsonl", "w", encoding="utf-8") as f:
        for it in items:
            name = Path(it["image"]).name
            fetch(name, img_dir / name)
            human, gpt = it["conversations"][0]["value"], it["conversations"][1]["value"]
            rec = {
                "id": it["id"],
                "image": str(img_dir / name),
                "source_image": it["image"],
                "prompt_raw": human,
                "prompt": human.replace("<image>\n", "", 1),
                "gt_answer": gpt,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"wrote {len(items)} items to {OUT}")


if __name__ == "__main__":
    main()
