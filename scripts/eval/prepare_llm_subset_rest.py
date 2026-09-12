"""Export every scoreable figure that has not been through the LLM eval yet.

The n=10 subset (design 7.62) was capped to bound cost. The owner has now
asked for the rest: "データ抽出してない残りのグラフ全部クロード各種で抽出して
評価して" (2026-09-12). That is 101 of the 111 scoreable figures.

Same withholding discipline as `prepare_llm_subset.py`, and the same honest
limit on it: images are copied out of the repository under opaque names
(`fig_001.png`), paired with a manifest carrying only what `ExtractionTask`
gives every other baseline (axis ranges and scales), and the mapping back to
figure_id stays here in `_key.json`. An agent with Bash could still walk into
the repository, so the claim is "renamed images, no ground truth in the
prompt, instructed not to search", not "impossible to cheat".

One thing this script fixes relative to the n=10 run: each model gets its own
sealed directory. In the first run all four models shared a working directory,
so a later model could have read an earlier one's answers. They demonstrably
did not (0-1.4% exact agreement), but the design should not have allowed it.

Selection is not stratified here -- it is *everything* not already evaluated,
so there is nothing to cherry-pick.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = REPO / "data/verified_pairs/registry.json"
KEY_DIR = REPO / "data/llm_subset_rest"
ALREADY = REPO / "data/llm_subset_n10/_key.json"
MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-fable-5", "claude-haiku-4-5"]
DEFAULT_OUT = pathlib.Path(
    "/tmp/claude-1000/-home-mato-repos-real-chart-bench/"
    "628beb76-383b-42e9-bf21-8d4188daf8dc/scratchpad/llm_eval_rest"
)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    pairings = select_verified_pairings(load_registry(REGISTRY))
    done = {v["figure_id"] for v in json.loads(ALREADY.read_text()).values()}
    rest = sorted(
        (p for p in pairings if p.figure_id not in done),
        key=lambda p: (int(p.paper_id), int(p.figure_id)),
    )

    key, tasks = {}, []
    for i, p in enumerate(rest, start=1):
        name = f"fig_{i:03d}.png"
        src = REPO / p.image_path
        key[name] = {
            "figure_id": p.figure_id,
            "paper_id": p.paper_id,
            "panel_label": p.panel_label,
            "image_path": p.image_path,
        }
        tasks.append(
            {
                "id": name,
                "x_range": list(p.x_range),
                "y_range": list(p.y_range),
                "x_scale": p.x_scale.value,
                "y_scale": p.y_scale.value,
            }
        )
        for model in MODELS:
            dst = args.out / model / "images" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst.with_suffix(src.suffix))
            if dst.with_suffix(src.suffix) != dst:
                # keep the opaque name stable regardless of the source suffix
                dst.with_suffix(src.suffix).rename(dst)

    KEY_DIR.mkdir(parents=True, exist_ok=True)
    (KEY_DIR / "_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n")
    (KEY_DIR / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
    (KEY_DIR / "predictions").mkdir(exist_ok=True)
    for model in MODELS:
        (args.out / model / "tasks.json").write_text(
            json.dumps(tasks, ensure_ascii=False, indent=2) + "\n"
        )

    print(f"{len(rest)} figures -> {args.out}/<model>/images/ for {len(MODELS)} models")
    print(f"key + tasks: {KEY_DIR}")


if __name__ == "__main__":
    main()
