"""labels.jsonl -> (image, prompt, target) examples for approach B
(docs/design/local-model.md, 方式B). torch-free, so it can be inspected and
counted without the GPU.

Every label passes the shared checks first: validate_label (shape) and
assert_no_benchmark_leak against every paper the benchmark registry has ever
considered. The prompt is the benchmark's own v3 single-shot prompt
(scripts/eval/local_vlm/worker_v3.build_prompt), filled with the task entry
and target answer from domain.vlm_training_example.

Usage (summary only): python examples.py <train-data dir> [...]
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import sys
from collections import Counter

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eval/local_vlm"))

from worker_v3 import build_prompt  # noqa: E402

from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.domain.training_data import (  # noqa: E402
    assert_no_benchmark_leak,
    validate_label,
)
from real_chart_bench.domain.vlm_training_example import (  # noqa: E402
    format_answer,
    noaxis_task,
    pixcal_task,
    split_of,
    target_answer,
)
from real_chart_bench.usecase.real_image_gate import benchmark_paper_ids  # noqa: E402

TRAIN_DATA = pathlib.Path.home() / ".cache/real-chart-bench/train-data"
REGISTRY = REPO / "data/verified_pairs/registry.json"
# the real figures are few (89 figures, 35 papers on 2026-10-06), so a larger
# share of their papers is held out to measure anything at all
VAL_FRACTION_BY_SOURCE = {"starrydata": 0.2}


def _unit(key: str, salt: str) -> float:
    h = hashlib.sha256(f"{salt}:{key}".encode()).digest()[:8]
    return int.from_bytes(h, "big") / 2**64


def load_examples(
    dirs: list[pathlib.Path],
    *,
    pixcal_fraction: float = 0.3,
    val_fraction: float = 0.03,
    limit_per_dir: int | None = None,
) -> tuple[list[dict], Counter]:
    """All usable examples of the given labels directories, and why the
    others were skipped."""
    bench = benchmark_paper_ids(load_registry(REGISTRY))
    examples: list[dict] = []
    skipped: Counter = Counter()
    for d in dirs:
        lines = (d / "labels.jsonl").read_text().splitlines()
        labels = [json.loads(line) for line in lines if line.strip()]
        assert_no_benchmark_leak(labels, bench)
        n = 0
        for lab in labels:
            if limit_per_dir is not None and n >= limit_per_dir:
                break
            if validate_label(lab):
                skipped[f"{d.name}: invalid label"] += 1
                continue
            key = f"{d.name}/{lab['image']}"
            fig_id = f"fig_{1 + int(_unit(key, 'fig') * 120):03d}.png"
            answer = target_answer(fig_id, lab)
            if answer is None:
                skipped[f"{d.name}: no values"] += 1
                continue
            task = None
            if _unit(key, "cond") < pixcal_fraction:
                task = pixcal_task(fig_id, lab)
            cond = "pixcal" if task is not None else "noaxis"
            if task is None:
                task = noaxis_task(fig_id)
            split_key = f"paper:{lab['paper_id']}" if lab.get("paper_id") else key
            examples.append(
                {
                    "key": key,
                    "source": lab["source"],
                    "image": str(d / lab["image"]),
                    "condition": cond,
                    "prompt": build_prompt(cond, task),
                    "target": format_answer(answer),
                    "answer": answer,
                    "fig_id": fig_id,
                    "split": split_of(
                        split_key, VAL_FRACTION_BY_SOURCE.get(lab["source"], val_fraction)
                    ),
                    "n_points": sum(len(s["x"]) for s in answer[fig_id]),
                }
            )
            n += 1
    return examples, skipped


def main() -> None:
    dirs = [pathlib.Path(a) for a in sys.argv[1:]] or sorted(
        p for p in TRAIN_DATA.iterdir() if (p / "labels.jsonl").exists()
    )
    ex, skipped = load_examples(dirs)
    by = Counter((e["source"], e["condition"], e["split"]) for e in ex)
    for k, v in sorted(by.items()):
        print(k, v)
    print("skipped", dict(skipped))
    if ex:
        print(ex[0]["prompt"][-600:])
        print(ex[0]["target"][:400])


if __name__ == "__main__":
    main()
