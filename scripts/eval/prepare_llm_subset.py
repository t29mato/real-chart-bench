"""Export a 10-figure subset for LLM evaluation, with the ground truth withheld.

The owner asked how Claude Sonnet / Opus / Fable / Haiku score as extraction
models, capped at ten figures to bound cost. Those models are reached here by
spawning them as subagents, which means they run with filesystem tools -- and
`data/verified_pairs/ground_truth.json` is in this repository. A model that
reads the answer key scores perfectly and measures nothing.

So the images are copied out of the repository under opaque names (`fig_01.png`
and so on) into a scratch directory, paired with a manifest carrying only what
`ExtractionTask` gives every other baseline: the axis ranges and scales. The
mapping back to figure_id stays here, in `_key.json`, and is never handed to
the model.

That is a mitigation, not a sandbox, and the results file says so: an agent
with Bash could still reach the repository if it went looking. The honest
claim is "renamed images, no ground truth in the prompt, instructed not to
search", not "impossible to cheat".

Selection is deterministic so the subset can be regenerated and re-run: the
scoreable figures are stratified by y-axis scale in proportion to the full set
(20 of 112 are log, so 2 of 10 are), and within each stratum evenly spaced
across figure_id order. No cherry-picking of easy figures, and no randomness to
reproduce.
"""

from __future__ import annotations

import json
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.domain.curve import ScaleType  # noqa: E402
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = REPO / "data/verified_pairs/registry.json"
OUT = pathlib.Path(
    "/tmp/claude-1000/-home-mato-repos-real-chart-bench/"
    "628beb76-383b-42e9-bf21-8d4188daf8dc/scratchpad/llm_eval"
)
N = 10


def evenly_spaced(seq, k):
    """k items spread across seq, endpoints included -- deterministic."""
    if k <= 0 or not seq:
        return []
    if k >= len(seq):
        return list(seq)
    step = (len(seq) - 1) / (k - 1) if k > 1 else 0
    return [seq[round(i * step)] for i in range(k)]


def main() -> None:
    pairings = select_verified_pairings(load_registry(REGISTRY))
    pairings.sort(key=lambda p: (p.paper_id, p.figure_id))

    log_y = [p for p in pairings if p.y_scale is ScaleType.LOG]
    lin_y = [p for p in pairings if p.y_scale is not ScaleType.LOG]
    n_log = max(1, round(N * len(log_y) / len(pairings)))
    chosen = evenly_spaced(log_y, n_log) + evenly_spaced(lin_y, N - n_log)
    chosen.sort(key=lambda p: (p.paper_id, p.figure_id))
    assert len(chosen) == N, len(chosen)

    if OUT.exists():
        shutil.rmtree(OUT)
    (OUT / "images").mkdir(parents=True)

    tasks, key = [], {}
    for i, p in enumerate(chosen, start=1):
        name = f"fig_{i:02d}.png"
        src = REPO / p.image_path
        shutil.copy(src, OUT / "images" / name)
        tasks.append({
            "id": name,
            "x_range": list(p.x_range),
            "y_range": list(p.y_range),
            "x_scale": p.x_scale.value,
            "y_scale": p.y_scale.value,
        })
        key[name] = {"figure_id": p.figure_id, "paper_id": p.paper_id,
                     "panel_label": p.panel_label, "image_path": p.image_path}

    (OUT / "tasks.json").write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n")
    (OUT / "_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2) + "\n")

    print(f"{N} 図を書き出した → {OUT}")
    for t in tasks:
        k = key[t["id"]]
        print(f"   {t['id']}  paper {k['paper_id']:<7} fig {k['figure_id']:<7} "
              f"x{t['x_range']} {t['x_scale']} / y{t['y_range']} {t['y_scale']}")
    print("\n_key.json はモデルには渡さない（採点時のみ使用）")


if __name__ == "__main__":
    main()
