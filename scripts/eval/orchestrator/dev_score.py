"""Pixel point F1 of a --dev orchestrator run (Starrydata validation figures,
papers outside the benchmark), condition 2 output (pixels), all series
pooled, radius 2% of the label's plot box long side. Starrydata digitizes only
some series, so precision is understated; compare runs with each other only.

  python scripts/eval/orchestrator/dev_score.py <dev run dir> [<dev run dir> ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.domain.marker_detection import pixel_point_f1  # noqa: E402

ROOT = Path.home() / ".cache/real-chart-bench/train-data/starrydata"


def main() -> None:
    labels = {Path(lab["image"]).name: lab for lab in
              (json.loads(x) for x in (ROOT / "labels.jsonl").read_text().splitlines())}
    for run in sys.argv[1:]:
        recs = [json.loads(x) for x in (Path(run) / "pixpts_px.jsonl").read_text().splitlines()]
        f1s = []
        for r in recs:
            lab = labels[r["fig"]]
            x0, y0, x1, y1 = lab["plot_bbox"]
            truth = [tuple(p) for s in lab["series"] for p in s["points_px"]]
            pred = [(x, y) for s in (r["parsed"] or []) for x, y in zip(s["x"], s["y"],
                                                                        strict=True)]
            f1s.append(pixel_point_f1(pred, truth, 0.02 * max(x1 - x0, y1 - y0)))
        steps = sum(r["n_steps"] for r in recs) / len(recs)
        print(f"{run}: {len(recs)} figures, pooled pixel F1 {sum(f1s) / len(f1s):.3f}, "
              f"steps {steps:.1f}, finish "
              f"{sum(r['finish'] == 'final' for r in recs)} final / "
              f"{sum(r['finish'] == 'fallback' for r in recs)} fallback")


if __name__ == "__main__":
    main()
