"""blob_extract against symbol_extract on the v3 synthetic stress cases
(docs/design/local-model.md「v3」), each series extracted by its own colour
inside the plot frame -- the choice an orchestrator makes -- and scored by
pixel point F1 against that series' label (radius 2% of the frame's long
side). Development data only.

  .venv/bin/python scripts/eval/orchestrator/v3_tools_dev.py [tune|check]
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.orchestrator_tools import ToolBox  # noqa: E402
from real_chart_bench.domain.marker_detection import pixel_point_f1  # noqa: E402

DEV = Path.home() / ".cache/real-chart-bench/orchestrator/dev-v3/stress"
REPORT = REPO / "data/local_orchestrator_runs/v3_tools_dev_report.json"
PCTS = (3, 8)


def one(lab: dict) -> list[dict]:
    tb = ToolBox(Path(lab["_path"]))
    x0, y0, x1, y1 = lab["plot_bbox"]
    radius = 0.02 * max(x1 - x0, y1 - y0)
    mask = {"frame": lab["plot_bbox"], "frame_margin": 3}
    short = min(x1 - x0, y1 - y0)
    rows = []

    def resolve(ref):
        raise KeyError(ref)

    for s in lab["series"]:
        truth = [tuple(p) for p in s["points_px"]]
        for pct in PCTS:
            common = {"color": s["color"], "distance_pct": pct, "mask": mask,
                      "min_diameter_px": max(3, 0.008 * short),
                      "max_diameter_px": max(8, 0.08 * short)}
            for tool in ("symbol_extract", "blob_extract"):
                r = tb.run(tool, common, resolve)
                pred = [(x, y) for ser in r["series"] for x, y in zip(ser["x"], ser["y"],
                                                                      strict=True)]
                rows.append({"kind": lab["v3"]["points"], "tool": tool, "pct": pct,
                             "f1": pixel_point_f1(pred, truth, radius)})
    return rows


def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else "tune"
    labs = []
    for line in (DEV / which / "labels.jsonl").read_text().splitlines():
        lab = json.loads(line)
        lab["_path"] = str(DEV / which / lab["image"])
        labs.append(lab)
    with ProcessPoolExecutor(16) as pool:
        rows = [r for rs in pool.map(one, labs) for r in rs]
    agg = defaultdict(list)
    for r in rows:
        agg[f"{r['kind']}|{r['tool']}|{r['pct']}"].append(r["f1"])
    out = {k: {"mean_f1": round(sum(v) / len(v), 4), "n_series": len(v)}
           for k, v in sorted(agg.items())}
    rep = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    rep[which] = out
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
