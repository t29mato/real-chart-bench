"""Runs one chart-to-table model (design §7.68, §7.75 (1)) over the current
scored figures and saves its raw text output.

Inference only. The worker (scripts/eval/chart2table/worker.py) runs in the
model's own venv and appends one record per figure to
data/chart2table_predictions/<model>.jsonl (committed; resumable). Parsing the
tables and scoring are a separate step, so a parser change never needs the GPU.

Progress: one line per figure in data/cache/chart2table/<model>.log --
`tail -f` it.

Usage:
    python scripts/eval/run_chart2table.py --model deplot \\
        --python ~/.cache/real-chart-bench/chart2table/bin/python
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from run_baselines import build_dataset  # noqa: E402

from real_chart_bench.adapter.lineformer_model_runner import image_key  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
CACHE = REPO / "data/cache/chart2table"
OUT_DIR = REPO / "data/chart2table_predictions"
WORKER = REPO / "scripts/eval/chart2table/worker.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", required=True, type=pathlib.Path)
    parser.add_argument("--pythonpath", default="", help="extra PYTHONPATH (TinyChart's code)")
    parser.add_argument(
        "--dataset",
        default="real",
        choices=["real", "plotqa"],
        help="real: the scored real figures; plotqa: PlotQA dot_line 100 (design 7.75 (B))",
    )
    args = parser.parse_args()

    if args.dataset == "plotqa":
        from synthetic_plotqa import load_items

        items = load_items()
    else:
        items, _ = build_dataset()
        items = [i for i in items if not i.figure_id.startswith("synthetic-")]
    cache = CACHE if args.dataset == "real" else CACHE / args.dataset
    out_dir = OUT_DIR if args.dataset == "real" else OUT_DIR / f"synthetic-{args.dataset}"
    (cache / "images").mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = cache / "manifest.jsonl"
    with manifest.open("w") as f:
        for item in items:
            path = cache / "images" / f"{item.figure_id}.png"
            path.write_bytes(item.task.image_bytes)
            record = {
                "figure_id": item.figure_id,
                "image_path": str(path),
                "image_key": image_key(item.task.image_bytes),
            }
            f.write(json.dumps(record) + "\n")

    env = {**os.environ, "PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "PYTHONUTF8": "1"}
    if args.pythonpath:
        env["PYTHONPATH"] = args.pythonpath
    cmd = [
        str(args.python.expanduser()),
        str(WORKER),
        "--model",
        args.model,
        "--manifest",
        str(manifest),
        "--output",
        str(out_dir / f"{args.model}.jsonl"),
    ]
    log_path = cache / f"{args.model}.log"
    with log_path.open("a") as log:
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        if proc.wait() != 0:
            raise SystemExit(f"{args.model} worker exited {proc.returncode}; see {log_path}")


if __name__ == "__main__":
    main()
