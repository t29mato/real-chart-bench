"""Runs pretrained LineFormer on the current verified set on a local GPU and
writes results/lineformer-pretrained-n{N}.json
(docs/handoff/2026-10-01-lineformer-rerun-issue.md).

Two stages, so a rescoring never needs the GPU again:

  1. inference -- scripts/eval/lineformer/worker.py under LineFormer's own
     Python 3.10 env (built by scripts/eval/lineformer/setup_local.sh) writes
     raw pixel-space series to data/lineformer_predictions/pretrained-n{N}.jsonl
     (committed; resumable -- figures already in it are skipped).
  2. scoring -- the same build_dataset()/run() as run_baselines.py, with
     adapter.lineformer_model_runner replaying stage 1's output. Same figure
     set, same ExtractionTask calibration, same payload as every other row.

Progress: every figure is logged to data/cache/lineformer/run.log as it
finishes -- `tail -f data/cache/lineformer/run.log`.

Usage:
    python scripts/eval/run_lineformer.py                 # inference + scoring
    python scripts/eval/run_lineformer.py --score-only    # rescore saved output
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

from run_baselines import RESULTS_DIR, build_dataset, run  # noqa: E402

from real_chart_bench.adapter.lineformer_model_runner import (  # noqa: E402
    LineFormerPrediction,
    PrecomputedLineFormerModelRunner,
    image_key,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CACHE_DIR = REPO_ROOT / "data/cache/lineformer"
PREDICTIONS_DIR = REPO_ROOT / "data/lineformer_predictions"
LOG_PATH = CACHE_DIR / "run.log"
WORKER = REPO_ROOT / "scripts/eval/lineformer/worker.py"
DEFAULT_LF_HOME = pathlib.Path.home() / ".cache/real-chart-bench/lineformer"


def _write_manifest(items) -> pathlib.Path:
    images_dir = CACHE_DIR / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = CACHE_DIR / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for item in items:
            image_path = images_dir / f"{item.figure_id}.img"
            image_path.write_bytes(item.task.image_bytes)
            record = {
                "figure_id": item.figure_id,
                "image_path": str(image_path),
                "image_key": image_key(item.task.image_bytes),
            }
            f.write(json.dumps(record) + "\n")
    return manifest_path


def _run_inference(items, predictions_path: pathlib.Path, lf_home: pathlib.Path) -> None:
    manifest_path = _write_manifest(items)
    src = lf_home / "LineFormer"
    cmd = [
        str(lf_home / "venv/bin/python"),
        str(WORKER),
        "--lineformer-src",
        str(src),
        "--config",
        str(src / "lineformer_swin_t_config.py"),
        "--checkpoint",
        str(src / "checkpoints/iter_3000.pth"),
        "--manifest",
        str(manifest_path),
        "--output",
        str(predictions_path),
    ]
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1"}
    with LOG_PATH.open("a") as log:
        proc = subprocess.Popen(
            cmd, cwd=src, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            log.write(line)
            log.flush()
            print(line, end="", flush=True)
        if proc.wait() != 0:
            raise SystemExit(f"LineFormer worker exited {proc.returncode}; see {LOG_PATH}")


def _load_predictions(path: pathlib.Path) -> list[LineFormerPrediction]:
    with path.open() as f:
        return [LineFormerPrediction.from_record(json.loads(line)) for line in f if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--lineformer-home", type=pathlib.Path, default=DEFAULT_LF_HOME)
    args = parser.parse_args()

    items, n_real = build_dataset()
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = PREDICTIONS_DIR / f"pretrained-n{n_real}.jsonl"

    if not args.score_only:
        _run_inference(items, predictions_path, args.lineformer_home)

    predictions = _load_predictions(predictions_path)
    missing = {i.figure_id for i in items} - {p.figure_id for p in predictions}
    if missing:
        raise SystemExit(f"{len(missing)} figure(s) have no prediction: {sorted(missing)[:5]}")
    n_errors = sum(p.error is not None for p in predictions)
    if n_errors == len(predictions):
        raise SystemExit("every figure errored -- refusing to write a 0.0 result")

    payload = run(
        "lineformer-pretrained",
        "LineFormer (pretrained, ICDAR2023)",
        PrecomputedLineFormerModelRunner(predictions),
    )
    payload["run_environment"] = "local GPU (no Colab) -- see design doc for versions"
    payload["raw_predictions"] = str(predictions_path.relative_to(REPO_ROOT))
    payload["n_worker_errors"] = n_errors
    payload["calibration_note"] = (
        "Pixel->data mapping takes the full image frame as the plot area, the "
        "same choice as the n42 Colab run (notebooks/lineformer_colab.ipynb); "
        "raw pixel output is kept so other mappings can be rescored without "
        "re-running inference."
    )
    out_path = RESULTS_DIR / f"lineformer-pretrained-n{n_real}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        f"wrote {out_path}: mean_summary_score={payload['mean_summary_score']:.4f} "
        f"over {payload['n_figures']} real figures ({n_errors} worker errors)",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
