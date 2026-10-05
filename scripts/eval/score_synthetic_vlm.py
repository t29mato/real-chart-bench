"""Scores local-VLM runs on PlotQA dot_line 100 (design §7.75 (B)) with the
same point metric as the real figures.

Reads data/synthetic/plotqa_dot_line/local_vlm_cuda/<model>/<condition>.jsonl
(worker_cuda.py with RCB_RUN_DIR=data/synthetic/plotqa_dot_line/run), turns
each answer into curves exactly as the real-figure scorer does
(score_llm_predictions.parse_curves), and writes
results/<model>-plotqa-dot-line[-noaxis].json with dataset_version
synthetic-plotqa-dot-line-n100[-noaxis] -- the noaxis rows rank with the
chart-to-table models, which also read the axes themselves.

Claude agents (design 7.80) answered the noaxis tasks in sealed batches;
their archived answers are data/synthetic/plotqa_dot_line/llm_run/noaxis/
<model>/part{1,2}.predictions.json (archive_llm_run_synthetic.py) and are
scored the same way, from the same tasks and key.

Usage: python scripts/eval/score_synthetic_vlm.py <model> [<model> ...]
       (a claude-* model is read from the agent archive)
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import UTC, datetime

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eval"))

from run_baselines import METRIC_LABEL  # noqa: E402
from score_llm_predictions import parse_curves  # noqa: E402
from synthetic_plotqa import RUN, load_items  # noqa: E402

from real_chart_bench.adapter.agent_run_archive import load_agent_run_parts  # noqa: E402
from real_chart_bench.adapter.lineformer_model_runner import image_key  # noqa: E402
from real_chart_bench.usecase.evaluate_dataset import (  # noqa: E402
    PRIMARY_POINT_TAU,
    evaluate_model_on_dataset,
    matcher_for_task,
)
from real_chart_bench.usecase.result_payload import (  # noqa: E402
    POINT_METRIC_LABEL,
    aggregate_dense_marker_metrics,
    aggregate_point_metrics,
    figure_result_row,
)

ARCHIVE = REPO / "data/synthetic/plotqa_dot_line/local_vlm_cuda"
AGENT_ARCHIVE = REPO / "data/synthetic/plotqa_dot_line/llm_run"
NAMES = {
    "qwen3.5-9b-bf16": "Qwen3.5-9B (bf16)",
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-fable-5-1": "Claude Fable 5.1",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}


class _Replay:
    def __init__(self, curves: dict[str, list]):
        self._curves = curves

    def extract(self, task):
        k = image_key(task.image_bytes)
        if k not in self._curves:
            raise RuntimeError("no usable answer for this figure")
        return self._curves[k]


def score(model: str, condition: str) -> dict:
    key = json.loads((RUN / "_key.json").read_text())
    items = load_items()
    by_fid = {i.figure_id: i for i in items}
    agent = model.startswith("claude-")
    if agent:
        answers = load_agent_run_parts(AGENT_ARCHIVE / condition / model, ["part1", "part2"])
    else:
        answers = {
            rec["fig"]: rec["parsed"]
            for rec in (
                json.loads(line)
                for line in (ARCHIVE / model / f"{condition}.jsonl").read_text().splitlines()
                if line.strip()
            )
            if not rec.get("error") and rec.get("parsed")
        }
    curves = {}
    for fig, answer in answers.items():
        if not answer:
            continue
        item = by_fid[f"plotqa-{key[fig]['figure_id']}"]
        curves[image_key(item.task.image_bytes)] = parse_curves(answer, item.task.x_scale)
    results = evaluate_model_on_dataset(_Replay(curves), items, matcher_for=matcher_for_task)
    per_figure = [figure_result_row(r) for r in results]
    suffix = "-noaxis" if condition == "noaxis" else ""
    if agent:
        run = {
            "raw_output": str((AGENT_ARCHIVE / condition / model).relative_to(REPO)),
            "prompt": "scripts/eval/llm_run_synthetic_prompt.md",
            "n_without_answer": len(items) - len(curves),
        }
        where = "Claude Code エージェント"
    else:
        env = json.loads((ARCHIVE / model / "env.json").read_text())
        run = {
            "hardware": f"{env['gpu']} (24GB), Linux",
            "engine": f"vLLM {env['vllm']}",
            "precision": env["precision"],
            "raw_output": str((ARCHIVE / model / f"{condition}.jsonl").relative_to(REPO)),
            "n_without_answer": len(items) - len(curves),
        }
        where = "ローカル RTX 4090"
    return {
        "model_id": f"{model}-plotqa-dot-line{suffix}",
        "model_name": f"{NAMES.get(model, model)}（"
        + ("軸レンジなし、" if suffix else "")
        + f"{where}、v3 プロンプト）",
        "execution": "cloud" if agent else "local",
        "dataset_version": f"synthetic-plotqa-dot-line-n{len(items)}{suffix}",
        "run_at": datetime.now(UTC).isoformat(),
        "n_figures": len(per_figure),
        "metric": METRIC_LABEL,
        "point_metric": POINT_METRIC_LABEL,
        "mean_summary_score": sum(p["summary_score"] for p in per_figure) / len(per_figure),
        "point_metrics": aggregate_point_metrics(per_figure, PRIMARY_POINT_TAU),
        "dense_marker_metrics": aggregate_dense_marker_metrics(per_figure),
        "per_figure": per_figure,
        ("agent_run" if agent else "local_run"): run,
        "condition": (
            "PlotQA dot_line 100(合成図、CC-BY-4.0)。"
            "プロンプトは実論文の図の v3 単発版と同じ。"
        ),
    }


def main() -> None:
    for model in sys.argv[1:]:
        for condition in ("calibrated", "noaxis"):
            src = (
                AGENT_ARCHIVE / condition / model
                if model.startswith("claude-")
                else ARCHIVE / model / f"{condition}.jsonl"
            )
            if not src.exists():
                continue
            payload = score(model, condition)
            out = REPO / "results" / f"{payload['model_id']}.json"
            out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
            m = payload["point_metrics"]["by_tau"][str(PRIMARY_POINT_TAU)]["macro"]
            run = payload.get("local_run") or payload["agent_run"]
            print(
                f"{payload['model_id']:<40} point F1 {m['point_f1']:.3f}  "
                f"no answer {run['n_without_answer']}"
            )


if __name__ == "__main__":
    main()
