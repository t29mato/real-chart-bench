"""Scores the chart-to-table models (design §7.68, §7.75 (1)) from their saved
raw text: data/chart2table_predictions/<model>.jsonl -> results/<model>-v0-noaxis.json.

These models take an image and nothing else, so they are rows of the
axis-withheld table (`<base>-noaxis`), next to the LLMs' noaxis rows. The LLMs
were told, per axis, which numbers to report (as printed / 10^tick for
log10-labelled axes / kelvin for a degC axis; data/llm_run_v3/noaxis/tasks.json).
These models cannot be told, so the same rule is applied here, after parsing,
deterministically -- and counted in the result file.

Parsing is adapter/chart_table_parser.py: literal rules, no interpolation. An
unparseable table scores as no answer and is counted (`n_unparseable`).

Usage: python scripts/eval/score_chart2table.py [model ...]   (default: every
       data/chart2table_predictions/*.jsonl)
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import UTC, datetime

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts/eval"))

from run_baselines import METRIC_LABEL, build_dataset  # noqa: E402

from real_chart_bench.adapter.chart_table_parser import parse_chart_table  # noqa: E402
from real_chart_bench.adapter.lineformer_model_runner import image_key  # noqa: E402
from real_chart_bench.domain.curve import Curve  # noqa: E402
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

PRED_DIR = REPO / "data/chart2table_predictions"
RESULTS = REPO / "results"

MODELS = {
    "deplot": {"name": "DePlot", "license": "Apache-2.0 (code and weights)"},
    "unichart": {
        "name": "UniChart (base-960)",
        "license": "code MIT, weights GPL-3.0",
        "row_separator": " & ",
    },
    "chartgemma": {
        "name": "ChartGemma",
        "license": "code GPL-3.0, weights MIT (base PaliGemma under the Gemma terms)",
    },
    "granite-vision": {"name": "Granite Vision 4.1 4B", "license": "Apache-2.0"},
    "tinychart": {
        "name": "TinyChart-3B-768",
        "license": (
            "code Apache-2.0; weights carry no license statement (evaluated, not redistributed)"
        ),
    },
}


def _report_rules() -> dict[str, tuple[bool, bool]]:
    """figure_id -> (x is degC printed / K stored, y ticks are log10 values)."""
    run = REPO / "data/llm_run_v3"
    key = json.loads((run / "_key.json").read_text())
    rules = {}
    for t in json.loads((run / "noaxis/tasks.json").read_text()):
        k = key[t["id"]]
        rules[f"{k['paper_id']}-{k['figure_id']}"] = (
            "Celsius" in t["x_report"],
            "base-10 logarithms" in t["y_report"],
        )
    return rules


def _pow10(y: float) -> float | None:
    """10^y, or None when y cannot be a log10 tick value (overflow)."""
    try:
        return 10.0**y
    except OverflowError:
        return None


class _Replay:
    def __init__(self, curves_by_key: dict[str, list[Curve]], errors: dict[str, str]):
        self._curves, self._errors = curves_by_key, errors

    def extract(self, task):
        k = image_key(task.image_bytes)
        if k in self._errors:
            raise RuntimeError(self._errors[k])
        return self._curves.get(k, [])


def score(model: str, dataset: str = "real") -> dict:
    """dataset "real": the scored real figures; "plotqa": PlotQA dot_line 100
    (design 7.75 (B); every axis is printed as-is, so no reporting rule)."""
    spec = MODELS[model]
    pred_dir = PRED_DIR if dataset == "real" else PRED_DIR / f"synthetic-{dataset}"
    records = [
        json.loads(line)
        for line in (pred_dir / f"{model}.jsonl").read_text().splitlines()
        if line.strip()
    ]
    if dataset == "plotqa":
        from synthetic_plotqa import load_items

        items = load_items()
    else:
        items, _ = build_dataset()
        items = [i for i in items if not i.figure_id.startswith("synthetic-")]
    by_fid = {r["figure_id"]: r for r in records}
    missing = [i.figure_id for i in items if i.figure_id not in by_fid]
    if missing:
        raise SystemExit(f"{model}: no record for {len(missing)} figure(s), e.g. {missing[:3]}")
    rules = _report_rules() if dataset == "real" else {}

    curves, errors = {}, {}
    n_unparseable = n_rows_dropped = n_x_rule = n_y_rule = n_overflow = 0
    for item in items:
        rec = by_fid[item.figure_id]
        k = image_key(item.task.image_bytes)
        if rec["error"]:
            errors[k] = rec["error"].strip().splitlines()[-1]
            continue
        table = parse_chart_table(rec["raw_text"], spec.get("row_separator"))
        n_rows_dropped += table.n_rows_dropped
        if not table.parsed:
            n_unparseable += 1
            curves[k] = []
            continue
        x_c, y_log = rules.get(item.figure_id, (False, False))
        n_x_rule += x_c
        n_y_rule += y_log
        out = []
        for s in table.series:
            pts = [
                (x + 273.15 if x_c else x, _pow10(y) if y_log else y)
                for x, y in zip(s.x, s.y, strict=True)
            ]
            n_overflow += sum(p[1] is None for p in pts)
            pts = [p for p in pts if p[1] is not None]
            if item.task.x_scale.value == "log":
                pts = [p for p in pts if p[0] > 0]
            if pts:
                out.append(
                    Curve(
                        x_values=tuple(p[0] for p in pts),
                        y_values=tuple(p[1] for p in pts),
                        series_label=s.label,
                        x_scale=item.task.x_scale,
                    )
                )
        curves[k] = out

    results = evaluate_model_on_dataset(
        _Replay(curves, errors), items, matcher_for=matcher_for_task
    )
    per_figure = [figure_result_row(r) for r in results]
    base = json.loads((RESULTS / "naive-cv-v0.json").read_text())["dataset_version"]
    first = records[0]
    synthetic = dataset != "real"
    return {
        "model_id": f"{model}-plotqa-dot-line" if synthetic else f"{model}-noaxis",
        "model_name": f"{spec['name']}（軸レンジなし、合成図で学習、ローカル）",
        "execution": "local",
        "dataset_version": (
            f"synthetic-plotqa-dot-line-n{len(items)}-noaxis" if synthetic else f"{base}-noaxis"
        ),
        "run_at": datetime.now(UTC).isoformat(),
        "n_figures": len(per_figure),
        "metric": METRIC_LABEL,
        "point_metric": POINT_METRIC_LABEL,
        "mean_summary_score": sum(p["summary_score"] for p in per_figure) / len(per_figure),
        "point_metrics": aggregate_point_metrics(per_figure, PRIMARY_POINT_TAU),
        "dense_marker_metrics": aggregate_dense_marker_metrics(per_figure),
        "per_figure": per_figure,
        "chart2table": {
            "repo": first["repo"],
            "prompt": first["prompt"],
            "prompt_source": first["prompt_source"],
            "generate": first["generate"],
            "trust_remote_code": first.get("trust_remote_code", False),
            "transformers": first["transformers"],
            "torch": first["torch"],
            "license": spec["license"],
            "raw_predictions": str((pred_dir / f"{model}.jsonl").relative_to(REPO)),
            "n_worker_errors": len(errors),
            "n_unparseable": n_unparseable,
            "n_rows_dropped_numeric": n_rows_dropped,
            "report_rule_applied": {"x_celsius_to_kelvin": n_x_rule, "y_10_pow": n_y_rule},
            # points on a log10-labelled axis whose value cannot be a log10
            # tick (10^y overflows): dropped, not scored
            "n_points_dropped_overflow": n_overflow,
            "seconds_per_figure_mean": round(
                sum(r.get("seconds", 0) for r in records) / len(records), 2
            ),
        },
        "condition": (
            "軸レンジなし。画像だけを入力し、モデル自身の表抽出をそのまま使う。"
            "LLM に渡した軸ごとの報告規則(log10 目盛は 10^値、°C 印字は K)は、"
            "採点時に同じ規則を決定論的に適用した(design 7.75)。"
        ),
    }


def main() -> None:
    args = sys.argv[1:]
    dataset = "real"
    if args and args[0].startswith("--dataset="):
        dataset = args.pop(0).split("=", 1)[1]
    pred_dir = PRED_DIR if dataset == "real" else PRED_DIR / f"synthetic-{dataset}"
    models = args or sorted(p.stem for p in pred_dir.glob("*.jsonl"))
    for model in models:
        payload = score(model, dataset)
        out = RESULTS / f"{payload['model_id']}.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
        pm = payload["point_metrics"]["by_tau"][str(PRIMARY_POINT_TAU)]["macro"]
        c = payload["chart2table"]
        print(
            f"{model:<15} point F1 {pm['point_f1']:.3f} (R {pm['point_recall']:.3f} "
            f"P {pm['point_precision']:.3f})  summary {payload['mean_summary_score']:.3f}  "
            f"unparseable {c['n_unparseable']}/{payload['n_figures']}  "
            f"errors {c['n_worker_errors']}"
        )


if __name__ == "__main__":
    main()
