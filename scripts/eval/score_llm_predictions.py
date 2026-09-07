"""Score LLM predictions on the 10-figure subset through the existing harness.

The models are reached by spawning them as subagents rather than over the API
(the owner's standing instruction: 「現在のClaudeCodeをそのまま使ってください。
APIはふよう」). An agent cannot be plugged into `ModelRunnerPort` directly, so
it writes its answers to a file and this replays them -- the same
`evaluate_model_on_dataset`, the same Hungarian matcher and normalised-y
metric, the same payload shape as every CV baseline. Nothing about the scoring
is special-cased for LLMs; only the transport differs.

A missing or malformed prediction is scored as a total miss rather than
skipped, exactly as `evaluate_model_on_dataset` treats a crashing extractor.
Dropping unanswered figures would flatter a model that declined the hard ones.

The subset gets its own `dataset_version` so no row is ever compared against a
run on a different figure set -- the same rule that keeps the LineFormer
comparable subset honest (design §7.36).
"""

from __future__ import annotations

import json
import pathlib
import sys
from datetime import UTC, datetime

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.domain.curve import Curve, ScaleType  # noqa: E402
from real_chart_bench.domain.matching import HungarianCurveMatcher  # noqa: E402
from real_chart_bench.domain.metrics import NormalizedYDistanceMetric  # noqa: E402
from real_chart_bench.usecase.evaluate_dataset import (  # noqa: E402
    DatasetItem,
    evaluate_model_on_dataset,
)
from real_chart_bench.usecase.model_runner import ExtractionTask  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
WORK = pathlib.Path(
    "/tmp/claude-1000/-home-mato-repos-real-chart-bench/"
    "628beb76-383b-42e9-bf21-8d4188daf8dc/scratchpad/llm_eval"
)

MODELS = {
    "claude-opus-5": "Claude Opus 5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-fable-5": "Claude Fable 5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}


class ReplayRunner:
    """Serves one model's recorded answers as a ModelRunnerPort."""

    def __init__(self, preds: dict[str, list[Curve]], order: list[str]):
        self._preds = preds
        self._order = order
        self._i = 0

    def extract(self, task: ExtractionTask) -> list[Curve]:
        name = self._order[self._i]
        self._i += 1
        if name not in self._preds:
            raise ValueError(f"{name}: モデルが回答を返していない")
        return self._preds[name]


def parse_curves(raw, x_scale: ScaleType, y_scale: ScaleType) -> list[Curve]:
    out = []
    for c in raw:
        xs, ys = c.get("x") or [], c.get("y") or []
        pairs = [(float(x), float(y)) for x, y in zip(xs, ys, strict=False)
                 if isinstance(x, int | float) and isinstance(y, int | float)]
        if len(pairs) < 2:
            continue
        out.append(Curve(x_values=tuple(p[0] for p in pairs),
                         y_values=tuple(p[1] for p in pairs),
                         x_scale=x_scale, y_scale=y_scale))
    return out


def main() -> None:
    tasks = json.loads((WORK / "tasks.json").read_text())
    key = json.loads((WORK / "_key.json").read_text())
    gt_all = json.loads((REPO / "data/verified_pairs/ground_truth.json").read_text())
    reg = {p.figure_id: p for p in load_registry(REPO / "data/verified_pairs/registry.json")}

    items, order = [], []
    for t in tasks:
        k = key[t["id"]]
        p = reg[k["figure_id"]]
        gt = [c for c in gt_all[k["figure_id"]] if c.get("x")]
        items.append(DatasetItem(
            figure_id=f"{k['paper_id']}-{k['figure_id']}",
            task=ExtractionTask(
                image_bytes=(REPO / p.image_path).read_bytes(),
                x_range=tuple(t["x_range"]), y_range=tuple(t["y_range"]),
                x_scale=ScaleType(t["x_scale"]), y_scale=ScaleType(t["y_scale"]),
            ),
            ground_truth=[Curve(x_values=tuple(c["x"]), y_values=tuple(c["y"]),
                                x_scale=ScaleType(t["x_scale"]),
                                y_scale=ScaleType(t["y_scale"])) for c in gt],
        ))
        order.append(t["id"])

    matcher = HungarianCurveMatcher(metric=NormalizedYDistanceMetric())
    written = []
    for model_id, model_name in MODELS.items():
        path = WORK / model_id / "predictions.json"
        if not path.exists():
            print(f"  {model_id}: 予測ファイルがない → スキップ")
            continue
        raw = json.loads(path.read_text())
        preds = {}
        for t in tasks:
            if t["id"] in raw:
                preds[t["id"]] = parse_curves(
                    raw[t["id"]], ScaleType(t["x_scale"]), ScaleType(t["y_scale"]))
        results = evaluate_model_on_dataset(ReplayRunner(preds, order), items, matcher=matcher)
        per_figure = [{
            "figure_id": r.figure_id,
            "summary_score": r.evaluation.summary_score,
            "match_rate": r.evaluation.match_rate,
            "mean_curve_distance": r.evaluation.mean_curve_distance,
            "mean_coverage_ratio": r.evaluation.mean_coverage_ratio,
            "error": r.error,
        } for r in results]
        payload = {
            "model_id": model_id,
            "model_name": model_name,
            "dataset_version": f"v0-eval-pilot-n112-llm-subset-n{len(items)}",
            "run_at": datetime.now(UTC).isoformat(),
            "n_figures": len(per_figure),
            "mean_summary_score": sum(p["summary_score"] for p in per_figure) / len(per_figure),
            "per_figure": per_figure,
            "notes": (
                "Claude Code のサブエージェントとして各モデルを起動し、"
                "図の画像と軸レンジ(ExtractionTask と同じ情報)だけを与えて抽出させた結果。"
                "採点は他のベースラインと同一(Hungarian マッチャ + 正規化Y距離)。"
                "汚染対策: 画像はリポジトリ外へ fig_NN.png という名前で複製し、"
                "正解データはプロンプトに含めず、リポジトリを探索しないよう指示した。"
                "ただしこれは緩和策でありサンドボックスではない — "
                "Bash を持つエージェントが探しにいけばリポジトリに到達しうる。"
                "10図のみの試行であり、112図の本評価とは別の dataset_version を持つ。"
            ),
        }
        out = RESULTS / f"{model_id}-v0.json"
        out.write_text(json.dumps(payload, indent=2) + "\n")
        written.append((model_id, payload["mean_summary_score"], len(preds)))
        print(f"  {model_id:<20} 平均 {payload['mean_summary_score']:.4f}  "
              f"（回答 {len(preds)}/{len(items)} 図） → {out.name}")

    if written:
        print("\n順位:")
        for i, (m, s, n) in enumerate(sorted(written, key=lambda w: -w[1]), 1):
            print(f"  {i}. {MODELS[m]:<20} {s:.4f}")


if __name__ == "__main__":
    main()
