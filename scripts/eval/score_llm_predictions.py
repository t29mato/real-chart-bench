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
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
SCRATCH = pathlib.Path(
    "/tmp/claude-1000/-home-mato-repos-real-chart-bench/"
    "628beb76-383b-42e9-bf21-8d4188daf8dc/scratchpad"
)

# Two conditions over the same ten figures, the same ground truth and the same
# metric. The only difference is whether the model was handed the axis extent
# -- which is the part that decides whether these scores mean "chart reading is
# solved" or "marker locating is solved given a perfect calibration".
def _resolve_pred(archive: pathlib.Path, work: pathlib.Path, model: str) -> pathlib.Path | None:
    """Archived copy in the repository first, live scratch run second.

    The scratch directories live under /tmp and a session restart wiped the
    first noaxis run outright. Predictions are therefore archived into the
    repository as each model finishes, and scoring reads the archive by
    preference so a published number can always be re-derived from a clone.
    """
    for candidate in (archive / f"{model}.json", work / model / "predictions.json"):
        if candidate.exists():
            return candidate
    return None


CONDITIONS = {
    "calibrated": {
        "work": SCRATCH / "llm_eval",
        "meta": REPO / "data/llm_subset_n10",
        "archive": REPO / "data/llm_subset_n10/predictions",
        "suffix": "",
        "label": "軸レンジを与えた条件",
    },
    "noaxis": {
        "work": SCRATCH / "llm_eval_noaxis",
        # The noaxis task list deliberately carries no ranges, so the ranges
        # needed to build ExtractionTask come from the calibrated export. The
        # model never saw them; they are used only to score.
        "meta": REPO / "data/llm_subset_n10",
        "archive": REPO / "data/llm_subset_n10_noaxis/predictions",
        "suffix": "-noaxis",
        "label": "軸レンジを与えない条件（モデルが目盛を自分で読む）",
    },
    # The remaining 101 of the 111 scoreable figures (owner request,
    # 2026-09-12: "データ抽出してない残りのグラフ全部"). Axis ranges given,
    # i.e. the same condition every CV baseline gets. tasks/_key live in the
    # repository here, not in scratch, because the first noaxis run was lost
    # to a /tmp wipe.
    "rest": {
        "work": SCRATCH / "llm_eval_rest",
        "meta": REPO / "data/llm_subset_rest",
        "archive": REPO / "data/llm_subset_rest/predictions",
        "suffix": "-rest",
        "label": "軸レンジを与えた条件（n=10サブセット以外の残り全図）",
    },
}

MODELS = {
    "claude-opus-5": "Claude Opus 5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "claude-fable-5": "Claude Fable 5",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}

# How hard each model actually worked, from its agent run. This belongs in the
# results because the four runs are not comparable as "model reads chart": each
# model was free to use tools, and they used wildly different amounts. Fable
# wrote a colour-segmentation pipeline; Opus read magnified crops across 94 tool
# calls; Sonnet and Haiku answered from the images in about 15. A reader
# comparing the scores without this would conclude something about eyesight
# that the numbers do not support.
EFFORT = {
    "claude-opus-5": {"tool_uses": 94, "tokens": 170675, "seconds": 1399,
                      "method": "拡大クロップを多数作って目視で読み取り。軸は枠端ではなく"
                                "印刷目盛で校正したと明記。fig_09では凡例の参照線2本も系列として報告。"},
    "claude-sonnet-5": {"tool_uses": 15, "tokens": 57590, "seconds": 186,
                        "method": "画像を直接読んで回答。補助スクリプトなし。"},
    "claude-fable-5": {"tool_uses": 58, "tokens": 153838, "seconds": 925,
                       "method": "自前のCVパイプラインを記述 — 色分割したブロブ重心から"
                                 "ガイド線を除去し、検出した目盛で校正。重ね描きで検証。"},
    "claude-haiku-4-5": {"tool_uses": 14, "tokens": 51153, "seconds": 180,
                         "method": "画像を直接読んで回答。補助スクリプトなし。"},
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


def parse_curves(raw, x_scale: ScaleType) -> list[Curve]:
    """Model answers as Curves, built the way an extractor adapter builds them.

    `Curve` carries x_scale and no y_scale, and the CV adapters set x_scale
    from the task -- see adapter/naive_cv_extractor.py. Matching that exactly
    matters: the metric is what compares these against the ground truth, and a
    difference here would show up as a score difference that has nothing to do
    with how well the model read the chart.
    """
    out = []
    for c in raw:
        xs, ys = c.get("x") or [], c.get("y") or []
        pairs = [(float(x), float(y)) for x, y in zip(xs, ys, strict=False)
                 if isinstance(x, int | float) and isinstance(y, int | float)]
        if len(pairs) < 2:
            continue
        out.append(Curve(x_values=tuple(p[0] for p in pairs),
                         y_values=tuple(p[1] for p in pairs),
                         x_scale=x_scale))
    return out


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "calibrated"
    if name not in CONDITIONS:
        raise SystemExit(f"条件は {list(CONDITIONS)} のいずれか (指定: {name})")
    cond = CONDITIONS[name]
    work = cond["work"]

    meta = cond["meta"]
    tasks = json.loads((meta / "tasks.json").read_text())
    key = json.loads((meta / "_key.json").read_text())
    gt_all = json.loads((REPO / "data/verified_pairs/ground_truth.json").read_text())
    reg = {p.figure_id: p for p in load_registry(REPO / "data/verified_pairs/registry.json")}
    # Derived, never hardcoded: the version string went stale at n112 once
    # 36342/34990 left the scoreable set (design 7.27's lesson).
    reg_scoreable = select_verified_pairings(
        load_registry(REPO / "data/verified_pairs/registry.json")
    )

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
            # Built exactly as run_baselines.py's _ground_truth_for does --
            # no scale arguments, series_label from prop_y -- so these figures
            # are scored against the same ground truth the CV baselines face.
            ground_truth=[Curve(x_values=tuple(c["x"]), y_values=tuple(c["y"]),
                                series_label=c.get("prop_y")) for c in gt],
        ))
        order.append(t["id"])

    matcher = HungarianCurveMatcher(metric=NormalizedYDistanceMetric())
    written = []
    for model_id, model_name in MODELS.items():
        path = _resolve_pred(cond["archive"], work, model_id)
        if path is None:
            print(f"  {model_id}: 予測ファイルがない → スキップ")
            continue
        raw = json.loads(path.read_text())
        preds = {}
        for t in tasks:
            if t["id"] in raw:
                preds[t["id"]] = parse_curves(raw[t["id"]], ScaleType(t["x_scale"]))
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
            "model_id": model_id + cond["suffix"],
            "model_name": model_name + ("（軸レンジなし）" if cond["suffix"] else ""),
            "dataset_version": (
                f"v0-eval-pilot-n{len(reg_scoreable)}-llm-subset-"
                f"n{len(items)}{cond['suffix']}"),
            "run_at": datetime.now(UTC).isoformat(),
            "n_figures": len(per_figure),
            "mean_summary_score": sum(p["summary_score"] for p in per_figure) / len(per_figure),
            "per_figure": per_figure,
            "agent_effort": EFFORT.get(model_id) if not cond["suffix"] else None,
            "condition": cond["label"],
            "notes": (
                "Claude Code のサブエージェントとして各モデルを起動し、"
                "図の画像と軸レンジ(ExtractionTask と同じ情報)だけを与えて抽出させた結果。"
                "採点は他のベースラインと同一(Hungarian マッチャ + 正規化Y距離)。"
                "汚染対策: 画像はリポジトリ外へ fig_NN.png という名前で複製し、"
                "正解データはプロンプトに含めず、リポジトリを探索しないよう指示した。"
                "ただしこれは緩和策でありサンドボックスではない — "
                "Bash を持つエージェントが探しにいけばリポジトリに到達しうる。"
                "実際にコピーが起きていないことは事後に確認した: モデル間で完全一致する"
                "(x,y)点は0〜1.4%で、独立に読み取った場合の水準である。"
                "重要な限界として、4モデルは同じ作業量では走っていない — agent_effort 参照。"
                "Fable は自前のCVパイプラインを書き、Opus は拡大クロップを多数作って読んだ一方、"
                "Sonnet と Haiku はツール実行15回前後で回答している。"
                "したがってこれは『どのモデルが図をよく読めるか』ではなく"
                "『ツールと時間を与えられたエージェントとしてどれだけ抽出できるか』の比較である。"
                "10図のみの試行であり、順位を確定させるには小さすぎる。"
                "112図の本評価とは別の dataset_version を持つ。"
            ),
        }
        out = RESULTS / f"{model_id}-v0{cond['suffix']}.json"
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
