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

from real_chart_bench.adapter.ground_truth_store import (  # noqa: E402
    ground_truth_revision,
    load_ground_truth,
)
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.domain.curve import Curve, ScaleType  # noqa: E402
from real_chart_bench.usecase.evaluate_dataset import (  # noqa: E402
    DatasetItem,
    evaluate_model_on_dataset,
    matcher_for_task,
)
from real_chart_bench.usecase.model_runner import ExtractionTask  # noqa: E402
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
GROUND_TRUTH_PATH = REPO / "data/verified_pairs/ground_truth.json"
GROUND_TRUTH_SUPPLEMENT_DIR = REPO / "data/verified_pairs/ground_truth_supplement"
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
        "name_suffix": "",
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
        "name_suffix": "（軸レンジなし）",
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
        "name_suffix": "（n=10以外の残り）",
        "label": "軸レンジを与えた条件（n=10サブセット以外の残り全図）",
    },
    # Every scoreable figure, axis ranges given: the n=10 run and the rest
    # run scored together in one pass. This replaces the earlier
    # -full files, which were assembled by hand from the two runs' published
    # per-figure scores and so could not follow a ground-truth change.
    "full": {
        "parts": ["calibrated", "rest"],
        "suffix": "-full",
        "name_suffix": "（採点対象全図）",
        "label": "軸レンジを与えた条件（採点対象全図: n=10 の実行と残りの実行を合わせて採点）",
    },
}

# A figure whose stored unit space changed *after* a model answered it. The
# archived answer is in the space the task showed at the time, so it has to be
# converted before it can be compared against ground truth that has since
# moved. This is arithmetic applied at scoring time -- the raw file under
# data/llm_subset_*/predictions/ is left exactly as the model wrote it, so the
# record of what was actually answered stays intact.
PREDICTION_RESCALE = {
    # 5904/13761: y migrated from SI (V/K) to the paper's printed units (µV/K)
    # on 2026-09-30, after the n=10 run. The models were shown y_range
    # [0.0, 2e-06] and answered in that space.
    "13761": {"y": 1e6},
    # The 2026-09-30 batch: 19 further axes whose stored range sat in a
    # different unit space than the printed one (V/K vs µV/K, W/(m*K^2) vs the
    # printed 1e-4 unit, 1/T vs 1000/T). Migrating the registry and ground
    # truth moved them after the models had already answered in the old space.
    "13164": {"y": 1e+06},
    "15114": {"y": 1e+06},
    "20121": {"y": 1e+06},
    "20739": {"y": 1e+06},
    "21283": {"y": 1e+06},
    "29154": {"y": 10000},
    "33296": {"y": 10000},
    "40587": {"x": 1000},
    "45323": {"x": 1000},
    "45356": {"x": 1000, "y": 0.01},
    "45360": {"x": 1000, "y": 0.01},
    "48871": {"x": 1000, "y": 0.01},
    "51437": {"x": 1000, "y": 0.01},
    "51438": {"x": 1000, "y": 0.01},
    "51439": {"x": 1000, "y": 0.01},
    "51440": {"x": 1000, "y": 0.01},
    "51441": {"x": 1000, "y": 0.01},
    "51442": {"x": 1000, "y": 0.01},
    "51688": {"x": 1000, "y": 0.01},
    # Two log axes stored in S/m against a figure printing S/cm -- log10 of the
    # stored range sat exactly +2.00 decades above the printed labels at both
    # endpoints, so a unit factor rather than a framing margin.
    "45818": {"y": 0.01},
    # 2026-10-02: 28331/28500 still stored sigma as 50000-70000 S/m against an
    # axis printed 5.0-7.0 under "sigma x10^4 (S/m)" -- missed by the 09-30
    # migration (its sibling 28492 is stored as printed). Moved to printed
    # units; answers given against the old range are scaled to match.
    "28500": {"y": 1e-4},
}
# The v2 calibrated condition was handed 28500's old 50000-70000 range too.
# Its noaxis condition was not (the model read the printed 5-7 itself).
PREDICTION_RESCALE_V2_CALIBRATED = {"28500": {"y": 1e-4}}

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


# The 2026-10-01 run (scripts/eval/prepare_llm_run_v2.py): every scoreable
# figure, both conditions, the models a subagent can be launched as today.
V2_ARCHIVE = REPO / "data/llm_run_v2"
MODELS_V2 = {
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-fable-5-1": "Claude Fable 5.1",
    "claude-haiku-4-5": "Claude Haiku 4.5",
}
CONDITIONS["v2-calibrated"] = {
    "v2": "calibrated",
    "models": MODELS_V2,
    # "-r2": Haiku 4.5 is in both runs, so ids must not collide with the
    # 2026-09 run's files
    "suffix": "-r2",
    "name_suffix": "（2026-10-01）",
    "label": "軸レンジを与えた条件（2026-10-01 実行、採点対象全図）",
}
CONDITIONS["v2-noaxis"] = {
    "v2": "noaxis",
    "models": MODELS_V2,
    "suffix": "-r2-noaxis",
    "name_suffix": "（軸レンジなし、2026-10-01）",
    "label": "軸レンジを与えない条件（2026-10-01 実行、採点対象全図）",
}
V2_NOTES = (
    "Claude Code のサブエージェントとして起動(2026-10-01、プロンプトは "
    "scripts/eval/llm_run_v2_prompt.md に保存)。1モデル・1条件あたり2バッチ(51図/50図)に分け、"
    "各バッチに独立したディレクトリ(リポジトリ外、fig_NNN.png の乱順の名前)を与えた。"
    "calibrated は ExtractionTask と同じ軸レンジ・スケールを、noaxis は軸ごとの報告規則"
    "(印字どおり / log10 目盛は 10^値 / °C 軸は K)だけを与えた。手法は各エージェントの自由で、"
    "スコアは『モデルの視力』ではなく『ツールを使えるエージェントとしての抽出性能』である。"
    "生の回答は data/llm_run_v2/ に保存。"
)


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
    # A condition is one run, or (full) several runs scored together. Task
    # ids repeat across runs (fig_001.png ...), so they are namespaced by run.
    parts = cond.get("parts", [name])
    tasks, key = [], {}
    if cond.get("v2"):
        v2_key = json.loads((V2_ARCHIVE / "_key.json").read_text())
        tasks = [
            {**t, "id": f"{name}:{t['id']}"}
            for t in json.loads((V2_ARCHIVE / cond["v2"] / "tasks.json").read_text())
        ]
        key = {f"{name}:{k}": v for k, v in v2_key.items()}
    for part in [] if cond.get("v2") else parts:
        meta = CONDITIONS[part]["meta"]
        tasks += [
            {**t, "id": f"{part}:{t['id']}"}
            for t in json.loads((meta / "tasks.json").read_text())
        ]
        key |= {
            f"{part}:{k}": v for k, v in json.loads((meta / "_key.json").read_text()).items()
        }
    gt_all = load_ground_truth(GROUND_TRUTH_PATH, GROUND_TRUTH_SUPPLEMENT_DIR)
    reg = {p.figure_id: p for p in load_registry(REPO / "data/verified_pairs/registry.json")}
    # Derived, never hardcoded: the version string went stale at n112 once
    # 36342/34990 left the scoreable set (design 7.27's lesson).
    reg_scoreable = select_verified_pairings(
        load_registry(REPO / "data/verified_pairs/registry.json")
    )

    # A figure the owner has since excluded must leave the score, even though
    # the model answered it and the answer stays archived. Scoring an excluded
    # figure would keep measuring something the registry says is unmeasurable.
    scoreable_ids = {p.figure_id for p in reg_scoreable}
    dropped = [t["id"] for t in tasks if key[t["id"]]["figure_id"] not in scoreable_ids]
    if dropped:
        print(f"  採点対象外になった図を除外: {len(dropped)}件 {dropped}")
    tasks = [t for t in tasks if key[t["id"]]["figure_id"] in scoreable_ids]

    items, order = [], []
    for t in tasks:
        k = key[t["id"]]
        p = reg[k["figure_id"]]
        gt = [c for c in gt_all[k["figure_id"]] if c.get("x")]
        items.append(DatasetItem(
            figure_id=f"{k['paper_id']}-{k['figure_id']}",
            task=ExtractionTask(
                image_bytes=(REPO / p.image_path).read_bytes(),
                # Always the registry's current ranges, never the ones the
                # model was shown: replay ignores them, but the scorer's span
                # floor (design 7.66) is set from them, and a task file from
                # before a unit migration holds the old space (28500).
                x_range=tuple(p.x_range),
                y_range=tuple(p.y_range),
                x_scale=p.x_scale,
                y_scale=p.y_scale,
            ),
            # Built exactly as run_baselines.py's _ground_truth_for does --
            # no scale arguments, series_label from prop_y -- so these figures
            # are scored against the same ground truth the CV baselines face.
            ground_truth=[Curve(x_values=tuple(c["x"]), y_values=tuple(c["y"]),
                                series_label=c.get("prop_y")) for c in gt],
        ))
        order.append(t["id"])

    gt_rev = ground_truth_revision(GROUND_TRUTH_SUPPLEMENT_DIR)
    written = []
    for model_id, model_name in cond.get("models", MODELS).items():
        raw = {}
        if cond.get("v2"):
            files = sorted((V2_ARCHIVE / cond["v2"] / model_id).glob("part*.predictions.json"))
            for f in files:
                raw |= {f"{name}:{k}": v for k, v in json.loads(f.read_text()).items()}
            if not files:
                print(f"  {model_id}: 予測ファイルがない → スキップ")
                continue
        for part in [] if cond.get("v2") else parts:
            c = CONDITIONS[part]
            path = _resolve_pred(c["archive"], c["work"], model_id)
            if path is None:
                break
            raw |= {f"{part}:{k}": v for k, v in json.loads(path.read_text()).items()}
        else:
            path = True
        if path is None:
            print(f"  {model_id}: 予測ファイルがない → スキップ")
            continue
        preds = {}
        for t in tasks:
            if t["id"] not in raw:
                continue
            answer = raw[t["id"]]
            # v2 answers were given in today's unit space; only the 2026-09
            # runs predate the display-unit migration
            rescale = (
                PREDICTION_RESCALE_V2_CALIBRATED
                if cond.get("v2") == "calibrated"
                else {}
                if cond.get("v2")
                else PREDICTION_RESCALE
            )
            factors = rescale.get(key[t["id"]]["figure_id"])
            if factors:
                answer = [
                    {
                        "x": [v * factors.get("x", 1.0) for v in c.get("x", [])],
                        "y": [v * factors.get("y", 1.0) for v in c.get("y", [])],
                    }
                    for c in answer
                ]
            preds[t["id"]] = parse_curves(answer, reg[key[t["id"]]["figure_id"]].x_scale)
        results = evaluate_model_on_dataset(
            ReplayRunner(preds, order), items, matcher_for=matcher_for_task
        )
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
            "model_name": model_name + cond["name_suffix"],
            "dataset_version": (
                f"v0-eval-pilot-n{len(reg_scoreable)}{gt_rev}-noaxis"
                if cond.get("v2") == "noaxis"
                # every scoreable figure, same as the CV/LineFormer rows, so all
                # of them rank in one table (design 7.66)
                else f"v0-eval-pilot-n{len(reg_scoreable)}{gt_rev}"
                if len(parts) > 1 or cond.get("v2")
                else f"v0-eval-pilot-n{len(reg_scoreable)}{gt_rev}-llm-subset-"
                f"n{len(items)}{cond['suffix']}"),
            "run_at": datetime.now(UTC).isoformat(),
            "n_figures": len(per_figure),
            "metric": "normalized-y-distance, span floor 5% of linear y axis (design 7.66)",
            "mean_summary_score": sum(p["summary_score"] for p in per_figure) / len(per_figure),
            "per_figure": per_figure,
            "agent_effort": EFFORT.get(model_id) if name == "calibrated" else None,
            "condition": cond["label"],
            "notes": V2_NOTES if cond.get("v2") else (
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
            print(f"  {i}. {cond.get('models', MODELS)[m]:<20} {s:.4f}")


if __name__ == "__main__":
    main()
