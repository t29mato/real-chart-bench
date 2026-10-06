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

from real_chart_bench.adapter.agent_run_archive import (  # noqa: E402
    OFFICIAL_PARTS,
    load_agent_run_parts,
)
from real_chart_bench.adapter.ground_truth_store import (  # noqa: E402
    ground_truth_revision,
    load_ground_truth,
)
from real_chart_bench.adapter.local_vlm_run import load_local_vlm_run  # noqa: E402
from real_chart_bench.adapter.printed_space import (  # noqa: E402
    PRINTED_SPACE_MIGRATION,
    answer_to_printed,
    answer_to_printed_if_old,
)
from real_chart_bench.adapter.tick_plot_areas import (  # noqa: E402
    load_tick_calibration,
    values_from_pixel_answer,
)
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.domain.curve import Curve, ScaleType  # noqa: E402
from real_chart_bench.usecase.evaluate_dataset import (  # noqa: E402
    PRIMARY_POINT_TAU,
    DatasetItem,
    evaluate_model_on_dataset,
    matcher_for_task,
)
from real_chart_bench.usecase.model_runner import ExtractionTask  # noqa: E402
from real_chart_bench.usecase.real_image_gate import select_verified_pairings  # noqa: E402
from real_chart_bench.usecase.result_payload import (  # noqa: E402
    POINT_METRIC_LABEL,
    aggregate_dense_marker_metrics,
    aggregate_point_metrics,
    figure_result_row,
)

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

# Local VLMs (design §7.69, §7.73 (3)): the same v2 tasks, answered in one
# inference per figure on a laptop with mlx-vlm instead of by an agent. Raw
# output lives in data/local_vlm_run_v2/<model>/<condition>.jsonl; only the
# parsed series list is replayed, and a figure whose output did not parse is
# left out -- a total miss, as for an LLM that did not answer. The tasks, key,
# ground truth and dataset_version are the v2 run's, so these rows rank in the
# same table as the Claude v2 rows of the same condition.
LOCAL_V2_ARCHIVE = REPO / "data/local_vlm_run_v2"
MODELS_LOCAL_V2 = {
    "qwen3.5-9b-8bit": "Qwen3.5-9B (8bit)",
    "qwen3.8-27b-8bit": "Qwen3.8-27B (8bit)",
    "gemma-4-31b-8bit": "Gemma 4 31B (8bit)",
}
CONDITIONS["local-v2-calibrated"] = {
    "v2": "calibrated",
    "local": True,
    "models": MODELS_LOCAL_V2,
    "suffix": "-local-v2",
    "name_suffix": "（ローカル、2026-10-03）",
    "label": "軸レンジを与えた条件（ローカル VLM、v2 プロンプトの単発版、採点対象全図）",
}
CONDITIONS["local-v2-noaxis"] = {
    "v2": "noaxis",
    "local": True,
    "models": MODELS_LOCAL_V2,
    "suffix": "-local-v2-noaxis",
    "name_suffix": "（軸レンジなし、ローカル、2026-10-03）",
    "label": "軸レンジを与えない条件（ローカル VLM、v2 プロンプトの単発版、採点対象全図）",
}
# LLM run v3 (design §7.73 (2), §7.74): v2 with one change to the prompt --
# measured points (markers) only, no fit / trend / guide / theory lines. Same
# figures, conditions, batch split and contamination guards; new seed. Its
# tasks were built from the registry of 2026-10-04, so the calibrated ranges
# the models saw already equal the registry's (checked: no figure differs) and
# no answer is rescaled. Same dataset_version as v2: the rows rank in the same
# table, v2 stays there as history, the way the 2026-09 runs did.
V3_ARCHIVE = REPO / "data/llm_run_v3"
CONDITIONS["v3-calibrated"] = {
    "v2": "calibrated",
    "run_dir": V3_ARCHIVE,
    "rescale": {},
    "models": MODELS_V2,
    "prompt": "v3",
    "suffix": "-r3",
    "name_suffix": "（v3 プロンプト、2026-10-04）",
    "label": "軸レンジを与えた条件（2026-10-04 v3 プロンプト実行、採点対象全図）",
}
CONDITIONS["v3-noaxis"] = {
    "v2": "noaxis",
    "run_dir": V3_ARCHIVE,
    "rescale": {},
    "models": MODELS_V2,
    "prompt": "v3",
    "suffix": "-r3-noaxis",
    "name_suffix": "（軸レンジなし、v3 プロンプト、2026-10-04）",
    "label": "軸レンジを与えない条件（2026-10-04 v3 プロンプト実行、採点対象全図）",
}
# Diagnostic only, never a leaderboard row (design §7.74): the first v3
# Sonnet attempts, made while the subagent's python3 had no Pillow, so it read
# every value by eye. Same model, prompt and figures as the official rerun --
# the difference between the two is what image tools are worth. Flagged
# `diagnostic` (the leaderboard drops it) and on its own dataset_version.
V3_NOPILLOW_PARTS = ("part1_nopillow", "part2_nopillow")
for _c in ("calibrated", "noaxis"):
    CONDITIONS[f"v3-{_c}-nopillow"] = {
        **CONDITIONS[f"v3-{_c}"],
        "models": {"claude-sonnet-5-5": "Claude Sonnet 5.5"},
        "pred_parts": V3_NOPILLOW_PARTS,
        "diagnostic": True,
        "suffix": "-r3-nopillow" + ("-noaxis" if _c == "noaxis" else ""),
        "name_suffix": " (v3, no image tools)"
        + ("（軸レンジなし）" if _c == "noaxis" else ""),
        "label": CONDITIONS[f"v3-{_c}"]["label"]
        + " — 診断行: 画像ツールなしの初回回答(リーダーボード対象外)",
    }
# Local VLMs on the v3 prompt (design §7.73 (2)/(3)): the same three models,
# a single-shot version of the v3 prompt (scripts/eval/local_vlm/prompt_v3/),
# the v3 run's tasks and key, no rescale (like Claude v3). The models ran one
# at a time, so seconds per figure are comparable between them. Same
# dataset_version as the main rows; the local v2 rows stay as history.
LOCAL_V3_ARCHIVE = REPO / "data/local_vlm_run_v3"
for _c in ("calibrated", "noaxis"):
    CONDITIONS[f"local-v3-{_c}"] = {
        "v2": _c,
        "run_dir": V3_ARCHIVE,
        "rescale": {},
        "local": True,
        "local_archive": LOCAL_V3_ARCHIVE,
        "sequential": True,
        "models": MODELS_LOCAL_V2,
        "local_prompt": "v3",
        "suffix": "-local-v3" + ("-noaxis" if _c == "noaxis" else ""),
        "name_suffix": "（"
        + ("軸レンジなし、" if _c == "noaxis" else "")
        + "ローカル、v3 プロンプト、2026-10-04）",
        "label": ("軸レンジを与えない条件" if _c == "noaxis" else "軸レンジを与えた条件")
        + "（ローカル VLM、v3 プロンプトの単発版、採点対象全図）",
    }
# design 7.75 (2): how much does quantization cost? Qwen3.5-9B on the RTX 4090
# with vLLM, the v3 single-shot prompt, one precision per row, all made from
# the same published weights (bf16 as is, fp8 converted in flight, w8a16 /
# w4a16 by scripts/eval/local_vlm/quantize_rtn.py).
LOCAL_CUDA_ARCHIVE = REPO / "data/local_vlm_run_cuda"
LOCAL_CUDA_PRECISIONS = {
    "bf16": "bf16",
    "fp8": "FP8",
    "w8a16": "8bit RTN",
    "w4a16": "4bit RTN",
}
for _p, _label in LOCAL_CUDA_PRECISIONS.items():
    for _c in ("calibrated", "noaxis"):
        CONDITIONS[f"local-cuda-{_p}-{_c}"] = {
            **CONDITIONS[f"local-v3-{_c}"],
            "local_archive": LOCAL_CUDA_ARCHIVE,
            "cuda": True,
            "models": {f"qwen3.5-9b-{_p}": f"Qwen3.5-9B ({_label})"},
            "suffix": "-local-cuda-v3" + ("-noaxis" if _c == "noaxis" else ""),
            "name_suffix": "（"
            + ("軸レンジなし、" if _c == "noaxis" else "")
            + "ローカル RTX 4090、v3 プロンプト、2026-10-05）",
            "label": ("軸レンジを与えない条件" if _c == "noaxis" else "軸レンジを与えた条件")
            + "（Qwen3.5-9B、RTX 4090 + vLLM、精度別、v3 プロンプトの単発版、採点対象全図）",
        }
# design 7.76/7.77: tick pixel positions given ("pixcal") -- the person
# calibrates the axes (two ticks per axis, pixel + value), the AI extracts the
# points. Claude: v3 prompt with a calibration block (llm_run_pixcal_prompt.md),
# sealed dirs, answers archived to data/llm_run_pixcal/. Local: the same block
# in the single-shot prompt (local_vlm/prompt_v3/condition_pixcal.md).
PIXCAL_ARCHIVE = REPO / "data/llm_run_pixcal"
CONDITIONS["pixcal"] = {
    "v2": "pixcal",
    "run_dir": PIXCAL_ARCHIVE,
    "rescale": {},
    "models": MODELS_V2,
    "prompt": "pixcal",
    "suffix": "-pixcal",
    "name_suffix": "（目盛のピクセル位置あり、2026-10-05）",
    "label": "目盛のピクセル位置を与えた条件(人が軸を校正し AI が点を取る、採点対象全図)",
    "notes": (
        "Claude Code のサブエージェントとして起動(2026-10-05〜06、Linux 機)。"
        "プロンプトは scripts/eval/llm_run_pixcal_prompt.md: v3 プロンプトの条件ブロックだけを"
        "『軸ごとに目盛2本のピクセル位置と値・スケール・画像サイズ』"
        "(WebPlotDigitizer の軸校正と同じ情報、data/verified_pairs/tick_calibration.json)に"
        "置き換えた。軸レンジは与えない。図・正解は v3 と同じ採点対象94図、2バッチ(47図/47図)、"
        "封印ディレクトリと INSTRUCTIONS.md は v3 と同じ運用で、乱数シードのみ変更。"
        "4モデル8バッチを並列に実行した。実行記録は design §7.78。生の回答は data/llm_run_pixcal/。"
    ),
}
CONDITIONS["local-cuda-bf16-pixcal"] = {
    "v2": "pixcal",
    "run_dir": PIXCAL_ARCHIVE,
    "rescale": {},
    "local": True,
    "local_archive": REPO / "data/local_vlm_run_cuda_pixcal",
    "sequential": True,
    "cuda": True,
    "models": {"qwen3.5-9b-bf16": "Qwen3.5-9B (bf16)"},
    "suffix": "-local-cuda-pixcal",
    "name_suffix": "（目盛のピクセル位置あり、ローカル RTX 4090、2026-10-05）",
    "label": "目盛のピクセル位置を与えた条件（Qwen3.5-9B、RTX 4090 + vLLM、採点対象全図）",
    "notes": (
        "Qwen3.5-9B (bf16) を RTX 4090 + vLLM で、量子化行(local-cuda-*)と同じ設定"
        "(単発プロンプト・JSON スキーマ・温度0・thinking 無効・max_tokens 8192)で実行した。"
        "差分は条件ブロックだけで、Claude の pixcal 行と同じ校正情報"
        "(scripts/eval/local_vlm/prompt_v3/condition_pixcal.md)を与えた。"
        "生出力は data/local_vlm_run_cuda_pixcal/。"
    ),
}
# design 7.79: two-stage pixcal for single-shot VLMs -- the model gives the
# markers' image positions only, and the scorer converts them with the same
# person-measured tick calibration the pixcal row hands to the agents. Same
# information, same version: the rows rank with pixcal.
for _coords, _tag in (("pixel", "px"), ("norm1000", "norm")):
    CONDITIONS[f"local-cuda-bf16-pixpts-{_tag}"] = {
        **CONDITIONS["local-cuda-bf16-pixcal"],
        "v2": f"pixpts_{_tag}",
        "version_tag": "pixcal",
        "pixel_answer": _coords,
        # converted through tick_calibration.json, which is in printed space
        "answers_in_printed_space": True,
        "suffix": f"-local-cuda-pixpts-{_tag}",
        "name_suffix": "（2段階: モデルは"
        + ("画素座標" if _coords == "pixel" else "0〜1000 の相対座標")
        + "、値への変換は目盛校正で、ローカル RTX 4090）",
        "label": "目盛のピクセル位置を与えた条件・2段階(Qwen3.5-9B、RTX 4090 + vLLM、採点対象全図)",
        "notes": (
            "pixcal の2段階版(design 7.79)。単発の VLM は校正式で値に換算できないため、"
            "モデルにはマーカー中心の画像内の位置だけを出させ("
            + (
                "画像ファイルの画素座標、原点左上・y 下向き"
                if _coords == "pixel"
                else "幅・高さを 0〜1000 とする相対座標、Qwen-VL のグラウンディングの慣習"
            )
            + ")、値への変換は採点側で、Claude の pixcal 行に渡したのと同じ目盛校正"
            "(data/verified_pairs/tick_calibration.json)で行った。"
            "プロンプトは v3 単発版の条件ブロックを "
            f"scripts/eval/local_vlm/prompt_v3/condition_pixpts_{_tag}.md に替えたもの。"
            "生出力は data/local_vlm_run_cuda_pixcal/。"
        ),
    }
# docs/design/local-model.md approach A: a trained marker detector gives marker
# centres in pixels; the scorer converts them through the same person-measured
# tick calibration as the pixpts rows (condition 2). One condition per run dir
# in data/local_model_runs/<run>/ (pixpts_px.jsonl + env.json).
LOCAL_MODEL_ARCHIVE = REPO / "data/local_model_runs"
for _env in sorted(LOCAL_MODEL_ARCHIVE.glob("*/env.json")):
    _run = _env.parent.name
    _meta = json.loads(_env.read_text())
    CONDITIONS[f"local-cuda-detector-{_run}"] = {
        **CONDITIONS["local-cuda-bf16-pixcal"],
        "v2": "pixpts_px",
        "version_tag": "pixcal",
        "pixel_answer": "pixel",
        "answers_in_printed_space": True,
        "local_archive": LOCAL_MODEL_ARCHIVE,
        "models": {_run: _meta["display_name"]},
        "suffix": "-local-cuda-pixcal-detector",
        "name_suffix": "（マーカー検出器、画素座標を目盛校正で換算、ローカル RTX 4090）",
        "label": "目盛のピクセル位置を与えた条件・検出器(方式A、RTX 4090、採点対象全図)",
        "notes": (
            "方式A(docs/design/local-model.md「方式A: 検出器」)。学習したマーカー検出器"
            "(" + _meta["architecture"] + ")がマーカー中心の画素位置と系列の振り分けを出し、"
            "値への変換は採点側で、Claude の pixcal 行と同じ目盛校正"
            "(data/verified_pairs/tick_calibration.json)で行った。"
            "学習データ: " + _meta.get("data", "") + "。"
            "しきい値などの設定は学習データから切り出した検証分割で決め、ベンチマークでは調整していない。"
            f"生出力は data/local_model_runs/{_run}/。"
        ),
    }
# docs/design/local-model.md 方式C: the detector's pixels converted by an
# automatic axis calibration (frame rules + Tesseract tick OCR) -- no person
# in the loop, so condition 1 (noaxis). Values are in the printed space (the
# OCR reads the printed labels). One condition per run dir with noaxis.jsonl.
for _env in sorted(LOCAL_MODEL_ARCHIVE.glob("*/noaxis.jsonl")):
    _run = _env.parent.name
    _meta = json.loads((_env.parent / "env.json").read_text())
    CONDITIONS[f"local-cuda-hybrid-{_run}"] = {
        **CONDITIONS["local-cuda-bf16-noaxis"],
        "answers_in_printed_space": True,
        "local_archive": LOCAL_MODEL_ARCHIVE,
        "models": {_run: _meta["display_name"]},
        "suffix": "-local-cuda-noaxis-hybrid",
        "name_suffix": "（分業型: 検出器 + 目盛 OCR の自動校正、軸レンジなし、ローカル）",
        "label": "軸レンジを与えない条件・分業型(方式C、検出器 + 自動校正、採点対象全図)",
        "notes": (
            "方式C(docs/design/local-model.md「方式C: 分業型」)。学習したマーカー検出器"
            "(" + _meta["architecture"] + ")の画素位置と系列の振り分けを、"
            "規則による枠・目盛の検出と Tesseract 5.3.4 による目盛ラベルの OCR で求めた"
            "軸の当てはめ(adapter/auto_axis_calibration.py)で値に直した。人の校正も"
            "Claude / GPT も使わない。自動校正に失敗した図は回答なし(全点ミス)。"
            "検出の後処理は学習データの検証分割と合成のストレス検証セットで決め、"
            "ベンチマークでは調整していない。値は印字空間。"
            f"生出力は data/local_model_runs/{_run}/noaxis.jsonl。"
        ),
    }
# docs/design/local-model.md 方式D: an orchestrator (local VLM, or a fixed
# script as the check) chooses and configures fixed tools
# (adapter/orchestrator_tools.py); every number comes from a tool.
# Condition 1: values through the calibration it chose (printed space);
# condition 2: pixel points, converted by the scorer through the person's
# tick calibration as for the detector rows. One run dir per run in
# data/local_orchestrator_runs/<run>/ (scripts/eval/orchestrator/run_local.py).
LOCAL_ORCH_ARCHIVE = REPO / "data/local_orchestrator_runs"
_ORCH_NOTES = (
    "方式D(docs/design/local-model.md「方式D: 司令塔 + 道具」)。"
    "司令塔は道具の選択と設定だけを行い、答えの数値はすべて道具"
    "(starry-digitizer の Symbol Extract / Line Extract の移植、方式A の検出器、"
    "方式C の目盛 OCR、色の候補、マスク、重ね描き)から来る。"
    "司令塔は最後に、どの道具の結果のどの系列を答えにするかを名指しする。"
    "1図あたり最大 12 手、終わらない図は固定の代替(最後の抽出結果の全系列)。"
    "プロンプトは scripts/eval/orchestrator/prompt.md、生出力・手順の記録は "
)
for _env in sorted(LOCAL_ORCH_ARCHIVE.glob("*/env.json")):
    _run = _env.parent.name
    _meta = json.loads(_env.read_text())
    if (_env.parent / "noaxis.jsonl").exists():
        CONDITIONS[f"local-orch-{_run}-noaxis"] = {
            **CONDITIONS["local-cuda-bf16-noaxis"],
            "answers_in_printed_space": True,
            "local_archive": LOCAL_ORCH_ARCHIVE,
            # the scripted policy only checks the loop (it is 方式C's -posta row again)
            **({"diagnostic": True} if _meta.get("policy") == "scripted" else {}),
            "models": {_run: _meta["display_name"]},
            "suffix": "-local-orch-noaxis",
            "name_suffix": "（司令塔 + 道具、軸レンジなし、ローカル）",
            "label": "軸レンジを与えない条件・司令塔 + 道具(方式D、採点対象全図)",
            "notes": _ORCH_NOTES + f"data/local_orchestrator_runs/{_run}/。"
            "主条件1: 司令塔が選んだ目盛校正(自動 OCR か、司令塔が読んだ目盛)で"
            "値に直した。値は印字空間。",
        }
    if (_env.parent / "pixpts_px.jsonl").exists():
        CONDITIONS[f"local-orch-{_run}-pixcal"] = {
            **CONDITIONS["local-cuda-bf16-pixcal"],
            "v2": "pixpts_px",
            "version_tag": "pixcal",
            "pixel_answer": "pixel",
            "answers_in_printed_space": True,
            "local_archive": LOCAL_ORCH_ARCHIVE,
            **({"diagnostic": True} if _meta.get("policy") == "scripted" else {}),
            "models": {_run: _meta["display_name"]},
            "suffix": "-local-orch-pixcal",
            "name_suffix": "（司令塔 + 道具、画素座標を目盛校正で換算、ローカル）",
            "label": "目盛のピクセル位置を与えた条件・司令塔 + 道具(方式D、採点対象全図)",
            "notes": _ORCH_NOTES + f"data/local_orchestrator_runs/{_run}/。"
            "主条件2: 答えは画素座標で、値への変換は採点側で人の目盛校正"
            "(data/verified_pairs/tick_calibration.json)で行った。",
        }
# docs/design/local-model.md 方式B: Qwen3.5-9B + a QLoRA adapter trained on
# synthetic / Starrydata charts (scripts/train/vlm_lora/), served by vLLM's LoRA
# support (scripts/eval/local_vlm/worker_ft.py). Same prompt, schema and
# decoding as the local-cuda bf16 rows, image capped at the training size; the
# adapter-free run under the same cap is the control. One row per run that
# has an archived answer file, so a run that has not happened yet is no row.
LOCAL_FT_ARCHIVE = REPO / "data/local_vlm_run_ft"
LOCAL_FT_MODELS = {
    "qwen3.5-9b-cap1600k": "Qwen3.5-9B (bf16、追加学習なし、画像 1.6MP 上限 = 対照)",
    "qwen3.5-9b-qlora-synth": "Qwen3.5-9B + QLoRA (合成データ)",
    "qwen3.5-9b-qlora-synth-real": "Qwen3.5-9B + QLoRA (合成 + Starrydata 実図)",
}
LOCAL_FT_NOTES = (
    "方式B(docs/design/local-model.md「方式B: VLM 追加学習」)。Qwen3.5-9B を QLoRA"
    "(NF4、視覚塔は凍結・非量子化、言語側の全線形層に LoRA)で追加学習し、"
    "v3 単発プロンプトに値の表を直接答えさせた。教師はベンチマーク外の合成図・既存データセット・"
    "Starrydata の人手デジタイズのみ(Claude / GPT の出力は使わない)。"
    "ベンチマークの論文は論文単位で学習から除外し、調整は学習データの検証分割だけで行った。"
    "推論は bf16 の公開重み + LoRA を vLLM で(local-cuda の bf16 行と同じ設定)、"
    "画像は学習と同じ 1.6MP 上限に縮小。生出力・設定(adapter と学習状態)は data/local_vlm_run_ft/。"
)
for _c, _base in (("noaxis", "local-cuda-bf16-noaxis"), ("pixcal", "local-cuda-bf16-pixcal")):
    CONDITIONS[f"local-cuda-ft-{_c}"] = {
        **CONDITIONS[_base],
        "local_archive": LOCAL_FT_ARCHIVE,
        "sequential": True,
        "models": {
            m: label
            for m, label in LOCAL_FT_MODELS.items()
            if (LOCAL_FT_ARCHIVE / m / f"{_c}.jsonl").exists()
        },
        "suffix": "-local-cuda-ft" + ("-v3-noaxis" if _c == "noaxis" else "-pixcal"),
        "name_suffix": "（"
        + ("軸レンジなし" if _c == "noaxis" else "目盛のピクセル位置あり")
        + "、ローカル RTX 4090、方式B 追加学習）",
        "label": ("軸レンジを与えない条件" if _c == "noaxis" else "目盛のピクセル位置を与えた条件")
        + "（Qwen3.5-9B + QLoRA、RTX 4090 + vLLM、採点対象全図）",
        "notes": LOCAL_FT_NOTES,
    }
# 方式D with Claude as the orchestrator: the upper bound of choosing the fixed
# tools. Sealed dirs by scripts/eval/prepare_orchestrator_claude.py; the agent
# may only run the `tool` command and look at images / overlays. Answers in
# data/llm_run_orchestrator/<noaxis|pixpts_px>/<model>/<part>.predictions.json
# (archive_orchestrator_claude.py): values (printed space) in condition 1,
# image pixels converted through the person's calibration in condition 2.
ORCH_CLAUDE_ARCHIVE = REPO / "data/llm_run_orchestrator"
for _c, _v2, _src in (("noaxis", "noaxis", "data/llm_run_v3"),
                      ("pixcal", "pixpts_px", "data/llm_run_pixcal")):
    CONDITIONS[f"orch-claude-{_c}"] = {
        "v2": _v2,
        "run_dir": REPO / _src,
        "pred_root": ORCH_CLAUDE_ARCHIVE,
        "rescale": {},
        # condition 2 converts pixels through the (printed) tick calibration;
        # condition 1's sealed tasks still carried the old per-axis rules for
        # 14 axes, and an answer may follow them or come out of a tool that read
        # the printed ticks -- decided per figure from the values (design 7.82)
        **(
            {"pixel_answer": "pixel", "version_tag": "pixcal", "answers_in_printed_space": True}
            if _c == "pixcal"
            else {"answer_space": "auto"}
        ),
        "models": {m: label for m, label in MODELS_V2.items()
                   if (ORCH_CLAUDE_ARCHIVE / _v2 / m).is_dir()},
        "pred_parts": ["part1", "part2"],
        "suffix": f"-orch-{_c}",
        "name_suffix": "（司令塔 + 道具のみ、"
        + ("軸レンジなし" if _c == "noaxis" else "目盛のピクセル位置あり") + "）",
        "label": ("軸レンジを与えない条件" if _c == "noaxis" else "目盛のピクセル位置を与えた条件")
        + "・Claude が司令塔、決まった道具だけを使う(方式D の上限、採点対象全図)",
        "notes": (
            "方式D(docs/design/local-model.md「方式D: 司令塔 + 道具」)の上限を測る行。"
            "Claude Code のサブエージェントを封印ディレクトリで起動し、用意した道具コマンド"
            "(starry-digitizer の Symbol Extract / Line Extract の移植、方式A の検出器、"
            "方式C の目盛 OCR、色の候補、マスク、重ね描き)と画像・重ね描きを見ることだけを許した。"
            "自分で画像処理のコードを書くことと、答えの数値を手で打つことは指示で禁止した"
            "(scripts/eval/orchestrator/claude_prompt.md)。答えは道具の answer コマンドが"
            "保存済みの道具の結果から書く。"
            + ("目盛は自動 OCR か、司令塔が読んだ目盛(manual)で校正し、値は印字空間。"
               if _c == "noaxis" else
               "答えは画素座標で、値への変換は採点側で人の目盛校正で行った。")
            + "推論時の利用だけで、学習には使わない。回答は data/llm_run_orchestrator/。"
        ),
    }
# design 7.81: GPT through the Codex CLI (ChatGPT sign-in, no API key), the
# same sealed-directory setup and v3 prompt as the Claude agents. A pilot on a
# seeded subset of the v3 figures first (the subscription's usage is small);
# its rows carry a subset version so they never rank with the full set.
CONDITIONS["codex-pilot10-calibrated"] = {
    "v2": "calibrated",
    "run_dir": REPO / "data/llm_run_codex/pilot10",
    "pred_root": REPO / "data/llm_run_codex",
    "rescale": {},
    "models": {"gpt-6.1-sol": "GPT-6.1-Sol (Codex CLI)"},
    "pred_parts": ["pilot10"],
    "subset_tag": "codex-pilot10",
    "suffix": "-codex-pilot10",
    "name_suffix": "（Codex CLI、軸レンジあり、10図の試行、2026-10-06）",
    "label": "Codex CLI の試行(v3 の10図、軸レンジあり)",
    "notes": (
        "OpenAI Codex CLI 0.160.0 を ChatGPT アカウント(Plus)でログインして実行した(API キーなし)。"
        "codex exec -m gpt-6.1-sol -c model_reasoning_effort=medium -s workspace-write"
        "(書き込みは作業ディレクトリのみ、ネットワークなし)。"
        "指示文は Claude v3 と同じ llm_run_v3_prompt.md の軸あり版を INSTRUCTIONS.md として"
        "封印ディレクトリに置いた。図は v3 の採点対象から固定シードで選んだ10図"
        "(prepare_llm_run_codex.py)。回答・イベントログ・最終メッセージは "
        "data/llm_run_codex/calibrated/。"
    ),
}
# the full Codex rows: the Claude run's tasks and key (v3 / pixcal), answers
# from data/llm_run_codex/<condition>/<model>/ -- the calibrated pilot's ten
# figures plus the batches that covered the rest
CODEX_MODELS = {"gpt-6.1-sol": "GPT-6.1-Sol (Codex CLI)", "gpt-5.5": "GPT-5.5 (Codex CLI)"}
CODEX_NOTES = (
    "OpenAI Codex CLI 0.160.0 を ChatGPT アカウント(Plus)でログインして実行した"
    "(API キーなし、2026-10-06)。"
    "codex exec -m <model> -c model_reasoning_effort=medium -s workspace-write"
    "(書き込みは作業ディレクトリのみ、ネットワークなし)。図・タスク・fig_NNN の名前・指示文は、"
    "同じ条件の Claude の実行(v3 / pixcal)と同一で、封印ディレクトリに INSTRUCTIONS.md として置いた"
    "(prepare_llm_run_codex.py)。回答・イベントログ・最終メッセージは data/llm_run_codex/。"
    "design §7.81。"
)
for _c, _src, _label in (
    ("calibrated", "data/llm_run_v3", "軸レンジあり"),
    ("noaxis", "data/llm_run_v3", "軸レンジなし"),
    ("pixcal", "data/llm_run_pixcal", "目盛のピクセル位置あり"),
):
    CONDITIONS[f"codex-{_c}"] = {
        "v2": _c,
        "run_dir": REPO / _src,
        "pred_root": REPO / "data/llm_run_codex",
        "rescale": {},
        "models": CODEX_MODELS,
        "pred_parts": ["pilot10", "part1", "part2"] if _c == "calibrated" else ["part1", "part2"],
        "suffix": "-codex" + ("" if _c == "calibrated" else f"-{_c}"),
        "name_suffix": f"（Codex CLI、{_label}、2026-10-06）",
        "label": f"{_label}の条件(GPT、Codex CLI、採点対象全図)",
        "notes": CODEX_NOTES,
    }
DIAGNOSTIC_VERSION_SUFFIX = "-diagnostic-no-image-tools"
V3_NOTES = (
    "Claude Code のサブエージェントとして起動(2026-10-04、"
    "オーナーの Mac (macOS) で実行。v2 は Linux 機で別マシン)。"
    "プロンプトは scripts/eval/llm_run_v3_prompt.md: "
    "v2 (llm_run_v2_prompt.md)との差分は『測定点(マーカー)のみ。"
    "近似線・回帰線・ガイド線・理論曲線は出さない』の1点だけ(design §7.73 (2))。"
    "図・条件・バッチ分割(49図/48図)・汚染対策は v2 と同じで、乱数シードのみ変更。"
    "指示文は起動メッセージに貼らず、"
    "各封印ディレクトリ内の INSTRUCTIONS.md として置いた。"
    "Sonnet 5.5 は初回、"
    "サブエージェントの python3 に Pillow がなく全バッチを目視で回答したため、"
    "起動メッセージで Pillow 入りインタプリタのパスを明示して再実行した"
    "(公式行は再実行分 part1/part2 のみ。"
    "初回分 *_nopillow は診断行として別ファイル)。"
    "API の利用上限で中断したバッチがいくつかあり、"
    "上限のリセット後に同じエージェントを文脈ごと再開した。"
    "自己申告の封印逸脱: Sonnet 再実行(軸あり part2)が"
    "一時ファイルをディレクトリ外の /private/tmp/w に書いた。"
    "Opus(軸なし part2)が作業ディレクトリの根を一度一覧し、"
    "_key.json を含むファイル名を見た(開いてはいない)。"
    "実行記録は design §7.74。"
    "calibrated の軸レンジは 2026-10-04 の registry から作ったので、"
    "回答の単位換算はしていない。生の回答は data/llm_run_v3/。"
)
V3_NOPILLOW_NOTES = (
    "診断行(リーダーボード対象外)。"
    "v3 の Sonnet 5.5 初回回答: サブエージェントの既定 python3 に Pillow がなく、"
    "画像処理を使わず目視で値を読んだ。同じモデル・同じプロンプト・同じ図で、"
    "公式行(claude-sonnet-5-5-v0-r3*.json、"
    "Pillow ありで再実行)との差が画像ツールの有無の効果。"
    "生の回答は data/llm_run_v3/<condition>/claude-sonnet-5-5/part{1,2}_nopillow.predictions.json。"
)

LOCAL_HARDWARE = "Apple M3 Max, 128GB unified memory (laptop)"
LOCAL_V2_NOTES = (
    "ローカル実行(ノート PC: Apple M3 Max 128GB、mlx-vlm 0.7.4 / mlx 0.32.3、2026-10-03〜04)。"
    "1図1回の推論で、エージェントではない(ツールなし)。プロンプトは LLM run v2 "
    "(scripts/eval/llm_run_v2_prompt.md)からエージェント向けの文を除いた単発版"
    "(scripts/eval/local_vlm/prompt/、差分は diff_vs_v2.diff)。図・タスク・軸レンジ・正解は "
    "Claude の v2 実行と同一。設定: greedy(温度0)、enable_thinking=False、JSON スキーマでの"
    "制約付きデコード、max_tokens 8192、HF_HUB_OFFLINE=1(ネットワークなしで実行)、"
    "画像は元ファイルを"
    "そのまま渡す。Gemma 4 は画像トークン予算を 1120 に上げた(processor.image_processor."
    "max_soft_tokens、既定 280)。パースできなかった図・実行時エラーの図は回答なし(全図ミス)として"
    "採点する(LLM と同じ規則)。件数は local_run を参照。3モデルを同時に並列実行したため、"
    "1図あたりの秒数はモデル間でも単独実行とも比較できない(記録はしているが使わない)。"
    "生出力(raw テキストを含む)は data/local_vlm_run_v2/、推論コードは scripts/eval/local_vlm/。"
)

LOCAL_V3_NOTES = (
    "ローカル実行(ノート PC: Apple M3 Max 128GB、mlx-vlm 0.7.4 / mlx 0.32.3、2026-10-04〜05)。"
    "1図1回の推論で、エージェントではない(ツールなし)。プロンプトは LLM run v3 "
    "(scripts/eval/llm_run_v3_prompt.md、測定点(マーカー)のみ)からエージェント向けの文を除いた単発版"
    "(scripts/eval/local_vlm/prompt_v3/、v2 単発版との差分は single_shot_v2_to_v3.diff)。"
    "図・タスク・軸レンジ・正解は Claude の v3 実行と同一"
    "(data/llm_run_v3 の tasks/_key、単位換算なし)。"
    "設定は v2 のローカル実行と同じ: greedy(温度0)、enable_thinking=False、JSON スキーマでの"
    "制約付きデコード、max_tokens 8192、HF_HUB_OFFLINE=1、画像は元ファイルのまま、"
    "Gemma 4 は画像トークン予算 1120。"
    "パースできなかった図・実行時エラーの図は回答なし(全図ミス)として採点する(local_run を参照)。"
    "今回は3モデルを1つずつ順に実行した(Qwen3.5-9B → Qwen3.8-27B → Gemma 4 31B、"
    "scripts/eval/local_vlm/run_v3_chain.sh)ので、1図あたりの秒数(local_run.seconds_per_figure)は"
    "モデル間で比較できる。"
    "生出力(raw テキストを含む)と実行ログは data/local_vlm_run_v3/、"
    "推論コードは scripts/eval/local_vlm/。"
)


LOCAL_CUDA_NOTES = (
    "量子化の影響を測る行(design 7.75 (2))。Qwen3.5-9B を RTX 4090 + vLLM で、"
    "Mac の v3 ローカル実行と同じ単発プロンプト・JSON スキーマ・温度0・thinking 無効・"
    "max_tokens 8192 で"
    "実行した(プロンプト組み立てとスキーマは worker_v3.py の関数をそのまま使う)。精度だけを変え、"
    "すべて同じ公開重みから作った: bf16 はそのまま、fp8 は vLLM の変換、w8a16 / w4a16 は "
    "scripts/eval/local_vlm/quantize_rtn.py の四捨五入量子化(キャリブレーションなし、group 128)。"
    "16図ずつまとめて推論したので、1図あたりの秒数はこのマシンの精度間でのみ比べられる。"
    "生出力は data/local_vlm_run_cuda/。"
)


def _local_run_block(
    model_id: str,
    condition: str,
    run,
    key: dict,
    scoreable: set,
    archive: pathlib.Path = LOCAL_V2_ARCHIVE,
    sequential: bool = False,
) -> dict:
    """What a reader of a local row needs to judge it: the exact model, the
    settings, and how many figures produced no usable answer and why. A
    sequential run (one model at a time) also gets its seconds per figure."""
    env = json.loads((archive / model_id / "env.json").read_text())

    def figs(ids):
        # paper-figure ids, and only those still scored (4 of the 101 tasks
        # left the scoreable set after the run)
        return [
            f"{key[i]['paper_id']}-{key[i]['figure_id']}"
            for i in ids
            if key[i]["figure_id"] in scoreable
        ]

    if env.get("kind") == "orchestrator":
        recs = [json.loads(line) for line in
                (archive / model_id / f"{condition}.jsonl").read_text().splitlines() if line]
        recs = [r for r in recs if key[r["fig"]]["figure_id"] in scoreable]
        return {
            "raw_output": f"{archive.relative_to(REPO)}/{model_id}/{condition}.jsonl",
            "transcripts": f"{archive.relative_to(REPO)}/{model_id}/transcripts/",
            "policy": env["policy"],
            "model_repo_id": env.get("repo_id"),
            "model_revision": env.get("revision"),
            "engine": f"vLLM {env['vllm']}, torch {env['torch']}" if env.get("vllm") else None,
            "hardware": f"{env['gpu']} (24GB) + CPU tools, Linux" if env.get("gpu")
            else "CPU, Linux",
            "max_steps": env["max_steps"],
            "view_long_side": env.get("view_long_side"),
            "temperature": env.get("temperature"),
            "enable_thinking": env.get("enable_thinking"),
            "constrained_json_schema": env.get("constrained_json_schema"),
            "finished_by_final_action": sum(r.get("finish") == "final" for r in recs),
            "finished_by_fallback": sum(r.get("finish") == "fallback" for r in recs),
            "steps_per_figure_mean": round(sum(r.get("n_steps", 0) for r in recs)
                                           / max(1, len(recs)), 2),
            "seconds_per_figure": _seconds_stats(run, key, scoreable),
            "tokens_per_figure": _token_stats(run, key, scoreable),
            "n_scored_figures_without_answer": len(figs(run.parse_failures))
            + len(figs(run.errors)),
            "runtime_errors": figs(run.errors),
        }
    if env.get("kind") == "detector":
        return {
            "raw_output": f"{archive.relative_to(REPO)}/{model_id}/{condition}.jsonl",
            "architecture": env["architecture"],
            "training_data": env.get("data"),
            "checkpoint_step": env.get("checkpoint_step"),
            "validation_best_pixel_f1": env.get("validation_best"),
            "peak_threshold": env.get("peak_threshold"),
            "group_threshold": env.get("group_threshold"),
            "long_side": env.get("long_side"),
            "hardware": f"{env['gpu']} (24GB), Linux"
            if env.get("gpu")
            else f"CPU ({env.get('cpu_threads')} threads), Linux",
            "engine": f"torch {env['torch']}",
            "concurrent_with_other_models": not sequential,
            **({"seconds_per_figure": _seconds_stats(run, key, scoreable)} if sequential else {}),
            "n_scored_figures_without_answer": len(figs(run.parse_failures))
            + len(figs(run.errors)),
            "runtime_errors": figs(run.errors),
        }
    return {
        "raw_output": f"{archive.relative_to(REPO)}/{model_id}/{condition}.jsonl",
        "model_repo_id": env["repo_id"],
        "model_revision": env["revision"],
        **(
            {
                # design 7.75 (2): the RTX 4090 run, one precision per row
                "hardware": f"{env['gpu']} (24GB), Linux",
                "engine": f"vLLM {env['vllm']}, torch {env['torch']}",
                "precision": env["precision"],
                "quantization": env.get("local_quantization")
                or {"bf16": "none (published bf16 weights)",
                    "fp8": "vLLM in-flight FP8 (W8A8 dynamic)"}.get(env["precision"]),
                "enforce_eager": env.get("enforce_eager"),
                "VLLM_USE_FLASHINFER_SAMPLER": env.get("VLLM_USE_FLASHINFER_SAMPLER"),
                "batched_chunk": env.get("chunk"),
            }
            if "vllm" in env
            else {
                "hardware": LOCAL_HARDWARE,
                "quantization": env.get("quantization") or "8bit (MLX, per repo id)",
                "mlx_vlm": env["mlx_vlm"],
                "mlx": env["mlx"],
            }
        ),
        # 方式B: the adapter and how it was trained, the image cap, and
        # tokens per figure (docs/design/local-model.md)
        **(
            {
                "adapter": env["adapter"],
                "lora_rank": env.get("lora_rank"),
                "adapter_train_state": env.get("adapter_train_state"),
                "max_pixels": env.get("max_pixels"),
                "tokens_per_figure": _token_stats(run, key, scoreable),
            }
            if "adapter" in env
            else {}
        ),
        "transformers": env.get("transformers"),
        "temperature": env["temperature"],
        "enable_thinking": env["enable_thinking"],
        "constrained_json_schema": env["constrained_json_schema"],
        "max_tokens": env["max_tokens"],
        "hf_hub_offline": env["HF_HUB_OFFLINE"] == "1",
        "image_max_soft_tokens": env.get("image_max_soft_tokens_set"),
        "concurrent_with_other_models": not sequential,
        "peak_memory_gb_max": run.peak_memory_gb_max,
        **({"seconds_per_figure": _seconds_stats(run, key, scoreable)} if sequential else {}),
        "n_scored_figures_without_answer": len(figs(run.parse_failures)) + len(figs(run.errors)),
        "parse_failures": figs(run.parse_failures),
        "truncated_at_max_tokens": figs(run.truncated),
        "runtime_errors": figs(run.errors),
        "accepted_with_parser_warning": figs(run.accepted_with_warning),
    }


def _token_stats(run, key: dict, scoreable: set) -> dict:
    """Prompt (image included) and generated tokens per scored figure."""

    def stats(d):
        v = sorted(n for f, n in (d or {}).items() if key[f]["figure_id"] in scoreable)
        if not v:
            return None
        return {"n": len(v), "mean": round(sum(v) / len(v), 1),
                "median": v[(len(v) - 1) // 2], "max": v[-1]}

    return {"prompt": stats(run.prompt_tokens), "generated": stats(run.generation_tokens)}


def _seconds_stats(run, key: dict, scoreable: set) -> dict:
    """Wall-clock seconds per scored figure (failures included: they took the
    time too). Comparable between models only when they ran one at a time."""
    secs = sorted(v for f, v in run.seconds.items() if key[f]["figure_id"] in scoreable)
    n = len(secs)
    median = (secs[(n - 1) // 2] + secs[n // 2]) / 2
    return {
        "n": n,
        "mean": round(sum(secs) / n, 1),
        "median": round(median, 1),
        "max": round(secs[-1], 1),
        "total": round(sum(secs), 1),
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
    # A condition is one run, or (full) several runs scored together. Task
    # ids repeat across runs (fig_001.png ...), so they are namespaced by run.
    parts = cond.get("parts", [name])
    tasks, key = [], {}
    # the archived agent run these answers belong to (v2 unless named)
    run_dir = cond.get("run_dir", V2_ARCHIVE)
    if cond.get("v2"):
        v2_key = json.loads((run_dir / "_key.json").read_text())
        tasks = [
            {**t, "id": f"{name}:{t['id']}"}
            for t in json.loads((run_dir / cond["v2"] / "tasks.json").read_text())
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
    if cond.get("pixel_answer"):
        tick_cal = load_tick_calibration(REPO / "data/verified_pairs/tick_calibration.json")
        from PIL import Image

        image_sizes = {t["id"]: Image.open(REPO / key[t["id"]]["image_path"]).size for t in tasks}
    written = []
    for model_id, model_name in cond.get("models", MODELS).items():
        raw = {}
        local_run = None
        if cond.get("local"):
            local_run = load_local_vlm_run(
                cond.get("local_archive", LOCAL_V2_ARCHIVE) / model_id / f"{cond['v2']}.jsonl"
            )
            raw = {f"{name}:{k}": v for k, v in local_run.answers.items()}
        elif cond.get("v2"):
            # exactly the named parts, never a glob: a model dir can hold
            # other attempts (v3 Sonnet's *_nopillow) that are not this row
            model_dir = cond.get("pred_root", run_dir) / cond["v2"] / model_id
            if not model_dir.is_dir():
                print(f"  {model_id}: 予測ファイルがない → スキップ")
                continue
            answers = load_agent_run_parts(model_dir, cond.get("pred_parts", OFFICIAL_PARTS))
            raw = {f"{name}:{k}": v for k, v in answers.items()}
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
                cond["rescale"]
                if "rescale" in cond
                else PREDICTION_RESCALE_V2_CALIBRATED
                if cond.get("v2") == "calibrated"
                else {}
                if cond.get("v2")
                else PREDICTION_RESCALE
            )
            if cond.get("pixel_answer"):
                k = key[t["id"]]
                answer = values_from_pixel_answer(
                    answer if isinstance(answer, list) else [],
                    tick_cal[(k["paper_id"], k["figure_id"])],
                    image_sizes[t["id"]],
                    cond["pixel_answer"],
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
            # design 7.82: the ground truth is now in each figure's printed
            # space; every answer so far followed the old rules (10^tick on a
            # log10-printed axis, kelvin on a degC axis) and is mapped back
            ops = PRINTED_SPACE_MIGRATION.get(key[t["id"]]["figure_id"])
            if ops and cond.get("answer_space") == "auto":
                pr = reg[key[t["id"]]["figure_id"]]
                answer, was_old = answer_to_printed_if_old(
                    answer if isinstance(answer, list) else [],
                    ops,
                    {"x": tuple(pr.x_range), "y": tuple(pr.y_range)},
                )
                print(f"  {key[t['id']]['figure_id']}: answer space "
                      + ("old rule -> converted" if was_old else "printed"))
            elif ops and not cond.get("answers_in_printed_space"):
                answer = answer_to_printed(answer if isinstance(answer, list) else [], ops)
            preds[t["id"]] = parse_curves(answer, reg[key[t["id"]]["figure_id"]].x_scale)
        results = evaluate_model_on_dataset(
            ReplayRunner(preds, order), items, matcher_for=matcher_for_task
        )
        per_figure = [figure_result_row(r) for r in results]
        payload = {
            "model_id": model_id + cond["suffix"],
            "model_name": model_name + cond["name_suffix"],
            # where the model ran (design §7.69): the deployment question --
            # a local row needs no data to leave the machine
            "execution": "local" if cond.get("local") else "cloud",
            "dataset_version": (
                f"v0-eval-pilot-n{len(reg_scoreable)}{gt_rev}-llm-subset-"
                f"n{len(items)}-{cond['subset_tag']}"
                if cond.get("subset_tag")
                else f"v0-eval-pilot-n{len(reg_scoreable)}{gt_rev}"
                + ("-noaxis" if cond["v2"] == "noaxis" else "")
                + DIAGNOSTIC_VERSION_SUFFIX
                if cond.get("diagnostic")
                else f"v0-eval-pilot-n{len(reg_scoreable)}{gt_rev}-"
                f"{cond.get('version_tag', cond['v2'])}"
                if cond.get("version_tag", cond.get("v2")) in ("noaxis", "pixcal")
                # every scoreable figure, same as the CV/LineFormer rows, so all
                # of them rank in one table (design 7.66)
                else f"v0-eval-pilot-n{len(reg_scoreable)}{gt_rev}"
                if len(parts) > 1 or cond.get("v2")
                else f"v0-eval-pilot-n{len(reg_scoreable)}{gt_rev}-llm-subset-"
                f"n{len(items)}{cond['suffix']}"),
            "run_at": datetime.now(UTC).isoformat(),
            "n_figures": len(per_figure),
            "metric": "normalized-y-distance, span floor 5% of linear y axis (design 7.66)",
            "point_metric": POINT_METRIC_LABEL,
            "mean_summary_score": sum(p["summary_score"] for p in per_figure) / len(per_figure),
            # design 7.67: the primary metric; summary_score is the reference
            "point_metrics": aggregate_point_metrics(per_figure, PRIMARY_POINT_TAU),
            # design 7.72: the figures too dense for point matching, on curve distance
            "dense_marker_metrics": aggregate_dense_marker_metrics(per_figure),
            "per_figure": per_figure,
            "agent_effort": EFFORT.get(model_id) if name == "calibrated" else None,
            "condition": cond["label"],
            # which saved prompt the row answered (scripts/eval/llm_run_<v>_prompt.md)
            **({"prompt": f"scripts/eval/llm_run_{cond['prompt']}_prompt.md"}
               if cond.get("prompt") else {}),
            # a local row's single-shot adaptation of that prompt
            **({"prompt": "scripts/eval/local_vlm/prompt_v3/single_shot_prompt.md"}
               if cond.get("local_prompt") == "v3" else {}),
            # a side measurement, left out of the leaderboard (build_leaderboard)
            **({"diagnostic": True} if cond.get("diagnostic") else {}),
            **(
                {"local_run": _local_run_block(
                    model_id,
                    cond["v2"],
                    local_run,
                    v2_key,
                    scoreable_ids,
                    archive=cond.get("local_archive", LOCAL_V2_ARCHIVE),
                    sequential=cond.get("sequential", False),
                )}
                if local_run
                else {}
            ),
            "notes": cond["notes"]
            if "notes" in cond
            else V3_NOPILLOW_NOTES
            if cond.get("diagnostic")
            else V3_NOTES
            if cond.get("prompt") == "v3"
            else LOCAL_CUDA_NOTES
            if cond.get("cuda")
            else LOCAL_V3_NOTES
            if cond.get("local_prompt") == "v3"
            else LOCAL_V2_NOTES
            if cond.get("local")
            else V2_NOTES
            if cond.get("v2")
            else (
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
