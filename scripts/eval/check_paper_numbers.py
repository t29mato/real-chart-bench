"""論文 4.1 / 4.2 の表の数値が、いまの results/ と合っているか調べる。

データセットが変わるたびに再採点はするが、**論文の表は手書きのまま取り残される**。
2026-10-07 の n94 → n91 でまさにそれが起きた(表は n94 の値、本文の図数も 94 / 80 のまま)。

既定では**直さず、食い違いを並べるだけ**である。`--apply` を付けると表の**数値セルだけ**を
いまの値で置き換える(太字などの体裁と桁数はそのまま保つ)。本文の「94図」のような
言い回しは触らない — 文の意味まで機械が書き換えるべきではないので、報告だけする。

実行:
    PYTHONPATH=src python3 scripts/eval/check_paper_numbers.py
    PYTHONPATH=src python3 scripts/eval/check_paper_numbers.py --apply
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
PAPER = REPO / "docs/paper/04-results.md"

# 表の手法名 -> 結果ファイル(条件ごと)
MODELS: dict[str, tuple[str, str]] = {
    "Claude Opus 5.5": ("claude-opus-5-5-v0-r3-noaxis", "claude-opus-5-5-v0-pixcal"),
    "GPT-6.1-Sol": ("gpt-6.1-sol-v0-codex-noaxis", "gpt-6.1-sol-v0-codex-pixcal"),
    "Claude Fable 5.1": ("claude-fable-5-1-v0-r3-noaxis", "claude-fable-5-1-v0-pixcal"),
    "Claude Sonnet 5.5": ("claude-sonnet-5-5-v0-r3-noaxis", "claude-sonnet-5-5-v0-pixcal"),
    "Gemma 4 31B(8bit)": ("gemma-4-31b-8bit-v0-local-v3-noaxis", ""),
    "Qwen3.8-27B(8bit)": ("qwen3.8-27b-8bit-v0-local-v3-noaxis", ""),
    "Qwen3.8-27B Q4(Codex CLI + Ollama)": ("", "qwen3.8-27b-v0-codex-local-pixcal"),
    "Qwen3.5-9B(bf16)": (
        "qwen3.5-9b-bf16-v0-local-cuda-v3-noaxis",
        "qwen3.5-9b-bf16-v0-local-cuda-pixcal",
    ),
    "Qwen3.5-9B(8bit)": ("qwen3.5-9b-8bit-v0-local-v3-noaxis", ""),
    "GPT-5.5": ("gpt-5.5-v0-codex-noaxis", "gpt-5.5-v0-codex-pixcal"),
    "Granite Vision 4.1": ("granite-vision-noaxis", ""),
    "DePlot": ("deplot-noaxis", ""),
    "UniChart base-960": ("unichart-noaxis", ""),
    "TinyChart-3B-768": ("tinychart-noaxis", ""),
    "ChartGemma": ("chartgemma-noaxis", ""),
    "Claude Haiku 4.5": ("claude-haiku-4-5-v0-r3-noaxis", "claude-haiku-4-5-v0-pixcal"),
    "LineFormer(事前学習)+ 目盛校正": ("", "lineformer-pretrained-tickcal"),
    "Qwen3.5-9B(bf16)、2段階・0〜1000 座標": ("", "qwen3.5-9b-bf16-v0-local-cuda-pixpts-norm"),
    "Qwen3.5-9B(bf16)、2段階・画素座標": ("", "qwen3.5-9b-bf16-v0-local-cuda-pixpts-px"),
}

# 列の位置 -> 指標。表は 手法 | 実行 | 点F1 | 再現率 | 適合率 | 位置誤差 | 秒/図 | トークン/図
COLUMNS = {
    2: "point_f1",
    3: "point_recall",
    4: "point_precision",
    5: "point_loc_error",
    6: "seconds_per_figure",
    7: "tokens_per_figure",
}
ROW = re.compile(r"^\| ([^|]+?) \| ([^|]+?) \|(.+)$")


def per_figure_cost(data: dict) -> dict:
    """秒/図とトークン/図。

    どちらも `run_cost` の合計を**採点した図数**で割る(2026-10-09 から全行で同じ規則)。
    図数は結果ファイルの `n_figures`(なければ `per_figure` の長さ)。
    トークンは提供元ごとの数え方のまま: Codex は `tokens.total_tokens`、
    Claude Code は `subagent_tokens`、ローカルは `prompt_tokens + generation_tokens`。
    記録がない量は返さない(表の「記録なし」を推測で埋めない)。
    """
    rc = data.get("run_cost") or {}
    if not rc.get("recorded"):
        return {}
    n = data.get("n_figures") or len(data.get("per_figure") or [])
    if not n:
        return {}
    out = {}
    if rc.get("seconds_total") is not None:
        out["seconds_per_figure"] = rc["seconds_total"] / n
    tokens = rc.get("tokens")
    if isinstance(tokens, dict) and tokens.get("total_tokens") is not None:
        total = tokens["total_tokens"]
    elif rc.get("subagent_tokens") is not None:
        total = rc["subagent_tokens"]
    elif rc.get("prompt_tokens") is not None and rc.get("generation_tokens") is not None:
        total = rc["prompt_tokens"] + rc["generation_tokens"]
    else:
        total = None
    if total is not None:
        out["tokens_per_figure"] = total / n
    return out


def current(stem: str) -> dict | None:
    path = RESULTS / f"{stem}.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    macro = (data.get("point_metrics") or {}).get("by_tau", {}).get("0.02", {}).get("macro")
    if not macro:
        return None
    return {
        **macro,
        **per_figure_cost(data),
        "dataset_version": data.get("dataset_version", ""),
        "n_point_figures": (data.get("point_metrics") or {}).get("n_figures"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="表の数値セルを書き換える")
    args = parser.parse_args()

    text = PAPER.read_text()
    lines = text.splitlines(keepends=True)
    drift: list[str] = []
    seen: set[str] = set()
    condition = 0  # 0 = まだ表に入っていない、1 = 4.1、2 = 4.2

    for row_index, raw in enumerate(lines):
        line = raw.rstrip("\n")
        if line.startswith("## 4.1"):
            condition = 1
        elif line.startswith("## 4.2"):
            condition = 2
        elif line.startswith("## 4.3"):
            condition = 0
        if not condition:
            continue
        match = ROW.match(line)
        if not match:
            continue
        name = match.group(1).strip()
        stems = MODELS.get(name)
        if not stems:
            continue
        stem = stems[condition - 1]
        if not stem:
            continue
        where = "完全自動" if condition == 1 else "軸校正"
        values = current(stem)
        if values is None:
            drift.append(f"  {name}({where}): 結果ファイルがない")
            continue
        seen.add(stem)
        cells = line.split("|")
        changed = False
        for column, metric in COLUMNS.items():
            cell_index = column + 1  # split の先頭は空文字なので1つずれる
            if cell_index >= len(cells) - 1:
                continue
            cell = cells[cell_index]
            printed = cell.replace("**", "").strip()
            try:
                written = float(printed.replace(",", ""))
            except ValueError:
                continue
            if metric not in values:
                drift.append(f"  {name}({where}) {metric}: 論文 {printed} → 結果ファイルに記録なし")
                continue
            digits = len(printed.split(".")[1]) if "." in printed else 0
            now = round(values[metric], digits)
            if abs(written - now) < 10**-digits / 2:
                continue
            shown = (
                f"{now:,.{digits}f}"
                if "," in printed or metric == "tokens_per_figure"
                else f"{now:.{digits}f}"
            )
            drift.append(f"  {name}({where}) {metric}: 論文 {printed} → いま {shown}")
            cells[cell_index] = cell.replace(printed, shown)
            changed = True
        if changed and args.apply:
            lines[row_index] = "|".join(cells) + "\n"

    versions = {current(s)["dataset_version"] for s in seen if current(s)}
    print(f"照合した結果ファイル: {len(seen)}")
    print(f"dataset_version: {', '.join(sorted(versions)) or '—'}")
    counts = {current(s)["n_point_figures"] for s in seen if current(s)}
    print(f"点指標の図数: {sorted(c for c in counts if c)}")

    stale_counts = [
        f"  本文に「{n}図」と書いてあるが、点指標の図数は {sorted(counts)}"
        for n in ("94", "80")
        if f"{n}図" in text and str(sorted(counts)[0] if counts else "") != n
    ]
    print(f"\n食い違い: {len(drift)} 件")
    for line in drift:
        print(line)
    for line in stale_counts:
        print(line)
    if args.apply and drift:
        PAPER.write_text("".join(lines))
        print(f"\n{PAPER.relative_to(REPO)} の表を書き換えた({len(drift)} セル)")
        print("本文の言い回し(図数など)は触っていない。上の報告を見て直すこと。")
        return
    if drift or stale_counts:
        sys.exit(1)


if __name__ == "__main__":
    main()
