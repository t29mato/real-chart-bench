"""docs/paper/*.md を1枚の読み物ページに組む。

論文は章ごとの Markdown で管理している(docs/paper/)。読むときは通しで
読みたいので、章を順に並べて目次を付けた HTML を出す。Markdown の変換は
ページ側で marked.js が行うので、本文は原稿のまま埋め込む — 論文を直したら
もう一度このスクリプトを走らせるだけで済む。

実行: python3 scripts/publish/build_paper_page.py [出力先.html]
"""

from __future__ import annotations

import datetime
import html
import json
import pathlib
import re
import sys

PAPER_DIR = pathlib.Path("docs/paper")
DEFAULT_OUT = pathlib.Path("build/paper.html")
# 00-outline.md は編集方針のメモなので本文には入れない
SKIP = {"00-outline.md"}

TEMPLATE_PATH = pathlib.Path(__file__).with_name("paper_page_template.html")


def chapter_title(markdown: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+)$", markdown, flags=re.MULTILINE)
    return match.group(1).strip() if match else fallback


def dataset_facts() -> dict[str, str]:
    """見出しに出す規模。正解データから数えるので、原稿と食い違わない。"""
    path = pathlib.Path("data/verified_pairs/ground_truth.json")
    if not path.exists():
        return {"n_fig": "94", "n_curve": "363", "n_pt": "7,223"}
    data = json.loads(path.read_text())
    curves = [c for cs in data.values() for c in cs if c.get("x")]
    figures = [f for f, cs in data.items() if any(c.get("x") for c in cs)]
    return {
        "n_fig": f"{len(figures):,}",
        "n_curve": f"{len(curves):,}",
        "n_pt": f"{sum(len(c['x']) for c in curves):,}",
    }


def main() -> None:
    out = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT
    sources = sorted(p for p in PAPER_DIR.glob("*.md") if p.name not in SKIP)
    if not sources:
        raise SystemExit(f"no chapters under {PAPER_DIR}")
    chapters = [
        {"title": chapter_title(p.read_text(), p.stem), "markdown": p.read_text()} for p in sources
    ]
    page = TEMPLATE_PATH.read_text().format(
        built=f"組版 {datetime.date.today().isoformat()} · 全 {len(chapters)} 章",
        chapters=json.dumps(chapters, ensure_ascii=False),
        **{k: html.escape(v) for k, v in dataset_facts().items()},
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(f"{out}  ({out.stat().st_size:,} bytes, {len(chapters)} 章)")
    for c in chapters:
        print(f"  {c['title']}")


if __name__ == "__main__":
    main()
