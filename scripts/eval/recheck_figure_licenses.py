"""データセットに入っている図のライセンスを、いま一度確認する。

**ライセンスは後から変わる。** 収集時に CC BY だった論文が、数年後に別の表示に
なっていることがある。本リポジトリは図のクロップ(= 改変物)を再配布しているので、
ND(改変禁止)が付いた論文の図は配れなくなる。

2026-10-07 に全件確認したところ、採点対象 30 論文のうち 1 本が該当した
(論文 446、`10.1515/amm-2015-0104`。収集時 cc-by → 現在 cc-by-nc-nd)。
docs/experiments/2026-10-07-license-rescreen.md に記録してある。

このスクリプトは**判定するだけで、何も消さない**。除外は司令塔の判断である
(CLAUDE.md「疑わしいものは司令塔に確認」)。

使い方:
    python3 scripts/eval/recheck_figure_licenses.py            # 採点対象だけ
    python3 scripts/eval/recheck_figure_licenses.py --all      # レジストリ全件
    python3 scripts/eval/recheck_figure_licenses.py --json out.json
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
import urllib.parse
import urllib.request

sys.path.insert(0, "src")

from real_chart_bench.adapter.verified_pairing_registry import load_registry
from real_chart_bench.usecase.real_image_gate import select_verified_pairings

REPO = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = REPO / "data/verified_pairs/registry.json"
ATTRIBUTION = REPO / "data/verified_pairs/ATTRIBUTION.md"
EMAIL = "tomoya.matou@gmail.com"

# 再配布してよいライセンス。クロップは改変物なので ND は入らない。
ALLOWED = ("cc-by", "cc-by-sa", "public-domain")
# 改変を禁じるもの。見つけたら必ず報告する。
NO_DERIVATIVES = "nd"

ATTRIBUTION_ROW = re.compile(r"\| \[(10\.[^\]]+)\]\([^)]+\) \| ([^|]+) \| `([^`]+)` \| (\w+)")


def attribution_rows() -> dict[str, tuple[str, str, str]]:
    """paper_id -> (DOI, 記録されたライセンス, 改変したか)。"""
    out: dict[str, tuple[str, str, str]] = {}
    for line in ATTRIBUTION.read_text().splitlines():
        match = ATTRIBUTION_ROW.match(line)
        if not match:
            continue
        doi, recorded, path, modified = match.groups()
        parts = pathlib.Path(path).parts
        if len(parts) >= 4:
            out[parts[3]] = (doi, recorded.strip(), modified)
    return out


def current_license(doi: str) -> str:
    """Unpaywall がいま返すライセンス。閉じていれば "closed"。"""
    url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(doi)}?email={EMAIL}"
    request = urllib.request.Request(url, headers={"User-Agent": "real-chart-bench/0.1"})
    try:
        data = json.load(urllib.request.urlopen(request, timeout=30))  # noqa: S310
    except Exception as exc:
        return f"error:{type(exc).__name__}"
    if not data.get("is_oa"):
        return "closed"
    return (data.get("best_oa_location") or {}).get("license") or "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--all", action="store_true", help="採点対象以外も見る")
    parser.add_argument("--json", type=pathlib.Path, help="結果の書き出し先")
    args = parser.parse_args()

    registry = load_registry(REGISTRY)
    pairings = registry if args.all else select_verified_pairings(registry)
    by_paper: dict[str, list] = {}
    for pairing in pairings:
        by_paper.setdefault(pairing.paper_id, []).append(pairing)
    rows = attribution_rows()

    print(f"{'論文':<8}{'DOI':<34}{'記録':<10}{'いま':<14}{'改変':<6}{'図'}")
    findings = []
    for paper_id in sorted(by_paper):
        doi, recorded, modified = rows.get(paper_id, ("", "?", "?"))
        now = current_license(doi) if doi else "?"
        if doi:
            time.sleep(0.15)
        figures = by_paper[paper_id]
        forbids_derivatives = NO_DERIVATIVES in now.split("-")
        off_allowlist = now not in ALLOWED and not now.startswith(("error:", "?"))
        mark = ""
        if off_allowlist:
            blocked = forbids_derivatives and modified == "yes"
            mark = "  <<< 改変物を配れない" if blocked else "  <<<"
            findings.append(
                {
                    "paper_id": paper_id,
                    "doi": doi,
                    "recorded": recorded,
                    "current": now,
                    "modified": modified,
                    "forbids_derivatives": forbids_derivatives,
                    "figure_ids": [p.figure_id for p in figures],
                }
            )
        print(f"{paper_id:<8}{doi:<34}{recorded:<10}{now:<14}{modified:<6}{len(figures)}{mark}")

    total_figures = sum(len(v) for v in by_paper.values())
    affected = sum(len(f["figure_ids"]) for f in findings)
    print(f"\n許容({' / '.join(ALLOWED)})から外れた論文: {len(findings)} / {len(by_paper)}")
    print(f"影響する図: {affected} / {total_figures}")
    for f in findings:
        note = "ND あり → 改変物の再配布は不可" if f["forbids_derivatives"] else "ND なし"
        print(
            f"  論文 {f['paper_id']} ({f['doi']}): {f['recorded']} → {f['current']}、"
            f"改変={f['modified']}、{len(f['figure_ids'])}図。{note}"
        )
    if findings:
        print("\n**除外は司令塔の判断。このスクリプトは何も消さない。**")
    if args.json:
        args.json.write_text(json.dumps(findings, ensure_ascii=False, indent=2) + "\n")
        print(f"\n書き出し: {args.json}")


if __name__ == "__main__":
    main()
