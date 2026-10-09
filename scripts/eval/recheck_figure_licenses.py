"""データセットに入っている図のライセンスを、いま一度確認する。

**ライセンスは後から変わる。** 収集時に CC BY だった論文が、数年後に別の表示に
なっていることがある。本リポジトリは図のクロップ(= 改変物)を再配布しているので、
ND(改変禁止)が付いた論文の図は配れなくなる。

2026-10-07 に全件確認したところ、採点対象 30 論文のうち 1 本が該当した
(論文 446、`10.1515/amm-2015-0104`。収集時 cc-by → 現在 cc-by-nc-nd)。
docs/experiments/2026-10-07-license-rescreen.md に記録してある。

このスクリプトは**判定するだけで、何も消さない**。除外は司令塔の判断である
(CLAUDE.md「疑わしいものは司令塔に確認」)。

`nc` サブセット(design §7.88.1)も同じように見る。判定は domain の `license_drift`:

- `exclude`     — ND が付いた(NC → NC-ND を含む)か closed になった。除外候補
- `subset_move` — core ↔ nc(例: CC BY → CC BY-NC)。サブセット移動の候補。**自動では動かさない**
- `review`      — ライセンスが空・未知になった。人が見る
- `lookup_error`— 問い合わせ失敗

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
from real_chart_bench.domain.dataset_subset import DatasetSubset, distribution_dir_name
from real_chart_bench.domain.licensing import DriftKind, license_drift
from real_chart_bench.usecase.real_image_gate import select_verified_pairings

REPO = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = REPO / "data/verified_pairs/registry.json"
ATTRIBUTIONS = tuple(
    REPO / "data" / distribution_dir_name(subset) / "ATTRIBUTION.md" for subset in DatasetSubset
)
EMAIL = "tomoya.matou@gmail.com"

_NOTE = {
    DriftKind.EXCLUDE: "ND あり / closed → 改変物の再配布は不可。除外候補",
    DriftKind.SUBSET_MOVE: "サブセット移動の候補(core ↔ nc)。自動では動かさない",
    DriftKind.REVIEW: "ライセンスが空・未知。人が確認",
}

ATTRIBUTION_ROW = re.compile(r"\| \[(10\.[^\]]+)\]\([^)]+\) \| ([^|]+) \| `([^`]+)` \| (\w+)")


def attribution_rows() -> dict[str, tuple[str, str, str]]:
    """paper_id -> (DOI, 記録されたライセンス, 改変したか)。"""
    out: dict[str, tuple[str, str, str]] = {}
    lines = [
        line for path in ATTRIBUTIONS if path.exists() for line in path.read_text().splitlines()
    ]
    for line in lines:
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
    pairings = (
        registry
        if args.all
        else [p for s in DatasetSubset for p in select_verified_pairings(registry, subset=s)]
    )
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
        drift = license_drift(recorded, now)
        mark = ""
        if drift.kind in _NOTE:
            mark = f"  <<< {drift.kind.value}"
            findings.append(
                {
                    "paper_id": paper_id,
                    "doi": doi,
                    "recorded": recorded,
                    "current": now,
                    "modified": modified,
                    "drift": drift.kind.value,
                    "recorded_subset": drift.recorded_subset and drift.recorded_subset.value,
                    "current_subset": drift.current_subset and drift.current_subset.value,
                    "figure_ids": [p.figure_id for p in figures],
                }
            )
        print(f"{paper_id:<8}{doi:<34}{recorded:<10}{now:<14}{modified:<6}{len(figures)}{mark}")

    total_figures = sum(len(v) for v in by_paper.values())
    affected = sum(len(f["figure_ids"]) for f in findings)
    print(f"\n記録時のサブセットから外れた論文: {len(findings)} / {len(by_paper)}")
    print(f"影響する図: {affected} / {total_figures}")
    for f in findings:
        note = _NOTE[DriftKind(f["drift"])]
        print(
            f"  論文 {f['paper_id']} ({f['doi']}): {f['recorded']} → {f['current']}、"
            f"改変={f['modified']}、{len(f['figure_ids'])}図。{note}"
        )
    if findings:
        print("\n**除外・サブセット移動は司令塔の判断。このスクリプトは何も変えない。**")
    if args.json:
        args.json.write_text(json.dumps(findings, ensure_ascii=False, indent=2) + "\n")
        print(f"\n書き出し: {args.json}")


if __name__ == "__main__":
    main()
