"""Exports human-confirmed Starrydata ground-truth issues to
data/gt_issues/ (design §7.48, 戦略メモ「柱G」).

Owner's plan (verbatim, from the 柱G strategy memo):

    `scripts/export/gt_issues.py` で、`human_confirmed` の一覧を Starrydata
    の識別子(paper ID / figure ID / property)付きで CSV/JSON に出力する。
    CC BY 4.0、`data/gt_issues/` に配置。ベンチ側の責務はここまで。
    Starrydata2 への反映・修正は職務側の作業として分離し、ベンチのコードから
    starrydata2.org に書き込みはしない。

This is a strictly one-way export: it only ever reads registry.json /
ground_truth.json / papers.json and writes into data/gt_issues/. Nothing
here makes any network call or writes to starrydata2.org.

CRITICAL invariant (owner rule, enforced in usecase/gt_issues.py via
VerifiedPairing.is_confirmed_gt_error, not re-checked here): only
``human_confirmed`` entries are exported. An ``llm_flagged`` GT_SUSPECT
entry is a suspicion, never a confirmed error -- it never appears in the
output, only in the summary's ``awaiting_human_review`` count.

Usage:
    python scripts/export/gt_issues.py
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.adapter.gt_issues_export import to_csv_text, to_json_export  # noqa: E402
from real_chart_bench.adapter.verified_pairing_registry import load_registry  # noqa: E402
from real_chart_bench.usecase.gt_issues import (  # noqa: E402
    select_confirmed_gt_issues,
    summarize_gt_suspect_review,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data/verified_pairs/registry.json"
GROUND_TRUTH_PATH = REPO_ROOT / "data/verified_pairs/ground_truth.json"
PAPERS_PATH = REPO_ROOT / "data/manifest/v0/papers.json"
OUT_DIR = REPO_ROOT / "data/gt_issues"
JSON_OUT_PATH = OUT_DIR / "gt_issues.json"
CSV_OUT_PATH = OUT_DIR / "gt_issues.csv"


def main() -> None:
    registry = load_registry(REGISTRY_PATH)
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text())
    papers_by_id = {p["paper_id"]: p for p in json.loads(PAPERS_PATH.read_text())}

    rows = select_confirmed_gt_issues(
        registry, ground_truth=ground_truth, papers_by_id=papers_by_id
    )
    summary = summarize_gt_suspect_review(registry)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    JSON_OUT_PATH.write_text(
        json.dumps(to_json_export(rows, summary), indent=2, ensure_ascii=False) + "\n"
    )
    CSV_OUT_PATH.write_text(to_csv_text(rows))

    print(f"gt_suspect entries in registry: {summary.total_gt_suspect}")
    print(
        f"human_confirmed (exported as GT issues): {summary.human_confirmed} "
        f"of {summary.total_gt_suspect}"
    )
    print(f"human_rejected (LLM was wrong, not exported): {summary.human_rejected}")
    print(
        f"awaiting human review (llm_flagged only, NOT exported -- a suspicion, "
        f"not a confirmed error): {summary.awaiting_human_review}"
    )
    if summary.human_confirmed == 0:
        print(
            "\nNo confirmed GT issues yet -- wrote a valid, empty export "
            f"({len(rows)} issue(s)) rather than fabricating placeholder rows."
        )
    print(f"\nwrote {JSON_OUT_PATH} and {CSV_OUT_PATH} ({len(rows)} confirmed GT issue(s))")


if __name__ == "__main__":
    main()
