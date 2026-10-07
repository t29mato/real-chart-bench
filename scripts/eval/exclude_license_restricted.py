"""ライセンス上再配布できない図を採点対象から外す(2026-10-07 オーナー判断)。

論文 446(`10.1515/amm-2015-0104`)は収集時 CC BY と記録していたが、現在 Unpaywall /
OpenAlex とも **cc-by-nc-nd** を返す(出版版)。本リポジトリはこの図のクロップ
= **改変物**を再配布しており、ND はそれを禁じる。

オーナー判断は「A除外(n94→n91、全結果行が変わる)にしてください」。

このスクリプトがすること:

1. レジストリの該当エントリに `excluded_reason` を書く(接頭辞 `license_restricted:`)。
   既存の除外と同じ仕組みで、`select_verified_pairings` が自動的に落とす。
   `rejection_category` は設定しない — ペアリングも画像も正解データも正しく、
   欠陥があるのは我々の配布権限の側だからである。
2. **クロップ画像そのものを作業ツリーから削除する。** 採点から外すだけでは不十分で、
   改変物を配っている状態が続いてしまう。

**git の履歴には残る。** 公開前に履歴の扱いを決める必要がある(司令塔案件)。

実行: python3 scripts/eval/exclude_license_restricted.py [--apply]
"""

from __future__ import annotations

import argparse
import json
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
REGISTRY = REPO / "data/verified_pairs/registry.json"

PAPER_ID = "446"
FIGURE_IDS = ("8724", "8725", "8726")
REASON = (
    "license_restricted: 収集時は cc-by と記録していたが、2026-10-07 の再確認で "
    "Unpaywall / OpenAlex とも cc-by-nc-nd を返した(出版版、"
    "Archives of Metallurgy and Materials)。本リポジトリはこの図のクロップ"
    "(= 改変物)を再配布しており、ND は改変物の再配布を禁じる。"
    "オーナー判断「A除外(n94→n91、全結果行が変わる)にしてください」"
    "(2026-10-07)。図・正解データ・ペアリングはいずれも正しく、"
    "欠陥は我々の配布権限の側にあるので rejection_category は設定しない。"
    "クロップ画像は作業ツリーから削除済み。"
    "確認は scripts/eval/recheck_figure_licenses.py、"
    "記録は docs/experiments/2026-10-07-license-rescreen.md。"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="実際に書き換える")
    args = parser.parse_args()

    registry = json.loads(REGISTRY.read_text())
    entries = registry["pairings"] if isinstance(registry, dict) else registry

    touched, images = [], []
    for entry in entries:
        if entry.get("paper_id") != PAPER_ID or entry.get("figure_id") not in FIGURE_IDS:
            continue
        if entry.get("excluded_reason"):
            print(f"  {entry['figure_id']}: すでに除外済み。触らない")
            continue
        entry["excluded_reason"] = REASON
        touched.append(entry["figure_id"])
        image = REPO / entry["image_path"]
        if image.exists():
            images.append(image)

    print(f"除外する図: {touched or 'なし'}")
    for image in images:
        print(f"削除する画像: {image.relative_to(REPO)}")

    if not args.apply:
        print("\n--apply を付けると書き換える(いまは何もしていない)")
        return

    REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
    for image in images:
        image.unlink()
    print(f"\nレジストリ更新: {REGISTRY.relative_to(REPO)}")
    print("次に: generate_attribution.py → rescore_all.py → leaderboard/generate.py")


if __name__ == "__main__":
    main()
