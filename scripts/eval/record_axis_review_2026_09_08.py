"""Record the owner's sign-off on today's axis corrections.

He reviewed all 34 corrected figures and accepted 30. Those 30 do not all carry
the same weight of evidence, and flattening that distinction would quietly
overstate what has been checked:

  20 figures  every reference verified. These were in the set he reviewed
              yesterday, where he judged the *values* correct and only the
              positions wrong ("pos", never "val"); Fable then measured all
              four positions blind and he has now accepted the corrections. Both
              halves of the reading -- what value, at what pixel -- have been
              through a human. These are promoted to `owner_reviewed`.

  10 figures  only the y-axis minimum was corrected, and only that line was
              shown to him. Their tick *values* have never been reviewed by
              anyone. They stay `llm_candidate`: `promote_tick_range` gates on
              `owner_reviewed` (design 7.57), so promoting them would authorise
              publishing tick ranges nobody has checked. The confirmation is
              recorded in the notes instead.

The remaining 4 he could not judge -- the crops were magnified too far to tell
which tick the line was on -- and are being re-rendered wider rather than
guessed at.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
AXP = REPO / "data/verified_pairs/axis_pixel_candidates.json"
DEC = REPO / "data/verified_pairs/review_decisions.json"

FULLY_VERIFIED = ["1531", "1534", "1536", "1537", "1538", "18872", "18874", "18875",
                  "18876", "28492", "28495", "28498", "28499", "28500", "28501",
                  "28502", "28505", "34988", "34991", "34993"]
Y_MIN_ONLY = ["11780", "11782", "11784", "12225", "12226", "1530", "1532", "1533",
              "34994", "48871"]
UNSURE = ["38968", "11779", "11781", "11783"]

NOTE_FULL = (
    "2026-09-08 owner review: 本日の軸位置修正をオーナーが目視確認し承認。"
    "この図は2026-09-07のレビューで値(x_min_label等)が正しいと判定済み(verdict=pos、"
    "値の誤りではなく位置の誤りとの判定)であり、位置は Claude Fable 5 が記録値を伏せて"
    "4参照点すべてを独立測定し、その修正結果を本日承認した。"
    "値と位置の双方が人手を通ったため status を owner_reviewed へ昇格する。"
)
NOTE_YMIN = (
    "2026-09-08 owner review: y軸下端の位置修正のみをオーナーが目視確認し承認。"
    "この図で確認されたのは y_min_px の位置だけであり、目盛の値(x_min_label等)および"
    "他3参照点の位置は未レビューのため status は llm_candidate のまま据え置く。"
    "promote_tick_range は owner_reviewed を要求する(設計§7.57)ので、"
    "ここで昇格させると誰も確認していない tick range の公開を許してしまう。"
)
NOTE_UNSURE = (
    "2026-09-08 owner review: 判定保留。修正前後を並べた拡大図を提示したが、"
    "拡大率が高すぎてどの目盛を指しているか判別できないとのこと。"
    "縮小した図で再提示する。"
)


def main() -> None:
    apply = "--apply" in sys.argv
    data = json.loads(AXP.read_text(), object_pairs_hook=collections.OrderedDict)
    by_id = {a["figure_id"]: a for a in data if "figure_id" in a}

    counts = collections.Counter()
    for fids, note, promote in ((FULLY_VERIFIED, NOTE_FULL, True),
                                (Y_MIN_ONLY, NOTE_YMIN, False),
                                (UNSURE, NOTE_UNSURE, False)):
        for fid in fids:
            a = by_id.get(fid)
            if a is None:
                counts["登録なし"] += 1
                continue
            counts["昇格" if promote else "記録のみ"] += 1
            if apply:
                a["notes"] = [n for n in a.get("notes", []) if n] + [note]
                if promote:
                    a["status"] = "owner_reviewed"

    if apply:
        AXP.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
        dec = json.loads(DEC.read_text(), object_pairs_hook=collections.OrderedDict)
        for fid in FULLY_VERIFIED + Y_MIN_ONLY:
            dec["decisions"].append(collections.OrderedDict([
                ("reviewed_at", "2026-09-08T00:00:00Z"), ("reviewer", "owner"),
                ("item_id", f"AXISFIX-{fid}"), ("figure_id", fid),
                ("question", "本日の軸位置修正の確認"), ("verdict", "ok"),
                ("note", "修正でOK"),
                ("acted_on", ("status を owner_reviewed へ昇格" if fid in FULLY_VERIFIED
                              else "y下端のみの確認のため status は据え置き、notes に記録")),
            ]))
        for fid in UNSURE:
            dec["decisions"].append(collections.OrderedDict([
                ("reviewed_at", "2026-09-08T00:00:00Z"), ("reviewer", "owner"),
                ("item_id", f"AXISFIX-{fid}"), ("figure_id", fid),
                ("question", "本日の軸位置修正の確認"), ("verdict", "unsure"),
                ("note", "拡大率が高すぎて判別できない。縮小して再提示が必要"),
                ("acted_on", "縮小版で再提示予定"),
            ]))
        DEC.write_text(json.dumps(dec, ensure_ascii=False, indent=2) + "\n")

    st = collections.Counter(a.get("status") for a in data if "figure_id" in a)
    print("処理:", dict(counts))
    print("status 内訳:", dict(st))
    print("\n(--apply なしのため書き込んでいない)" if not apply else "\n書き込み済み")


if __name__ == "__main__":
    main()
