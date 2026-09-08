"""Apply Fable's independent axis re-measurement where it disagrees with ours.

The owner flagged 20 of 24 reviewed axis readings as positionally wrong and
asked for Fable specifically to re-check them -- a sound pick, since in the
no-axis run Fable was the only model to handle a reciprocal 1000/T axis and the
only one to build a tick-detection pipeline rather than read by eye.

It measured all 21 blind: given the tick *values* but never our recorded pixel
positions, so agreement is evidence rather than an echo. It located each tick as
a darkness-weighted centroid of the perpendicular stroke, cross-checked every
axis for uniform spacing and against the mirror strokes on the opposite spine,
and inspected 8x crops of all 84 positions.

Two things came out of the comparison:

  - the owner's eye was right every time, including the extremes. He called
    44283's x axis "50px左", Fable measures 66.2; he called 18869/4(b) "右に
    3ピクセルくらい", Fable measures 3.9.

  - this morning's y-bottom snap holds up: y_min now agrees within 0.3 px on
    all 21 figures, independently confirmed.

So what remains is mostly x, plus y_max on a few. Anything within 1 px is left
alone -- below that the two methods are measuring the same stroke and churning
the data would only lose the audit trail.

`owner_reviewed` entries are never touched: a human has already accepted those,
and a machine measurement does not get to overrule that silently.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
AXP = REPO / "data/verified_pairs/axis_pixel_candidates.json"
# Which measurement batch to apply. The first covered the 21 figures the owner
# had flagged; the second, every remaining llm_candidate reading.
BATCHES = {
    "1": "fable_axis",
    "2": "fable_axis2",
}
# Positions only. Where a figure needs its tick *values* changed the fix is not
# mechanical -- it changes what the axis means -- so those are excluded here and
# put to the owner instead. 45906/45323 is the one such case in batch 2: the
# recorded pixels are the frame, but the values attached to them (-2.4, 3.3) are
# neither the frame's (-2.5, 3.5) nor the printed ticks' (-2, 3).
VALUE_QUESTIONS = {"45323"}
MIN_DIFF = 1.0
KEYS = ["x_min_px", "x_max_px", "y_min_px", "y_max_px"]


def main() -> None:
    apply = "--apply" in sys.argv
    batch = next((a for a in sys.argv[1:] if a in BATCHES), "1")
    WORK = pathlib.Path(
        "/tmp/claude-1000/-home-mato-repos-real-chart-bench/"
        "d1d6d9ff-97be-43f5-ae89-32fbeb7fe7d9/scratchpad/" + BATCHES[batch]
    )
    data = json.loads(AXP.read_text(), object_pairs_hook=collections.OrderedDict)
    by_id = {a["figure_id"]: a for a in data if "figure_id" in a}
    key = json.loads((WORK / "_key.json").read_text())
    fab = json.loads((WORK / "measured.json").read_text())

    changed, skipped = [], collections.Counter()
    for name, meas in sorted(fab.items()):
        k = key.get(name)
        if not k:
            continue
        a = by_id.get(k["figure_id"])
        if a is None:
            skipped["登録なし"] += 1
            continue
        if a.get("status") == "owner_reviewed":
            skipped["人が確認済み"] += 1
            continue
        if k["figure_id"] in VALUE_QUESTIONS:
            skipped["値の確認が必要（オーナーへ）"] += 1
            continue
        bb = a.get("pixel_bbox_mean") or {}
        edits = []
        for kk in KEYS:
            old, new = bb.get(kk), meas.get(kk)
            if old is None or new is None:
                continue
            if abs(new - old) < MIN_DIFF:
                continue
            edits.append((kk, old, round(float(new), 1)))
        if not edits:
            skipped["1px未満で一致"] += 1
            continue
        changed.append((k["paper_id"], k["figure_id"], meas.get("confidence"), edits))
        if apply:
            for kk, _old, new in edits:
                bb[kk] = new
            detail = "、".join(f"{kk} {o}→{n}" for kk, o, n in edits)
            a["notes"] = [n for n in a.get("notes", []) if n] + [
                f"2026-09-08 訂正({detail}): オーナーが軸位置の誤りを指摘した21図について、"
                f"Claude Fable 5 に記録値を伏せて独立に再測定させた結果を採用。"
                f"目盛は垂直ストロークの濃度重み付き重心として求め、目盛間隔の等間隔性と"
                f"反対側スパインのミラー目盛で検証済み(確度: {meas.get('confidence')})。"
                + (f" 測定側の注記: {meas.get('note')}" if meas.get("note") else ""),
            ]

    if apply:
        AXP.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    print(f"{'paper/figure':<16}{'確度':<8}変更")
    for pid, fid, conf, edits in changed:
        detail = ", ".join(
            f"{kk.replace('_px', '')} {o:.1f}→{n:.1f} ({n - o:+.1f})" for kk, o, n in edits)
        print(f"{pid + '/' + fid:<16}{str(conf):<8}{detail}")
    print(f"\n変更 {len(changed)} 図 / 参照点 {sum(len(e[3]) for e in changed)} 箇所")
    print("対象外:", dict(skipped))
    print("\n(--apply なしのため書き込んでいない)" if not apply else "\n書き込み済み")


if __name__ == "__main__":
    main()
