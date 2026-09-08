"""Snap every recorded y-axis minimum onto the tick mark it was meant to mark.

The owner's review of 24 unreviewed axis readings found 20 wrong, and almost
every note said the same thing: "y◯◯が上に2pxくらいずれてる". Magnifying
28331/28500 showed why -- the recorded position sits on the top edge of the
"5.0" label's glyphs, about 4 px above the tick stroke. Both vision models made
the same mistake on the bottom y reference, and only on that one: measured
against detected tick marks, y_min_px is off by a median of -2.75 px while
y_max_px, x_min_px and x_max_px all sit at 0.00.

A constant +2.75 would be the wrong repair -- the per-figure error runs from
-2.0 to -4.0 -- so each entry is snapped to its own nearest tick instead.

Guards, because a bad "fix" applied across a hundred entries is worse than the
bug:
  - only y_min_px is touched; the other three references measure clean
  - a candidate must lie within MAX_SNAP px, so a missing tick cannot drag the
    reference onto a distant unrelated stroke
  - the snap must not cross y_max_px, and must leave the axis span intact
  - anything already within TOLERANCE is left alone
  - owner_reviewed entries are skipped: a human has already looked at those

Nothing is written without --apply; the default is a dry run.
"""
from __future__ import annotations

import collections
import json
import pathlib
import sys

import numpy as np
from PIL import Image

REPO = pathlib.Path(__file__).resolve().parents[2]
AXP = REPO / "data/verified_pairs/axis_pixel_candidates.json"
REG = REPO / "data/verified_pairs/registry.json"

MAX_SNAP = 8.0      # px; beyond this the nearest stroke is a different tick
TOLERANCE = 1.0     # px; already correct


def runs(idx, gap=2):
    out = []
    for i in idx:
        if out and i - out[-1][-1] <= gap:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def y_ticks(dark):
    """Tick stroke centres along whichever vertical spine carries them."""
    H, W = dark.shape
    cols = dark.sum(0)
    spines = [float(np.mean(g)) for g in runs([j for j in range(W) if cols[j] > H * 0.35])]
    found = []
    for sp in spines:
        s = int(round(sp))
        for lo, hi in ((s + 3, s + 17), (s - 17, s - 3)):
            lo, hi = max(0, lo), min(W, hi)
            if hi - lo < 4:
                continue
            prof = dark[:, lo:hi].sum(1)
            idx = np.flatnonzero(prof >= (hi - lo) * 0.7)
            found += [float(np.mean(g)) for g in runs(idx) if len(g) <= 8]
    return sorted(set(found))


def main() -> None:
    apply = "--apply" in sys.argv
    data = json.loads(AXP.read_text(), object_pairs_hook=collections.OrderedDict)
    reg = {e["figure_id"]: e for e in json.loads(REG.read_text())}

    moved, skipped = [], collections.Counter()
    for a in data:
        if "_meta" in a:
            continue
        fid = a.get("figure_id")
        if a.get("status") == "owner_reviewed":
            skipped["人が確認済み"] += 1
            continue
        bb = a.get("pixel_bbox_mean") or {}
        rec, top = bb.get("y_min_px"), bb.get("y_max_px")
        e = reg.get(fid)
        if rec is None or e is None:
            skipped["y下端または登録なし"] += 1
            continue
        path = REPO / e["image_path"]
        if not path.exists():
            skipped["画像なし"] += 1
            continue
        g = np.asarray(Image.open(path).convert("L")).astype(int)
        ticks = y_ticks(g < 128)
        if not ticks:
            skipped["目盛を検出できず"] += 1
            continue
        near = min(ticks, key=lambda t: abs(t - rec))
        d = near - rec
        if abs(d) <= TOLERANCE:
            skipped["既に一致"] += 1
            continue
        if abs(d) > MAX_SNAP:
            skipped["最寄り目盛が遠すぎる"] += 1
            continue
        if top is not None and (near - top) * (rec - top) <= 0:
            skipped["上端をまたぐ"] += 1
            continue
        moved.append((fid, e["paper_id"], rec, round(near, 1), d))
        if apply:
            bb["y_min_px"] = round(near, 1)
            a["notes"] = [n for n in a.get("notes", []) if n] + [
                f"2026-09-08 訂正: y軸下端の画素位置を {rec} → {round(near, 1)} "
                f"({d:+.2f}px)。記録値は目盛ではなくラベル文字の上端に乗っていた"
                f"(オーナーのレビューで24図中20図に同じずれ、y下端のみ中央値-2.75px、"
                f"他3参照点は0.00px)。検出した目盛位置へスナップした。",
            ]

    if apply:
        AXP.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")

    print(f"{'figure':<8}{'paper':<8}{'記録値':>9}{'目盛':>9}{'補正':>8}")
    for fid, pid, rec, near, d in sorted(moved, key=lambda r: r[4]):
        print(f"{fid:<8}{pid:<8}{rec:9.1f}{near:9.1f}{d:+8.2f}")
    if moved:
        ds = [m[4] for m in moved]
        print(f"\n補正 {len(moved)} 件  中央値 {np.median(ds):+.2f}px  "
              f"範囲 {min(ds):+.2f}〜{max(ds):+.2f}px")
    print("対象外:", dict(skipped))
    print("\n(--apply なしのため書き込んでいない)" if not apply else "\n書き込み済み")


if __name__ == "__main__":
    main()
