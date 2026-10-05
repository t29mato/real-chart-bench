"""One-shot migration (design 7.82, owner decision 2026-10-06): store the
ground truth of the 13 log10-printed y axes and the one degC-printed x axis
in the space the figure prints, so a single "report as printed" rule serves
every axis.

Rewrites, for every figure in printed_space.PRINTED_SPACE_MIGRATION:
  data/verified_pairs/ground_truth.json   -- the axis's values (and unit)
  data/verified_pairs/registry.json       -- x_range / y_range, the scale
                                             (log -> linear), a note
  data/verified_pairs/tick_calibration.json -- tick values, the scale

A registry entry that already carries `printed_space_migrated` is skipped,
so re-running is a no-op. Starrydata's own database is not touched.

Usage: python scripts/eval/migrate_ground_truth_to_printed_space.py
"""

from __future__ import annotations

import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.printed_space import (  # noqa: E402
    PRINTED_SPACE_MIGRATION,
    to_printed,
)

VP = REPO / "data/verified_pairs"
DATE = "2026-10-06"


def _conv(values, op):
    out = [to_printed(v, op) for v in values]
    if any(v is None for v in out):
        raise SystemExit(f"undefined printed value in {values}")
    return out


def main() -> None:
    reg = json.loads((VP / "registry.json").read_text())
    gt = json.loads((VP / "ground_truth.json").read_text())
    cal = json.loads((VP / "tick_calibration.json").read_text())
    cal_by = {c["figure_id"]: c for c in cal["figures"]}
    done = 0
    for e in reg:
        if not isinstance(e, dict) or e.get("figure_id") not in PRINTED_SPACE_MIGRATION:
            continue
        if e.get("printed_space_migrated"):
            continue
        fid = e["figure_id"]
        for axis, op in PRINTED_SPACE_MIGRATION[fid].items():
            e[f"{axis}_range"] = _conv(e[f"{axis}_range"], op)
            e[f"{axis}_scale"] = "linear"
            for c in gt[fid]:
                c[axis] = _conv(c[axis], op)
                if op == "log10":
                    # Starrydata's unit string need not be the display unit
                    # the values are in, so it is kept aside, not wrapped
                    c[f"unit_{axis}_before_printed_space"] = c.get(f"unit_{axis}")
                    c[f"unit_{axis}"] = "log10 of the quantity, as printed on the axis"
                else:
                    c[f"unit_{axis}"] = "°C"
            tc = cal_by[fid]
            tc[f"{axis}_scale"] = "linear"
            for t in tc[axis]:
                t["value"] = round(to_printed(t["value"], op), 9)
            tc["conversion"] = f"{axis}: value = printed (design 7.82)"
        e["printed_space_migrated"] = (
            f"{DATE}: {PRINTED_SPACE_MIGRATION[fid]} -- ground truth, ranges and ticks "
            "now in the figure's printed space (design 7.82)"
        )
        done += 1
    (VP / "registry.json").write_text(json.dumps(reg, indent=2, ensure_ascii=False) + "\n")
    (VP / "ground_truth.json").write_text(json.dumps(gt, indent=2, ensure_ascii=False) + "\n")
    (VP / "tick_calibration.json").write_text(json.dumps(cal, indent=1, ensure_ascii=False))
    print(f"migrated {done} figures")


if __name__ == "__main__":
    main()
