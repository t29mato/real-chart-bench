"""Records the one new suspicion raised by the 2026-09-09 alignment sweep
(``scripts/eval/sweep_gt_image_alignment.py``) into registry.json.

44283/39578 (FIGURE5(B), rho vs T for Sr(1-x)Ba(x)Si2) is the only figure the
sweep flags that is not already accounted for by an existing excluded_reason,
a confirmed GT problem, or a printed-axis/GT unit-space difference. Its GT
points score *worse* against the drawn ink than a deliberately wrong
placement (d_true 3.16 px vs d_null 2.70 px), and on a 3x crop three of the
four GT curves visibly follow no drawn series -- only the x=0.00 series lines
up. The registry evidence for this entry checked ranges ("x-range 7.6-299.4K
matches the chart's 0-300K axis") and magnitude ordering, both of which a
within-range distortion passes.

Per design §7.48 this is recorded as ``llm_flagged`` and NOT as a GT error:
the finding comes from an automated check plus a model's visual read, and
only a human may promote it to ``human_confirmed``. Note also that this
entry's axis reading is still ``llm_candidate``, so a wrong axis reading is
not excluded as the cause -- though the printed y ticks (0/3/6/9/12/15) were
checked against the recorded labels by eye and agree. Scoring is deliberately
NOT changed here (no excluded_reason): that is an owner decision.

Run: python scripts/eval/record_alignment_sweep_2026_09_09.py
"""

from __future__ import annotations

import json
from pathlib import Path

REGISTRY = Path(__file__).resolve().parents[2] / "data" / "verified_pairs" / "registry.json"
TARGET = ("44283", "39578")

NOTE = (
    " [2026-09-09 ALIGNMENT SWEEP: flagged as gt_suspect/llm_flagged. Mapping this "
    "entry's GT through the recorded axis reading places the points a median 3.16 px "
    "from the nearest ink, WORSE than the 2.70 px a deliberately wrong placement "
    "(2% diagonal shift) scores on the same image; every other non-excluded figure in "
    "the sweep scores well below its own wrong-placement baseline. On a 3x crop of the "
    "low-T region, three of the four GT curves follow no drawn series -- only x=0.00 "
    "lines up. The range/ordering checks in the evidence above all still hold, which is "
    "the point: they cannot see a distortion that stays inside the right range. "
    "NOT a confirmed GT error (design §7.48 -- only a human promotes to human_confirmed), "
    "and this entry's axis reading is itself still llm_candidate, so a wrong axis reading "
    "is not ruled out; the printed y ticks 0/3/6/9/12/15 do match the recorded labels on "
    "inspection. Scoring unchanged pending owner review. "
    "See docs/experiments/2026-09-09-gt-image-alignment-sweep.md.]"
)


def main() -> None:
    registry = json.loads(REGISTRY.read_text())
    for entry in registry:
        if (entry["paper_id"], entry["figure_id"]) != TARGET:
            continue
        if entry.get("gt_suspect_status") is not None:
            print(f"{TARGET} already flagged ({entry['gt_suspect_status']}); leaving as is")
            return
        entry["rejection_category"] = "gt_suspect"
        entry["gt_suspect_status"] = "llm_flagged"
        entry["rejection_evidence"] = {
            # The ranges DO agree -- that is why verification passed.
            "axis_range_mismatch": False,
            "point_count_mismatch": None,
            "y_value_offset_magnitude": None,
            "missing_series": True,
        }
        entry["evidence"] = entry["evidence"] + NOTE
        REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n")
        print(f"flagged {TARGET[0]}/{TARGET[1]} as gt_suspect/llm_flagged")
        return
    raise SystemExit(f"{TARGET} not found in registry")


if __name__ == "__main__":
    main()
