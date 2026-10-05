"""Approach B of the local extractor (docs/design/local-model.md, 方式B):
a VLM fine-tuned to answer the v3 single-shot prompt directly.

One labels.jsonl line (domain.training_data) becomes the task entry the
prompt pastes after "Task:" and the answer the model should give, in the
same shape the benchmark scores: {"<id>": [{"label", "x", "y"}, ...]} with
values in the figure's printed space (design 7.82). The prompt text itself is
the benchmark's (scripts/eval/local_vlm/prompt_v3/), so training and scoring
cannot drift apart.

Two conditions:
  noaxis -- no calibration; the task carries only the report rules, here
            always the printed-space rule (labels hold printed values)
  pixcal -- the person-measured calibration of the pixcal row: two ticks per
            axis (pixel + value), the scales and the image size
"""

from __future__ import annotations

import hashlib
import json
import math

# The wording of the benchmark's noaxis tasks for an axis read as printed.
PRINTED_REPORT_RULE = (
    "Report numbers on the same scale as the printed tick labels. Do not apply a "
    "multiplier written in the axis title (e.g. '(10^4 S/m)' or 'x10^4'); if a tick "
    "label itself is written like 5x10^4, report 50000."
)
SIGNIFICANT_DIGITS = 4


def noaxis_task(fig_id: str) -> dict:
    return {"id": fig_id, "x_report": PRINTED_REPORT_RULE, "y_report": PRINTED_REPORT_RULE}


def _outer_ticks(axis: dict | None) -> list[dict] | None:
    ticks = [t for t in (axis or {}).get("ticks") or [] if t.get("px") is not None]
    if len(ticks) < 2:
        return None
    ordered = sorted(ticks, key=lambda t: t["px"])
    lo, hi = ordered[0], ordered[-1]
    if lo["px"] == hi["px"] or lo["value"] == hi["value"]:
        return None
    return [lo, hi]


def pixcal_task(fig_id: str, label: dict) -> dict | None:
    """The pixcal task entry, or None when the label has no usable calibration.

    The two outermost ticks per axis, x ticks left to right and y ticks
    bottom to top (largest pixel_y first), as a person clicks them; pixels
    to 0.1 px and values to four significant digits, as in the benchmark's
    calibration file."""
    axes = label.get("axes")
    if not axes:
        return None
    xt = _outer_ticks(axes.get("x"))
    yt = _outer_ticks(axes.get("y"))
    if xt is None or yt is None:
        return None
    yt = yt[::-1]
    return {
        "id": fig_id,
        "image_size": [label["width"], label["height"]],
        "x_scale": axes["x"]["scale"],
        "y_scale": axes["y"]["scale"],
        "x_ticks": [{"pixel_x": round(t["px"], 1), "value": round_sig(t["value"])} for t in xt],
        "y_ticks": [{"pixel_y": round(t["px"], 1), "value": round_sig(t["value"])} for t in yt],
    }


def round_sig(value: float, digits: int = SIGNIFICANT_DIGITS) -> int | float:
    """Round to `digits` significant digits; an integral result becomes an int
    so the answer reads 300, not 300.0 (fewer tokens, as a person types it)."""
    r = float(f"{value:.{digits}g}")
    if r.is_integer() and abs(r) < 1e15:
        return int(r)
    return r


def target_answer(fig_id: str, label: dict) -> dict | None:
    """The answer the model should give, or None when the label has no
    usable values (it is then not a training example for approach B)."""
    out = []
    for i, s in enumerate(label.get("series") or [], start=1):
        pts = s.get("points_value")
        if pts is None:
            return None
        if any(not (math.isfinite(x) and math.isfinite(y)) for x, y in pts):
            return None
        if not pts:
            continue
        pts = sorted(pts, key=lambda p: (p[0], p[1]))
        name = s.get("label") or f"series {i}"
        out.append(
            {
                "label": str(name),
                "x": [round_sig(p[0]) for p in pts],
                "y": [round_sig(p[1]) for p in pts],
            }
        )
    if not out:
        return None
    return {fig_id: out}


def format_answer(answer: dict) -> str:
    return json.dumps(answer, ensure_ascii=False)


def split_of(key: str, val_fraction: float) -> str:
    """'train' or 'val', fixed by a hash of `key` (an image path, or a
    paper_id so one paper's figures never straddle the split)."""
    h = int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big") / 2**64
    return "val" if h < val_fraction else "train"
