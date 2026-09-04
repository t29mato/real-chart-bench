"""One-off migration (design §7.57): promotes printed tick-label axis
ranges from data/verified_pairs/axis_pixel_candidates.json into
data/verified_pairs/registry.json as x_tick_range/y_tick_range, for the
subset of entries whose axis reading has been human-reviewed
(status == "owner_reviewed" there).

Why: registry.json's x_range/y_range record the drawn axis FRAME extent,
not the printed tick values -- necessary because GT data routinely lies
outside the outermost printed tick. But the printed ticks are also ground
truth (the only axis-reading ground truth a model reading the chart could
ever produce), and until now they lived only in axis_pixel_candidates.json
-- a file whose own schema calls 73/111 of its entries "candidate"
(unreviewed LLM output). See domain/verified_pairing.py's
TickRangeProvenance/promote_tick_range and docs/design/benchmark-architecture.md
§7.57 for the full rationale.

Deliberately promotes ONLY "owner_reviewed" entries -- mirrors design
§7.48's llm_flagged/human_confirmed discipline. The remaining llm_candidate
entries are left untouched in axis_pixel_candidates.json; they are not
verified data.

Follows the same "mutate the raw registry dicts in place, preserve key
order, run through the domain model as a correctness check before writing"
convention as convert_ground_truth_to_display_units.py (design §7.47) --
run against real data this way rather than through
adapter.serialize_entry, so the diff on registry.json is purely additive
and every other field is untouched byte-for-byte.
"""

from __future__ import annotations

import json
from pathlib import Path

from real_chart_bench.adapter.verified_pairing_registry import parse_registry, serialize_entry
from real_chart_bench.domain.verified_pairing import promote_tick_range

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data/verified_pairs/registry.json"
AXIS_PIXEL_CANDIDATES_PATH = REPO_ROOT / "data/verified_pairs/axis_pixel_candidates.json"


def _key(entry: dict) -> tuple[str, str, str | None]:
    return (entry["paper_id"], entry["figure_id"], entry.get("panel_label"))


def main() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    axp_raw = json.loads(AXIS_PIXEL_CANDIDATES_PATH.read_text())
    axp_by_key = {_key(a): a for a in axp_raw if "_meta" not in a}

    promoted = []
    diverging_axes = 0

    for entry in registry:
        if entry.get("status") != "verified":
            continue
        candidate = axp_by_key.get(_key(entry))
        if candidate is None or candidate["status"] != "owner_reviewed":
            continue

        x_tick_range = (float(candidate["x_min_label"]), float(candidate["x_max_label"]))
        y_tick_range = (float(candidate["y_min_label"]), float(candidate["y_max_label"]))

        # Correctness check: build the domain object and promote it through
        # the real promote_tick_range helper, which re-validates every
        # invariant (frame-range presence, source refusal) -- this run is
        # thrown away, only its absence-of-error and its values are used;
        # the actual file mutation below still edits the raw dict directly
        # to keep the JSON diff additive-only and byte-stable elsewhere.
        pairing = parse_registry([entry])[0]
        promoted_pairing = promote_tick_range(
            pairing,
            x_tick_range=x_tick_range,
            y_tick_range=y_tick_range,
            candidate_status=candidate["status"],
        )
        assert promoted_pairing.x_tick_range == x_tick_range
        assert promoted_pairing.y_tick_range == y_tick_range

        entry["x_tick_range"] = list(x_tick_range)
        entry["y_tick_range"] = list(y_tick_range)
        entry["tick_range_source"] = "owner_reviewed"

        entry_diverged = False
        if list(entry["x_range"]) != list(x_tick_range):
            entry_diverged = True
            diverging_axes += 1
        if list(entry["y_range"]) != list(y_tick_range):
            entry_diverged = True
            diverging_axes += 1

        promoted.append(
            {
                "key": _key(entry),
                "figure_reference": entry.get("figure_reference"),
                "x_range": entry["x_range"],
                "x_tick_range": list(x_tick_range),
                "y_range": entry["y_range"],
                "y_tick_range": list(y_tick_range),
                "diverges": entry_diverged,
            }
        )

    # Re-parse + re-serialize every mutated entry through the adapter as a
    # final sanity check that the dicts we just hand-mutated are exactly
    # what serialize_entry(parse_registry(...)) would itself produce (i.e.
    # the manual mutation didn't drift from the domain model's own
    # round-trip), without actually using its output to write the file.
    for entry in registry:
        pairing = parse_registry([entry])[0]
        serialize_entry(pairing, base=entry)

    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")

    print(f"promoted: {len(promoted)} entries")
    print(f"diverging axes (tick != frame) among promoted: {diverging_axes}")
    for p in promoted:
        if p["diverges"]:
            print(f"  {p['key']}: x_range={p['x_range']} x_tick_range={p['x_tick_range']} "
                  f"y_range={p['y_range']} y_tick_range={p['y_tick_range']}")


if __name__ == "__main__":
    main()
