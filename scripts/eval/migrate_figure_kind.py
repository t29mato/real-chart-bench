"""One-off migration (design §7.59): writes figure_kind/figure_tags into
data/verified_pairs/registry.json for the 139 VERIFIED entries, derived
mechanically from the classification pass recorded in the source JSON this
script reads (a scratchpad artifact, not committed to the repo -- see the
task record / §7.59 for provenance).

Derivation, per entry (matched to registry.json by figure_id, which is
unique across the registry):

- figure_kind: has_markers == false -> "line_only", otherwise "markers".
  This is the whole taxonomy now (design §7.59) -- the earlier line/
  scatter/mixed connector-vs-fit distinction is abandoned.
- figure_tags: the source's `tags` list, filtered to the observation-
  quality tags that survive the taxonomy cut -- inset, schematic,
  panel_bleed, error_bars, scan_artifact, reference_curve, dense_overlap.
  `fitting_line` is deliberately dropped: it encoded exactly the
  connector-vs-fit judgement the owner abandoned as unreliable, and
  carrying it forward would invite rebuilding that distinction on data
  nobody trusts.

Additive only: every other field on every entry (verified AND rejected) is
untouched byte-for-byte. Follows the same "mutate the raw registry dicts in
place, preserve key order, run through the domain model as a correctness
check before writing" convention as promote_tick_ranges.py (design §7.57).
"""

from __future__ import annotations

import json
from pathlib import Path

from real_chart_bench.adapter.verified_pairing_registry import parse_registry, serialize_entry

REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / "data/verified_pairs/registry.json"
SOURCE_PATH = Path(
    "/tmp/claude-1000/-home-mato-repos-real-chart-bench/"
    "628beb76-383b-42e9-bf21-8d4188daf8dc/scratchpad/final_merged.json"
)

# design §7.59: the observation-quality tags that survive the classification
# cut. "fitting_line" is deliberately excluded -- see module docstring.
SURVIVING_TAGS = frozenset(
    {
        "inset",
        "schematic",
        "panel_bleed",
        "error_bars",
        "scan_artifact",
        "reference_curve",
        "dense_overlap",
    }
)


def _figure_kind(has_markers: bool) -> str:
    return "markers" if has_markers else "line_only"


def main() -> None:
    registry = json.loads(REGISTRY_PATH.read_text())
    source = json.loads(SOURCE_PATH.read_text())

    source_by_figure_id = {e["figure_id"]: e for e in source}
    assert len(source_by_figure_id) == len(source), "duplicate figure_id in source"

    migrated = 0
    kind_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    dropped_fitting_line = 0

    for entry in registry:
        source_entry = source_by_figure_id.get(entry["figure_id"])
        if source_entry is None:
            continue  # not in the source pass (e.g. rejected entries)
        assert entry.get("status") == "verified", (
            f"source entry for figure_id={entry['figure_id']} matched a "
            f"non-verified registry entry -- unexpected"
        )

        kind = _figure_kind(source_entry["has_markers"])
        raw_tags = source_entry.get("tags", [])
        if "fitting_line" in raw_tags:
            dropped_fitting_line += 1
        tags = [t for t in raw_tags if t in SURVIVING_TAGS]
        unexpected = [t for t in raw_tags if t not in SURVIVING_TAGS and t != "fitting_line"]
        assert not unexpected, f"unexpected tag(s) {unexpected} on figure_id={entry['figure_id']}"

        entry["figure_kind"] = kind
        entry["figure_tags"] = tags

        migrated += 1
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        for t in tags:
            tag_counts[t] = tag_counts.get(t, 0) + 1

    assert migrated == len(source), f"migrated {migrated}, expected {len(source)}"

    # Re-parse + re-serialize every entry through the adapter as a final
    # sanity check that the dicts we just hand-mutated are exactly what
    # serialize_entry(parse_registry(...)) would itself produce, without
    # actually using its output to write the file.
    for entry in registry:
        pairing = parse_registry([entry])[0]
        serialize_entry(pairing, base=entry)

    REGISTRY_PATH.write_text(json.dumps(registry, indent=2, ensure_ascii=False) + "\n")

    print(f"migrated: {migrated} entries")
    print(f"figure_kind counts: {kind_counts}")
    print(f"figure_tags counts: {tag_counts}")
    print(f"fitting_line dropped from: {dropped_fitting_line} entries")


if __name__ == "__main__":
    main()
