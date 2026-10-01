"""Converts a starry-digitizer "Export Project" .zip into a supplementary
ground-truth record: series the figure draws but Starrydata never digitized
(reference curves from other works, a comparison sample). See
data/verified_pairs/ground_truth_supplement/README.md for the workflow and the
figure list, and adapter/ground_truth_store.py for how scoring uses it.

Digitize *only* the missing series -- the Starrydata series stay as they are.

Usage:
    python scripts/eval/import_gt_supplement.py \\
        --project ~/Downloads/sd-20261002-1200.zip \\
        --paper-id 18869 --figure-id 18874 \\
        --digitized-by t29mato --digitized-at 2026-10-02 \\
        --reason "3 reference curves from other works drawn in the figure"

Never overwrites an existing record silently -- pass --force to replace one.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from real_chart_bench.adapter.ground_truth_store import load_ground_truth  # noqa: E402
from real_chart_bench.adapter.starry_digitizer_import import (  # noqa: E402
    convert_project_to_annotation,
    load_project_from_zip,
)

SUPPLEMENT_DIR = REPO / "data/verified_pairs/ground_truth_supplement"
GROUND_TRUTH_PATH = REPO / "data/verified_pairs/ground_truth.json"


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--project", required=True, type=pathlib.Path)
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--figure-id", required=True)
    parser.add_argument("--digitized-by", required=True)
    parser.add_argument("--digitized-at", required=True, help="YYYY-MM-DD")
    parser.add_argument("--reason", required=True, help="why these series are not in Starrydata")
    parser.add_argument("--tool", default="starry-digitizer")
    parser.add_argument("--notes", default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    converted = convert_project_to_annotation(
        load_project_from_zip(args.project),
        paper_id=args.paper_id,
        figure_id=args.figure_id,
        annotator_id=args.digitized_by,
        annotated_at=args.digitized_at,
        tool=args.tool,
    )
    record = {
        "paper_id": args.paper_id,
        "figure_id": args.figure_id,
        "digitized_by": args.digitized_by,
        "digitized_at": args.digitized_at,
        "tool": args.tool,
        "reason": args.reason,
        "notes": args.notes,
        "curves": converted["curves"],
    }
    out = SUPPLEMENT_DIR / f"{args.paper_id}-{args.figure_id}.json"
    if out.exists() and not args.force:
        raise SystemExit(f"error: {out} already exists -- pass --force to replace it on purpose")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
    try:
        load_ground_truth(GROUND_TRUTH_PATH, SUPPLEMENT_DIR)  # validates every record
    except Exception:
        out.unlink()
        raise
    labels = ", ".join(c["series_label"] for c in record["curves"])
    print(f"wrote {out}: {len(record['curves'])} series ({labels})")


if __name__ == "__main__":
    main()
