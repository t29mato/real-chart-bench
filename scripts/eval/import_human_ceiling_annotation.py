"""Converts a starry-digitizer "Export Project" .zip into a
data/human_ceiling/annotations/*.json record (see FORMAT.md's "Using
starry-digitizer for the re-digitization" section for the full workflow this
wraps: load the image in starry-digitizer, calibrate axes, digitize each
series into its own dataset named after that series, then Export Project).

Usage:
    python scripts/eval/import_human_ceiling_annotation.py \\
        --project ~/Downloads/sd-20260904-1200.zip \\
        --paper-id 4173 --figure-id 20120 \\
        --annotator-id t29mato --annotated-at 2026-09-04 \\
        [--notes "series '1300 degC' excluded: overlaps '1250 degC', could not separate"] \\
        [--out data/human_ceiling/annotations/4173-20120__t29mato.json]

Defaults --annotation-source to "human" (this script exists to support the
*human* re-digitization workflow -- see require_human_ceiling() in
domain/human_ceiling.py for what that field controls downstream). Only pass
--annotation-source llm/automated deliberately, e.g. to reuse this converter
for a non-human digitizer that happens to also export starry-digitizer
project files.

Never overwrites an existing annotation file silently -- pass --force to
replace one on purpose.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.adapter.starry_digitizer_import import (  # noqa: E402
    convert_project_to_annotation,
    load_project_from_zip,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
ANNOTATIONS_DIR = REPO_ROOT / "data/human_ceiling/annotations"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--project", required=True, type=pathlib.Path, help="starry-digitizer 'Export Project' .zip"
    )
    parser.add_argument("--paper-id", required=True)
    parser.add_argument("--figure-id", required=True)
    parser.add_argument(
        "--annotator-id", required=True, help="the human annotator's identity/handle"
    )
    parser.add_argument("--annotated-at", required=True, help="ISO date, YYYY-MM-DD")
    parser.add_argument(
        "--annotation-source",
        default="human",
        choices=["human", "llm", "automated"],
        help="default: human -- change only deliberately, see module docstring",
    )
    parser.add_argument("--tool", default="starry-digitizer")
    parser.add_argument(
        "--notes",
        default=None,
        help="free text, e.g. a series that could not be reliably digitized",
    )
    parser.add_argument(
        "--out",
        type=pathlib.Path,
        default=None,
        help="default: data/human_ceiling/annotations/<paper>-<figure>__<annotator>.json",
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing output file")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    project = load_project_from_zip(args.project)
    record = convert_project_to_annotation(
        project,
        paper_id=args.paper_id,
        figure_id=args.figure_id,
        annotator_id=args.annotator_id,
        annotated_at=args.annotated_at,
        annotation_source=args.annotation_source,
        tool=args.tool,
        notes=args.notes,
    )

    out_path = (
        args.out
        or ANNOTATIONS_DIR / f"{args.paper_id}-{args.figure_id}__{args.annotator_id}.json"
    )
    if out_path.exists() and not args.force:
        print(
            f"error: {out_path} already exists -- pass --force to overwrite it on purpose",
            file=sys.stderr,
        )
        raise SystemExit(1)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(record, indent=2) + "\n")

    n_curves = len(record["curves"])
    n_points = sum(len(c["x"]) for c in record["curves"])
    print(f"wrote {out_path} ({n_curves} curve(s), {n_points} point(s) total)")
    if args.annotation_source != "human":
        print(
            f"NOTE: annotation_source={args.annotation_source!r} -- this will NOT count "
            "toward a 'human ceiling' score (see domain.human_ceiling.require_human_ceiling())."
        )


if __name__ == "__main__":
    main()
