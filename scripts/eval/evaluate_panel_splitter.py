"""Runs PyMuPdfPanelSplitter against every locally-collected candidate image
(design §7.14, 司令塔承認 2026-08-16: パネル分割器の精度評価) and:

1. Tabulates detection statistics (grid detected vs single-panel fallback,
   panel-count distribution) across the full 2,458-image pool.
2. Writes a stratified sample (multi-panel detections + single-panel
   fallbacks) to --out-dir for manual visual audit — mirrors
   deep-digitizer's own manual-audit methodology (design §7.10).

Usage:
    python scripts/eval/evaluate_panel_splitter.py --sample-size 12
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.adapter.panel_layout import PyMuPdfPanelSplitter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--images-dir", type=pathlib.Path, default=pathlib.Path("data/raw/images")
    )
    parser.add_argument(
        "--out-dir", type=pathlib.Path, default=pathlib.Path("data/eval/panel_splitter")
    )
    parser.add_argument("--sample-size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    # max_panels=12: a real scientific figure essentially never has more
    # panels than this; anything beyond is noise-driven over-segmentation
    # (see design §7.14 finding: default max_panels=26 alone still let
    # through implausible 13-26 "panel" counts on noisy real images).
    splitter = PyMuPdfPanelSplitter(max_panels=12)
    image_paths = sorted(args.images_dir.glob("*/*"))
    print(f"found {len(image_paths)} candidate images under {args.images_dir}", file=sys.stderr)

    multi_panel: list[pathlib.Path] = []
    single_panel: list[pathlib.Path] = []
    panel_count_hist: dict[int, int] = {}
    failures = 0

    for i, path in enumerate(image_paths, start=1):
        try:
            panels = splitter.split(path.read_bytes())
        except Exception as e:  # noqa: BLE001
            failures += 1
            if failures <= 5:
                print(f"  FAILED on {path}: {e}", file=sys.stderr)
            continue

        n = len(panels)
        panel_count_hist[n] = panel_count_hist.get(n, 0) + 1
        (multi_panel if n > 1 else single_panel).append(path)

        if i % 500 == 0:
            print(f"  processed {i}/{len(image_paths)}", file=sys.stderr)

    stats = {
        "total_images": len(image_paths),
        "decode_or_split_failures": failures,
        "images_with_multi_panel_grid_detected": len(multi_panel),
        "images_treated_as_single_panel": len(single_panel),
        "multi_panel_rate": len(multi_panel) / max(1, len(image_paths) - failures),
        "panel_count_histogram": {str(k): v for k, v in sorted(panel_count_hist.items())},
    }
    print(json.dumps(stats, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "stats.json").write_text(json.dumps(stats, indent=2))

    random.seed(args.seed)
    half = args.sample_size // 2
    sample = random.sample(multi_panel, min(half, len(multi_panel))) + random.sample(
        single_panel, min(args.sample_size - half, len(single_panel))
    )

    sample_dir = args.out_dir / "sample"
    sample_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sample:
        panels = splitter.split(path.read_bytes())
        paper_id = path.parent.name
        stem = f"{paper_id}__{path.stem}"
        (sample_dir / f"{stem}__original{path.suffix}").write_bytes(path.read_bytes())
        for panel in panels:
            (sample_dir / f"{stem}__panel_{panel.label}.png").write_bytes(panel.image_bytes)
        manifest.append({"paper_id": paper_id, "source": str(path), "n_panels": len(panels)})

    (args.out_dir / "sample_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {len(sample)}-image review sample -> {sample_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
