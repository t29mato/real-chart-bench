"""Prepares (but does NOT upload) a Hugging Face Hub dataset layout for
Tier 2 (image-paired) papers (design §7.13/§7.14, 司令塔承認 2026-08-16).

司令塔方針: Tier 2画像はHuggingFace Hub(dataset repo)でホストする。ただし
**アップロードは本リポジトリの公開承認後、かつHFトークン受領後**。このスクリプトは
その日のために変換ロジック・データセットカードだけを先に用意するもので、
`--upload` を渡さない限りネットワークには一切触れない。`--upload` を渡しても
`HF_TOKEN` 環境変数が無ければ即座に拒否する(誤アップロード防止の多重ガード)。

Output layout (data/hf_dataset/, gitignored — regeneratable from
data/manifest/v0/ + data/raw/images/, not itself a source of truth):

    data/hf_dataset/
        README.md          # dataset card draft (HF front-matter + description)
        metadata.jsonl      # one row per Tier-2 paper (image_files are paths
                             # relative to data/raw/images/<paper_id>/, NOT
                             # copied — avoids duplicating 546MB until the
                             # actual upload step)

Dataset subsets (design §7.88.1): only ``core`` papers (CC BY / BY-SA / CC0)
go into data/hf_dataset/. ``nc`` papers (CC BY-NC / BY-NC-SA) go into their own
directory, data/hf_dataset_nc/ (gitignored), with their own card
(``license: cc-by-nc-4.0``, non-commercial use only) -- a separate HF dataset
(or config) at upload time, never mixed into the core one. It is only written
when at least one nc paper has images.

Usage:
    python scripts/publish/prepare_hf_dataset.py              # prepare only
    python scripts/publish/prepare_hf_dataset.py --upload      # refuses
                                                                # without HF_TOKEN
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

from real_chart_bench.domain.dataset_subset import DatasetSubset  # noqa: E402
from real_chart_bench.usecase.dataset_subsets import manifest_entry_subset  # noqa: E402

REPO_ID = "real-chart-bench/thermoelectric-v0"  # placeholder, confirm with 司令塔 before real use
DATASET_CARD = """\
---
license: cc-by-4.0
task_categories:
- image-to-text
- table-question-answering
tags:
- chart-data-extraction
- scientific-figures
- thermoelectric-materials
pretty_name: real-chart-bench v0 (thermoelectric materials, Tier 2 image-paired)
---

# real-chart-bench v0 — thermoelectric materials (Tier 2, image-paired)

Candidate figure images extracted from CC BY 4.0 open-access papers (design
doc `docs/design/benchmark-architecture.md` §7.9-§7.13), paired at the
*paper* level (not yet the individual figure/panel level — see the
open pairing problem in §7.10/§7.12) with ground-truth XY curves digitized
by [Starrydata](https://www.starrydata2.org/) (CC BY 4.0, NIMS MDR).

- **Ground truth license**: CC BY 4.0 (Starrydata / NIMS MDR)
- **Figure image license**: CC BY 4.0 (verified per-paper via OpenAlex; see
  each row's `license` field for the source paper's exact license string)
- **Caveat**: `image_files` for a paper is the *candidate pool* extracted
  from its PDF (embedded raster images + whole-page renders), not yet
  automatically matched to a specific `figure_id`. Treat this as weak
  supervision until the pairing problem (§7.10) is solved.

See `docs/design/benchmark-architecture.md` (real-chart-bench repository)
for the full collection methodology.
"""


NC_DATASET_CARD = """\
---
license: cc-by-nc-4.0
task_categories:
- image-to-text
- table-question-answering
tags:
- chart-data-extraction
- scientific-figures
- thermoelectric-materials
pretty_name: real-chart-bench v0 NC subset (non-commercial use only)
---

# real-chart-bench v0 — NC subset (non-commercial use only)

**Non-commercial use only.** The figure images here come from open-access
papers published under CC BY-NC or CC BY-NC-SA (see each row's `license`). They
are kept apart from the core dataset (CC BY / CC BY-SA / CC0), which has no such
restriction, and are scored separately -- never pooled with the core numbers
(design doc `docs/design/benchmark-architecture.md` §7.88).

- **Figure image license**: CC BY-NC or CC BY-NC-SA, per row. Rows under
  CC BY-NC-SA carry its ShareAlike condition: anything you build from them must
  be distributed under the same licence.
- **Ground truth license**: CC BY 4.0 (Starrydata / NIMS MDR), as in the core set.
- **Caveat**: as in the core set, `image_files` is the per-paper candidate pool,
  not yet matched to a specific `figure_id`.
"""


def build_metadata(
    manifest_dir: pathlib.Path,
    raw_images_dir: pathlib.Path,
    subset: DatasetSubset = DatasetSubset.CORE,
) -> list[dict]:
    papers = json.loads((manifest_dir / "papers.json").read_text())
    figures = json.loads((manifest_dir / "figures.json").read_text())
    curves = json.loads((manifest_dir / "curves.json").read_text())

    figures_by_paper: dict[str, list[dict]] = collections.defaultdict(list)
    for fig in figures:
        figures_by_paper[fig["paper_id"]].append(fig)
    curves_by_figure: dict[str, list[dict]] = collections.defaultdict(list)
    for curve in curves:
        curves_by_figure[curve["figure_id"]].append(curve)

    rows = []
    for paper in papers:
        if not paper.get("n_extracted_images"):
            continue  # Tier 1 only (no image pool) — not part of the HF dataset
        if manifest_entry_subset(paper) is not subset:
            continue  # design 7.88.1: one subset per dataset, never mixed

        paper_id = paper["paper_id"]
        image_dir = raw_images_dir / paper_id
        image_files = sorted(p.name for p in image_dir.glob("*")) if image_dir.exists() else []
        if not image_files:
            # manifest says images exist but local files are missing; skip, don't fabricate
            continue

        rows.append(
            {
                "paper_id": paper_id,
                "doi": paper["doi"],
                "license": paper.get("license_id"),
                "image_files": image_files,  # relative to images/<paper_id>/ at upload time
                "figures": [
                    {
                        "figure_id": fig["figure_id"],
                        "figure_reference": fig["figure_reference"],
                        "split": fig["split"],
                        "curves": [
                            {
                                "series_label": c["series_label"],
                                "n_points": c["n_points"],
                                "license": c["license"],
                            }
                            for c in curves_by_figure.get(fig["figure_id"], [])
                        ],
                    }
                    for fig in figures_by_paper.get(paper_id, [])
                ],
            }
        )
    return rows


def upload(out_dir: pathlib.Path) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(
            "REFUSING to upload: HF_TOKEN is not set, and the repository has not yet "
            "received public-release approval (design §7.13/§7.14, 司令塔承認待ち). "
            "This is intentional — do not set HF_TOKEN to bypass this without confirming "
            "the release approval first.",
            file=sys.stderr,
        )
        raise SystemExit(1)
    # Deliberately not implemented further: the actual huggingface_hub upload
    # call is written at approval time, not before, so there is no "just add
    # a token" path to an accidental early publish.
    raise NotImplementedError(
        "Upload step intentionally stubbed — implement with huggingface_hub.HfApi "
        "once repo + HF token approval is granted (design §7.13/§7.14)."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest-dir", type=pathlib.Path, default=pathlib.Path("data/manifest/v0")
    )
    parser.add_argument(
        "--raw-images-dir", type=pathlib.Path, default=pathlib.Path("data/raw/images")
    )
    parser.add_argument("--out-dir", type=pathlib.Path, default=pathlib.Path("data/hf_dataset"))
    parser.add_argument("--upload", action="store_true", help="refuses unless HF_TOKEN is set")
    args = parser.parse_args()

    for subset, out_dir, card in (
        (DatasetSubset.CORE, args.out_dir, DATASET_CARD),
        (DatasetSubset.NC, args.out_dir.with_name(args.out_dir.name + "_nc"), NC_DATASET_CARD),
    ):
        rows = build_metadata(args.manifest_dir, args.raw_images_dir, subset)
        if subset is not DatasetSubset.CORE and not rows:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "metadata.jsonl").open("w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        (out_dir / "README.md").write_text(card)

        n_images = sum(len(r["image_files"]) for r in rows)
        print(f"prepared {len(rows)} {subset.value} papers, {n_images} images -> {out_dir}")
    print(f"(not uploaded; repo_id placeholder: {REPO_ID} — confirm with 司令塔 before real use)")

    if args.upload:
        upload(args.out_dir)


if __name__ == "__main__":
    main()
