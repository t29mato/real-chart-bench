"""Dataset card for the nc subset (design §7.88.1, review M3).

The card's licence is derived from the rows it describes, never hard-coded:
the Hugging Face ``license`` metadata takes one identifier, so it is that
identifier when every row has the same licence and HF knows it
(``cc-by-nc-4.0``, ``cc-by-nc-sa-4.0``, ...); otherwise ``license: other`` with
a ``license_name``, and the body lists every licence present with its paper
count. The per-row ``license`` field in metadata.jsonl stays the source of truth.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any

from real_chart_bench.domain.licensing import normalize_license_id

# NC licences that have a Hugging Face licence identifier of the same name
_HF_NC_IDS = frozenset(
    f"cc-by-nc{sa}-{v}" for sa in ("", "-sa") for v in ("2.0", "3.0", "4.0")
)
_MIXED_NAME = "cc-by-nc-family"


def _licences(rows: Iterable[Mapping[str, Any]]) -> Counter[str]:
    return Counter(normalize_license_id(r.get("license")) or "unknown" for r in rows)


def card_licence_metadata(rows: Iterable[Mapping[str, Any]]) -> dict[str, str]:
    licences = _licences(rows)
    if len(licences) == 1:
        (only,) = licences
        if only in _HF_NC_IDS:
            return {"license": only}
    return {"license": "other", "license_name": _MIXED_NAME}


def render_nc_dataset_card(rows: list[Mapping[str, Any]]) -> str:
    licences = _licences(rows)
    meta = card_licence_metadata(rows)
    front = "\n".join(f"{k}: {v}" for k, v in meta.items())
    listing = "\n".join(
        f"- `{lic}` ({n} paper{'s' if n != 1 else ''})" for lic, n in sorted(licences.items())
    )
    share_alike = any("sa" in lic.split("-") for lic in licences)
    sa_note = (
        "\n- **ShareAlike**: rows under a CC BY-NC-SA licence carry its ShareAlike\n"
        "  condition: anything you build from them must be distributed under the\n"
        "  same licence."
        if share_alike
        else ""
    )
    other_note = (
        "\n\n`license: other` because the rows do not share one licence with a\n"
        "Hugging Face identifier; the licences are listed above and each row's\n"
        "`license` field in metadata.jsonl is authoritative."
        if meta["license"] == "other"
        else ""
    )
    return f"""\
---
{front}
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
papers published under Creative Commons NonCommercial licences. They are kept
apart from the core dataset (CC BY / CC BY-SA / CC0), which has no such
restriction, and are scored separately -- never pooled with the core numbers
(design doc `docs/design/benchmark-architecture.md` §7.88).

Figure image licences in this dataset (per row in the `license` field):

{listing}{other_note}

- **Ground truth license**: CC BY 4.0 (Starrydata / NIMS MDR), as in the core set.{sa_note}
- **Caveat**: as in the core set, `image_files` is the per-paper candidate pool,
  not yet matched to a specific `figure_id`.
"""
