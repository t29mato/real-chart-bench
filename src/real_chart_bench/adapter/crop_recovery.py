"""Pillow-backed crop-box recovery (scaling-verification 第0段,
figure-fetch-distribution Phase 0 / §3.5).

Decodes a committed crop and a candidate source image, delegates the exact
rectangle search to the pure domain.crop_recovery.locate_crop, then re-cuts
the source and judges the result against the committed file **byte for
byte** -- the acceptance bar figure-fetch-distribution §9 sets for Phase 0.

Why Pillow and not pymupdf (which adapter/panel_layout.py uses): byte
identity is a property of the *encoder*, and the committed crops were
written by Pillow. Judging them against a pymupdf-encoded PNG would fail
every entry for a reason that has nothing to do with the crop box.

CROP_ENCODER below names the one canonical procedure byte identity is
judged against. It is fixed in advance, deliberately: tuning encoder
options per entry until the bytes line up would be fitting the procedure
to the answer, and would make "byte-identical" mean nothing.
"""

from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from real_chart_bench.domain.crop_recovery import (
    ALL_ORIENTATIONS,
    CropOrientation,
    CropPlacement,
    locate_crop,
)
from real_chart_bench.domain.verified_pairing import CropRecipe

# The canonical crop procedure: open the source, Image.crop(box), apply the
# quarter turn if any, Image.save(format="PNG") with no further options.
# This is what produced the crops re-cut on 2026-09-30 (commit f2f7bde).
CROP_ENCODER = "pillow Image.save(format='PNG'), default options"

_HASH_CHUNK = 1 << 20

# Why the recipe is withheld, in the one word the report groups failures by.
NO_EXACT_PLACEMENT = "no_exact_placement"
AMBIGUOUS_PLACEMENT = "ambiguous_placement"
ORIENTATION_NOT_RECORDABLE = "orientation_not_recordable"
BYTES_DIFFER = "bytes_differ"


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class CropRecoveryResult:
    """What was learned about one committed crop.

    ``recipe`` is non-None only when the box was found unambiguously, its
    orientation is recordable, AND re-cutting the source reproduces the
    committed file byte for byte. Anything less leaves ``recipe`` None with
    ``failure_reason`` set -- there is deliberately no "close enough"
    recipe, because tick_calibration.json's pixel coordinates are defined
    against the committed bytes.

    ``box`` / ``orientation`` are still reported on a failure when a
    placement *was* found, so a near miss is visible in the report instead
    of being flattened into "not found".
    """

    crop_path: str
    source_image_path: str | None
    final_sha256: str
    recipe: CropRecipe | None = None
    box: tuple[int, int, int, int] | None = None
    orientation: CropOrientation | None = None
    placement_count: int = 0
    pixels_identical: bool = False
    bytes_identical: bool = False
    reencoded_sha256: str | None = None
    failure_reason: str | None = None

    @property
    def is_accepted(self) -> bool:
        return self.recipe is not None


def _decode(path: Path | str) -> Image.Image:
    """Decode without changing mode: a greyscale crop of a greyscale source
    must stay greyscale, or the arrays could not be compared exactly."""
    image = Image.open(path)
    image.load()
    return image


def _as_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image)


def _encode_png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _recut(source: Image.Image, placement: CropPlacement) -> Image.Image:
    x0, y0, x1, y1 = placement.box
    region = source.crop((x0, y0, x1, y1))
    rotation = placement.orientation.rotation_deg
    if rotation:
        # Image.rotate(expand=True) turns counter-clockwise, matching
        # numpy.rot90 and CropRecipe.rotation_deg.
        region = region.rotate(rotation, expand=True)
    return region


def reproduce_crop(source_path: Path | str, recipe: CropRecipe) -> str:
    """Execute a recorded recipe and return the sha256 of the resulting PNG.

    This is the forward direction -- what a fetch script does once the box
    is known (figure-fetch-distribution §3.3's ``final_sha256 一致?`` rung):
    no search, just cut and encode. Comparing the result against the
    entry's ``final_sha256`` is the whole check.
    """
    source_image = _decode(source_path)
    placement = CropPlacement(
        box=recipe.box,
        orientation=next(
            orientation
            for orientation in CropOrientation.rotations()
            if orientation.rotation_deg == recipe.rotation_deg
        ),
    )
    return hashlib.sha256(_encode_png(_recut(source_image, placement))).hexdigest()


def recover_crop(
    source_path: Path | str,
    crop_path: Path | str,
    *,
    orientations: tuple[CropOrientation, ...] = ALL_ORIENTATIONS,
) -> CropRecoveryResult:
    """Locate ``crop_path`` inside ``source_path`` and verify byte identity."""
    final_sha256 = sha256_file(crop_path)
    crop_image = _decode(crop_path)
    source_image = _decode(source_path)

    placements = locate_crop(
        _as_array(source_image), _as_array(crop_image), orientations=orientations
    )

    partial = {
        "crop_path": str(crop_path),
        "source_image_path": str(source_path),
        "final_sha256": final_sha256,
        "placement_count": len(placements),
    }

    if not placements:
        return CropRecoveryResult(**partial, failure_reason=NO_EXACT_PLACEMENT)

    placement = placements[0]
    partial["box"] = placement.box
    partial["orientation"] = placement.orientation
    partial["pixels_identical"] = True

    if len(placements) > 1:
        return CropRecoveryResult(**partial, failure_reason=AMBIGUOUS_PLACEMENT)
    if not placement.orientation.is_recordable:
        return CropRecoveryResult(**partial, failure_reason=ORIENTATION_NOT_RECORDABLE)

    reencoded = _encode_png(_recut(source_image, placement))
    reencoded_sha256 = hashlib.sha256(reencoded).hexdigest()
    bytes_identical = reencoded_sha256 == final_sha256
    partial["reencoded_sha256"] = reencoded_sha256
    partial["bytes_identical"] = bytes_identical

    if not bytes_identical:
        return CropRecoveryResult(**partial, failure_reason=BYTES_DIFFER)

    return CropRecoveryResult(
        **partial,
        recipe=CropRecipe(
            source_image_path=str(source_path),
            box=placement.box,
            rotation_deg=placement.orientation.rotation_deg,
        ),
    )
