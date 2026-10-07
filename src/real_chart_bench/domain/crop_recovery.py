"""Recovering the rectangle a hand-made crop occupies inside its source
image (scaling-verification 第0段, figure-fetch-distribution Phase 0 / §3.5).

Why this exists: 123 of the 139 verified pairings point at a crop under
data/verified_pairs/crops/ that a human cut by hand, and **no crop box was
ever recorded**. The crop bytes are the only artefact, and the pixel
coordinates in tick_calibration.json are defined against exactly those
bytes -- so main condition 2 (人が軸を校正) cannot be reproduced from a
re-fetched PDF until the box is known. figure-fetch-distribution §9 makes
the recovery an acceptance test of that whole design: anything short of
byte identity withdraws it.

The search here is **exact**. A placement is reported only when the
source's sub-rectangle is bit-identical to the crop; a best-correlation
answer is deliberately not representable. An approximate box recorded as
if it were the real one would silently shift every calibrated pixel.

Every exact placement is returned, not the "best" one: a crop taken from a
featureless region genuinely sits in many places, and that ambiguity has
to reach the caller instead of being resolved by a tiebreak nobody wrote
down.

Pure numpy array logic only -- no image codec, no I/O. Decoding a PNG/JPEG
into an array and judging byte identity of the re-encoded file are adapter
concerns (see adapter/crop_recovery.py), mirroring the
domain/panel_layout.py <-> adapter/panel_layout.py split.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from scipy.signal import fftconvolve


class CropOrientation(Enum):
    """How a cut-out rectangle was oriented before being saved.

    The eight members are the dihedral group of the rectangle -- every
    orientation reachable without resampling, so every orientation a crop
    could have been saved in while staying pixel-exact.

    Only the four rotations are *recordable*: figure-fetch-distribution
    §3.2 fixes the recipe vocabulary at ``{box, rotation_deg}`` with
    rotation_deg in {0, 90, 180, 270}. The four mirror members exist
    because paper 2.4's orientation fixes are real -- papers 83 and 5902
    were extracted vertically flipped and corrected with numpy.flipud,
    5904 with a transpose plus two flips -- and a mirror that silently
    came back as "no match" would be indistinguishable from a wrong
    source image. They are detected and reported, never written to the
    registry.

    Angles are counter-clockwise, matching numpy.rot90: a crop at
    ROTATE_90 equals ``np.rot90(source[y0:y1, x0:x1], k=1)``.
    """

    ROTATE_0 = "rotate_0"
    ROTATE_90 = "rotate_90"
    ROTATE_180 = "rotate_180"
    ROTATE_270 = "rotate_270"
    MIRROR_VERTICAL = "mirror_vertical"
    MIRROR_HORIZONTAL = "mirror_horizontal"
    TRANSPOSE = "transpose"
    ANTITRANSPOSE = "antitranspose"

    @property
    def rotation_deg(self) -> int | None:
        """The quarter-turn angle, or None for the mirror members -- which
        have no rotation_deg, rather than a rotation_deg of 0."""
        return _ROTATION_DEG.get(self)

    @property
    def is_recordable(self) -> bool:
        """True iff figure-fetch-distribution §3.2's ``rotation_deg`` can
        express this orientation."""
        return self.rotation_deg is not None

    @property
    def swaps_axes(self) -> bool:
        return self in _AXIS_SWAPPING

    @classmethod
    def rotations(cls) -> tuple[CropOrientation, ...]:
        return (cls.ROTATE_0, cls.ROTATE_90, cls.ROTATE_180, cls.ROTATE_270)


_ROTATION_DEG = {
    CropOrientation.ROTATE_0: 0,
    CropOrientation.ROTATE_90: 90,
    CropOrientation.ROTATE_180: 180,
    CropOrientation.ROTATE_270: 270,
}

_AXIS_SWAPPING = frozenset(
    {
        CropOrientation.ROTATE_90,
        CropOrientation.ROTATE_270,
        CropOrientation.TRANSPOSE,
        CropOrientation.ANTITRANSPOSE,
    }
)

ALL_ORIENTATIONS: tuple[CropOrientation, ...] = tuple(CropOrientation)


@dataclass(frozen=True)
class CropPlacement:
    """Where a crop sits in its source, and how it was turned.

    ``box`` is ``(x0, y0, x1, y1)`` in the *source* image's pixel
    coordinates, half-open on the far edge -- the same convention as numpy
    slicing and PIL's ``Image.crop``, so a box is usable without a
    translation step that could introduce an off-by-one.
    """

    box: tuple[int, int, int, int]
    orientation: CropOrientation

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


def _orient(region: np.ndarray, orientation: CropOrientation) -> np.ndarray:
    """Apply ``orientation`` to a cut-out region (source space -> crop space)."""
    if orientation is CropOrientation.ROTATE_0:
        return region
    if orientation is CropOrientation.ROTATE_90:
        return np.rot90(region, k=1)
    if orientation is CropOrientation.ROTATE_180:
        return np.rot90(region, k=2)
    if orientation is CropOrientation.ROTATE_270:
        return np.rot90(region, k=3)
    if orientation is CropOrientation.MIRROR_VERTICAL:
        return region[::-1]
    if orientation is CropOrientation.MIRROR_HORIZONTAL:
        return region[:, ::-1]
    if orientation is CropOrientation.TRANSPOSE:
        return np.swapaxes(region, 0, 1)
    if orientation is CropOrientation.ANTITRANSPOSE:
        return np.swapaxes(region, 0, 1)[::-1, ::-1]
    raise AssertionError(f"unhandled orientation {orientation}")  # pragma: no cover


def _unorient(crop: np.ndarray, orientation: CropOrientation) -> np.ndarray:
    """Inverse of :func:`_orient` (crop space -> source space), so the result
    can be searched for as a plain sub-rectangle."""
    inverse = {
        CropOrientation.ROTATE_90: CropOrientation.ROTATE_270,
        CropOrientation.ROTATE_270: CropOrientation.ROTATE_90,
    }.get(orientation, orientation)
    return _orient(crop, inverse)


def apply_placement(source: np.ndarray, placement: CropPlacement) -> np.ndarray:
    """Re-cut ``source`` according to ``placement``.

    The inverse of the search: feeding a recovered placement back through
    this is how byte identity is checked (adapter/crop_recovery.py).
    """
    x0, y0, x1, y1 = placement.box
    height, width = source.shape[:2]
    if x0 < 0 or y0 < 0 or x1 > width or y1 > height or x0 >= x1 or y0 >= y1:
        raise ValueError(
            f"box {placement.box} is empty or outside a {width}x{height} source image"
        )
    return _orient(source[y0:y1, x0:x1], placement.orientation)


# Tolerance on the sum-of-squared-differences prefilter below. The window
# sums of source^2 are computed exactly (int64 integral image); only the
# cross-correlation term goes through an FFT, whose float64 relative error
# (~1e-16) over the largest source in the dataset (2927x5017) leaves an
# absolute error of order 1e-4 -- four orders below this tolerance. The
# prefilter can therefore only ever over-admit, and every admitted
# candidate is then compared bit for bit, so a false positive is
# impossible and a false negative would need a source ~1e12 times larger.
_SSD_TOLERANCE = 1.0


def _window_square_sums(plane: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Exact per-window sums of ``plane**2`` for every window of ``shape``."""
    squares = plane.astype(np.int64) ** 2
    integral = np.zeros((squares.shape[0] + 1, squares.shape[1] + 1), dtype=np.int64)
    integral[1:, 1:] = squares.cumsum(axis=0).cumsum(axis=1)
    height, width = shape
    return (
        integral[height:, width:]
        - integral[:-height or None, width:]
        - integral[height:, :-width or None]
        + integral[:-height or None, :-width or None]
    ).astype(np.float64)


def _exact_offsets(source: np.ndarray, template: np.ndarray) -> list[tuple[int, int]]:
    """Every (x, y) where ``template`` sits bit-identically in ``source``."""
    source_height, source_width = source.shape[:2]
    template_height, template_width = template.shape[:2]
    if template_height > source_height or template_width > source_width:
        return []

    source_plane = source if source.ndim == 2 else source[:, :, source.shape[2] // 2]
    template_plane = template if template.ndim == 2 else template[:, :, template.shape[2] // 2]

    cross = fftconvolve(
        source_plane.astype(np.float64),
        template_plane.astype(np.float64)[::-1, ::-1],
        mode="valid",
    )
    ssd = (
        _window_square_sums(source_plane, (template_height, template_width))
        - 2.0 * cross
        + float((template_plane.astype(np.int64) ** 2).sum())
    )

    found: list[tuple[int, int]] = []
    for y, x in np.argwhere(ssd < _SSD_TOLERANCE):
        y, x = int(y), int(x)
        region = source[y : y + template_height, x : x + template_width]
        if np.array_equal(region, template):
            found.append((x, y))
    return found


def locate_crop(
    source: np.ndarray,
    crop: np.ndarray,
    *,
    orientations: tuple[CropOrientation, ...] = ALL_ORIENTATIONS,
) -> tuple[CropPlacement, ...]:
    """Every exact placement of ``crop`` inside ``source``.

    Both arrays are decoded pixel arrays: ``(h, w)`` greyscale or
    ``(h, w, c)``. A crop whose channel count differs from the source's is
    never a sub-rectangle of it, and yields no placement rather than being
    coerced into one -- a greyscale crop of an RGB source means the crop
    was converted after cutting, which is a different (and unrecorded)
    recipe.

    Returns placements in reading order (top-to-bottom, left-to-right,
    orientations in declaration order). Empty when there is no exact match.
    """
    if source.ndim != crop.ndim:
        return ()
    if source.ndim == 3 and source.shape[2] != crop.shape[2]:
        return ()

    placements: list[CropPlacement] = []
    for orientation in orientations:
        template = np.ascontiguousarray(_unorient(crop, orientation))
        for x, y in _exact_offsets(source, template):
            placements.append(
                CropPlacement(
                    box=(x, y, x + template.shape[1], y + template.shape[0]),
                    orientation=orientation,
                )
            )

    order = {orientation: index for index, orientation in enumerate(orientations)}
    placements.sort(key=lambda p: (p.box[1], p.box[0], order[p.orientation]))
    return tuple(placements)
