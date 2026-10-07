"""TDD for domain/crop_recovery.py -- recovering the rectangle a hand-made
crop occupies inside its source image (scaling-verification 第0段,
figure-fetch-distribution Phase 0 / §3.5).

The search is *exact*: a placement is only reported when the source's
sub-rectangle is bit-identical to the crop. An approximate (best
correlation) answer is deliberately not representable -- design
figure-fetch-distribution §9 makes byte identity the acceptance bar, and a
near-miss recorded as a hit would silently invalidate the pixel
coordinates in tick_calibration.json.
"""

import numpy as np
import pytest

from real_chart_bench.domain.crop_recovery import (
    CropOrientation,
    CropPlacement,
    apply_placement,
    locate_crop,
)

RNG = np.random.default_rng(20261007)


def _noise(height: int, width: int) -> np.ndarray:
    """Textured source -- a flat image would legitimately match everywhere."""
    return RNG.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


# --- CropOrientation ---------------------------------------------------------


def test_the_eight_orientations_are_the_dihedral_group_of_the_rectangle():
    assert {o.value for o in CropOrientation} == {
        "rotate_0",
        "rotate_90",
        "rotate_180",
        "rotate_270",
        "mirror_vertical",
        "mirror_horizontal",
        "transpose",
        "antitranspose",
    }


def test_only_the_four_rotations_have_a_rotation_angle():
    recordable = {o for o in CropOrientation if o.rotation_deg is not None}

    assert {o.rotation_deg for o in recordable} == {0, 90, 180, 270}
    assert CropOrientation.MIRROR_VERTICAL.rotation_deg is None
    assert CropOrientation.TRANSPOSE.rotation_deg is None


def test_a_mirror_is_not_recordable_as_a_crop_recipe():
    assert CropOrientation.ROTATE_90.is_recordable
    assert not CropOrientation.MIRROR_HORIZONTAL.is_recordable


# --- the match case ----------------------------------------------------------


def test_finds_the_one_rectangle_an_axis_aligned_crop_came_from():
    source = _noise(200, 300)
    crop = source[40:130, 70:210]

    placements = locate_crop(source, crop)

    assert placements == (
        CropPlacement(box=(70, 40, 210, 130), orientation=CropOrientation.ROTATE_0),
    )


def test_the_box_is_in_source_coordinates_and_half_open():
    source = _noise(60, 80)
    crop = source[0:10, 0:20]

    (placement,) = locate_crop(source, crop)

    assert placement.box == (0, 0, 20, 10)
    assert np.array_equal(apply_placement(source, placement), crop)


def test_a_whole_image_copy_is_reported_as_the_full_rectangle():
    source = _noise(50, 70)

    (placement,) = locate_crop(source, source.copy())

    assert placement.box == (0, 0, 70, 50)


def test_a_greyscale_two_dimensional_crop_is_matched_too():
    source = RNG.integers(0, 256, size=(90, 120), dtype=np.uint8)
    crop = source[10:40, 20:60]

    (placement,) = locate_crop(source, crop)

    assert placement.box == (20, 10, 60, 40)


# --- rotations ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("orientation", "turns"),
    [
        (CropOrientation.ROTATE_90, 1),
        (CropOrientation.ROTATE_180, 2),
        (CropOrientation.ROTATE_270, 3),
    ],
)
def test_detects_a_crop_that_was_rotated_after_cutting(orientation, turns):
    source = _noise(200, 300)
    region = source[40:130, 70:210]
    crop = np.rot90(region, k=turns)

    placements = locate_crop(source, crop)

    assert placements == (CropPlacement(box=(70, 40, 210, 130), orientation=orientation),)
    assert np.array_equal(apply_placement(source, placements[0]), crop)


def test_detects_a_mirrored_crop_separately_from_the_rotations():
    source = _noise(120, 160)
    region = source[20:70, 30:110]
    crop = region[::-1]

    (placement,) = locate_crop(source, crop)

    assert placement.orientation is CropOrientation.MIRROR_VERTICAL
    assert placement.box == (30, 20, 110, 70)
    assert not placement.orientation.is_recordable


def test_restricting_the_search_to_rotations_hides_a_mirrored_crop():
    source = _noise(120, 160)
    crop = source[20:70, 30:110][::-1]

    assert locate_crop(source, crop, orientations=CropOrientation.rotations()) == ()


# --- the no-match case -------------------------------------------------------


def test_a_crop_that_is_not_in_the_source_yields_no_placement():
    source = _noise(200, 300)
    crop = _noise(40, 50)

    assert locate_crop(source, crop) == ()


def test_a_one_pixel_edit_is_enough_to_reject_a_placement():
    source = _noise(200, 300)
    crop = source[40:130, 70:210].copy()
    crop[45, 60, 1] = (int(crop[45, 60, 1]) + 128) % 256

    assert locate_crop(source, crop) == ()


def test_a_crop_larger_than_its_candidate_source_yields_no_placement():
    assert locate_crop(_noise(10, 10), _noise(20, 20)) == ()


def test_a_crop_with_a_different_channel_count_yields_no_placement():
    source = _noise(60, 60)
    crop = source[0:20, 0:20, 0]

    assert locate_crop(source, crop) == ()


# --- ambiguity must stay visible --------------------------------------------


def test_every_exact_placement_is_returned_so_ambiguity_is_not_hidden():
    source = np.full((40, 90, 3), 255, dtype=np.uint8)
    crop = np.full((5, 5, 3), 255, dtype=np.uint8)

    placements = locate_crop(source, crop, orientations=(CropOrientation.ROTATE_0,))

    assert len(placements) == (40 - 5 + 1) * (90 - 5 + 1)


def test_placements_come_back_in_a_stable_reading_order():
    source = np.full((12, 12, 3), 7, dtype=np.uint8)
    crop = np.full((2, 2, 3), 7, dtype=np.uint8)

    boxes = [p.box for p in locate_crop(source, crop, orientations=(CropOrientation.ROTATE_0,))]

    assert boxes == sorted(boxes, key=lambda b: (b[1], b[0]))


# --- apply_placement ---------------------------------------------------------


def test_apply_placement_rejects_a_box_outside_the_source():
    source = _noise(20, 20)
    placement = CropPlacement(box=(0, 0, 30, 10), orientation=CropOrientation.ROTATE_0)

    with pytest.raises(ValueError, match="outside"):
        apply_placement(source, placement)
