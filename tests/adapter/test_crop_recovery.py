"""TDD for adapter/crop_recovery.py -- the file-level half of 第0段: decode a
committed crop and its candidate source, locate the box, then re-cut the
source and check the result against the committed file *byte for byte*
(figure-fetch-distribution §9: "1件でも落ちたら本設計は撤回").

Tests write real PNG/JPEG files through Pillow rather than mocking, because
the whole question is what the bytes on disk are.
"""

import hashlib

import numpy as np
import pytest
from PIL import Image

from real_chart_bench.adapter.crop_recovery import (
    CROP_ENCODER,
    recover_crop,
    reproduce_crop,
    sha256_file,
)
from real_chart_bench.domain.crop_recovery import CropOrientation
from real_chart_bench.domain.verified_pairing import CropRecipe

RNG = np.random.default_rng(7)


def _write_source(path, array):
    Image.fromarray(array).save(path)
    return path


def _write_crop(path, array):
    """The canonical crop procedure: Pillow's default PNG encoder, which is
    what produced the crops re-cut on 2026-09-30."""
    Image.fromarray(array).save(path, format="PNG")
    return path


@pytest.fixture
def source_array():
    return RNG.integers(0, 256, size=(120, 200, 3), dtype=np.uint8)


# --- the match case ----------------------------------------------------------


def test_recovers_the_box_of_a_crop_cut_from_the_source(tmp_path, source_array):
    source = _write_source(tmp_path / "src.png", source_array)
    crop = _write_crop(tmp_path / "crop.png", source_array[20:90, 30:150])

    result = recover_crop(source, crop)

    assert result.recipe is not None
    assert result.recipe.box == (30, 20, 150, 90)
    assert result.recipe.rotation_deg == 0
    assert result.recipe.source_image_path == str(source)


def test_a_recovered_crop_reports_identical_pixels_and_identical_bytes(tmp_path, source_array):
    source = _write_source(tmp_path / "src.png", source_array)
    crop = _write_crop(tmp_path / "crop.png", source_array[0:40, 0:60])

    result = recover_crop(source, crop)

    assert result.pixels_identical
    assert result.bytes_identical
    assert result.is_accepted
    assert result.final_sha256 == sha256_file(crop)
    assert result.reencoded_sha256 == result.final_sha256
    assert result.failure_reason is None


def test_the_encoder_whose_output_byte_identity_is_judged_against_is_named():
    assert "pillow" in CROP_ENCODER.lower()


def test_recovers_a_crop_that_was_rotated_a_quarter_turn(tmp_path, source_array):
    source = _write_source(tmp_path / "src.png", source_array)
    crop = _write_crop(tmp_path / "crop.png", np.rot90(source_array[10:60, 20:120], k=1))

    result = recover_crop(source, crop)

    assert result.recipe.box == (20, 10, 120, 60)
    assert result.recipe.rotation_deg == 90
    assert result.bytes_identical


def test_recovers_a_greyscale_crop_without_promoting_it_to_rgb(tmp_path):
    grey = RNG.integers(0, 256, size=(80, 90), dtype=np.uint8)
    source = _write_source(tmp_path / "src.png", grey)
    crop = _write_crop(tmp_path / "crop.png", grey[5:45, 10:70])

    result = recover_crop(source, crop)

    assert result.recipe.box == (10, 5, 70, 45)
    assert result.bytes_identical


def test_a_crop_cut_from_a_jpeg_source_is_recovered_from_the_decoded_pixels(tmp_path):
    """The real sources are JPEGs extracted from PDFs; the crops are PNGs of
    the decoded pixels, so the search runs on decoded arrays, not on the
    compressed streams."""
    array = RNG.integers(0, 256, size=(100, 140, 3), dtype=np.uint8)
    jpeg = tmp_path / "src.jpg"
    Image.fromarray(array).save(jpeg, quality=95)
    decoded = np.asarray(Image.open(jpeg).convert("RGB"))
    crop = _write_crop(tmp_path / "crop.png", decoded[10:70, 20:100])

    result = recover_crop(jpeg, crop)

    assert result.recipe.box == (20, 10, 100, 70)
    assert result.bytes_identical


# --- the no-match case -------------------------------------------------------


def test_a_crop_that_is_not_in_the_source_is_reported_as_not_found(tmp_path, source_array):
    source = _write_source(tmp_path / "src.png", source_array)
    other = RNG.integers(0, 256, size=(30, 40, 3), dtype=np.uint8)
    crop = _write_crop(tmp_path / "crop.png", other)

    result = recover_crop(source, crop)

    assert result.recipe is None
    assert not result.pixels_identical
    assert not result.bytes_identical
    assert not result.is_accepted
    assert result.failure_reason == "no_exact_placement"
    assert result.final_sha256 == sha256_file(crop)


def test_a_crop_larger_than_the_candidate_source_is_reported_as_not_found(tmp_path):
    source = _write_source(tmp_path / "src.png", RNG.integers(0, 256, (20, 20, 3), dtype=np.uint8))
    crop = _write_crop(tmp_path / "crop.png", RNG.integers(0, 256, (40, 40, 3), dtype=np.uint8))

    result = recover_crop(source, crop)

    assert result.recipe is None
    assert result.failure_reason == "no_exact_placement"


def test_a_mirrored_crop_is_found_but_refused_as_unrecordable(tmp_path, source_array):
    """§2.4's orientation fixes include vertical mirrors (papers 83, 5902),
    which are not rotations -- the recipe vocabulary of §3.2 cannot express
    them, so they are reported, never recorded."""
    source = _write_source(tmp_path / "src.png", source_array)
    crop = _write_crop(tmp_path / "crop.png", source_array[10:60, 20:120][::-1])

    result = recover_crop(source, crop)

    assert result.recipe is None
    assert result.failure_reason == "orientation_not_recordable"
    assert result.orientation is CropOrientation.MIRROR_VERTICAL
    assert result.box == (20, 10, 120, 60)


def test_an_ambiguous_crop_is_refused_rather_than_guessed(tmp_path):
    flat = np.full((40, 40, 3), 255, dtype=np.uint8)
    source = _write_source(tmp_path / "src.png", flat)
    crop = _write_crop(tmp_path / "crop.png", flat[0:4, 0:4])

    result = recover_crop(source, crop)

    assert result.recipe is None
    assert result.failure_reason == "ambiguous_placement"
    assert result.placement_count > 1


def test_pixels_can_match_exactly_while_the_file_bytes_do_not(tmp_path, source_array):
    """The committed crops written on 2026-08-30 decode bit-identically but
    their deflate streams come from a different zlib build. The box is
    exact; the file bytes are not reproducible here. Phase 0's bar is byte
    identity, so this is a failure and must be reported as one."""
    source = _write_source(tmp_path / "src.png", source_array)
    region = source_array[0:50, 0:60]
    crop = tmp_path / "crop.png"
    Image.fromarray(region).save(crop, format="PNG", compress_level=1)

    result = recover_crop(source, crop)

    assert result.recipe is None
    assert result.pixels_identical
    assert not result.bytes_identical
    assert not result.is_accepted
    assert result.failure_reason == "bytes_differ"
    assert result.box == (0, 0, 60, 50)
    assert result.reencoded_sha256 != result.final_sha256


# --- executing a recorded recipe (the forward direction) ---------------------


def test_executing_a_recovered_recipe_reproduces_the_crops_hash(tmp_path, source_array):
    source = _write_source(tmp_path / "src.png", source_array)
    crop = _write_crop(tmp_path / "crop.png", source_array[15:95, 25:175])

    recipe = recover_crop(source, crop).recipe

    assert reproduce_crop(source, recipe) == sha256_file(crop)


def test_executing_a_rotated_recipe_applies_the_quarter_turn(tmp_path, source_array):
    source = _write_source(tmp_path / "src.png", source_array)
    crop = _write_crop(tmp_path / "crop.png", np.rot90(source_array[0:40, 0:70], k=3))

    recipe = CropRecipe(source_image_path=str(source), box=(0, 0, 70, 40), rotation_deg=270)

    assert reproduce_crop(source, recipe) == sha256_file(crop)


def test_executing_a_recipe_with_the_wrong_box_gives_a_different_hash(tmp_path, source_array):
    source = _write_source(tmp_path / "src.png", source_array)
    crop = _write_crop(tmp_path / "crop.png", source_array[15:95, 25:175])
    off_by_one = CropRecipe(source_image_path=str(source), box=(25, 16, 175, 96))

    assert reproduce_crop(source, off_by_one) != sha256_file(crop)


# --- hashing -----------------------------------------------------------------


def test_sha256_file_hashes_the_bytes_on_disk(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc")

    assert sha256_file(path) == hashlib.sha256(b"abc").hexdigest()
