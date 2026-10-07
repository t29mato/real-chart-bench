"""TDD for the crop recipe on VerifiedPairing (scaling-verification 第0段 /
figure-fetch-distribution §3.2, §3.5).

``crop`` + ``final_sha256`` together make the image handed to the scorer
byte-determined: without them, a hand-made crop under
data/verified_pairs/crops/ cannot be regenerated, and the pixel
coordinates in tick_calibration.json -- which main condition 2 is scored
against -- have nothing reproducible to be defined on.

The pair is both-or-neither, the same discipline as y2_range/y2_scale
(design §7.84): a box with no hash is a claim nobody can check, and a
hash with no box says the bytes matter but not how to get them.
"""

import pytest

from real_chart_bench.domain.verified_pairing import (
    CropRecipe,
    VerificationStatus,
    VerifiedPairing,
)

SHA = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


def _base_kwargs(**overrides):
    kwargs = dict(
        paper_id="36305",
        figure_id="45818",
        image_path="data/verified_pairs/crops/36305/fig5a.png",
        panel_label="a",
        x_range=(0.0, 1.0),
        y_range=(0.0, 1.0),
        status=VerificationStatus.VERIFIED,
        verified_at="2026-10-07",
        evidence="ok",
    )
    kwargs.update(overrides)
    return kwargs


def _recipe(**overrides):
    kwargs = dict(
        source_image_path="data/verified_pairs/images/36305/p09_embedded_12.jpg",
        box=(0, 0, 1605, 1425),
        rotation_deg=0,
    )
    kwargs.update(overrides)
    return CropRecipe(**kwargs)


# --- CropRecipe --------------------------------------------------------------


def test_a_recipe_records_the_source_image_the_box_and_the_rotation():
    recipe = _recipe()

    assert recipe.source_image_path == "data/verified_pairs/images/36305/p09_embedded_12.jpg"
    assert recipe.box == (0, 0, 1605, 1425)
    assert recipe.rotation_deg == 0


def test_the_rotation_defaults_to_zero():
    recipe = CropRecipe(source_image_path="s.jpg", box=(1, 2, 3, 4))

    assert recipe.rotation_deg == 0


def test_the_recipe_exposes_the_crops_width_and_height():
    recipe = _recipe(box=(10, 20, 110, 70))

    assert recipe.width == 100
    assert recipe.height == 50


def test_a_quarter_turn_swaps_the_final_width_and_height():
    recipe = _recipe(box=(10, 20, 110, 70), rotation_deg=90)

    assert (recipe.final_width, recipe.final_height) == (50, 100)


def test_an_upright_recipe_keeps_the_box_width_and_height():
    recipe = _recipe(box=(10, 20, 110, 70), rotation_deg=180)

    assert (recipe.final_width, recipe.final_height) == (100, 50)


@pytest.mark.parametrize("box", [(10, 0, 10, 5), (0, 10, 5, 10), (10, 0, 5, 5), (0, 10, 5, 5)])
def test_an_empty_or_inverted_box_is_rejected(box):
    with pytest.raises(ValueError, match="box"):
        _recipe(box=box)


def test_a_negative_origin_is_rejected():
    with pytest.raises(ValueError, match="box"):
        _recipe(box=(-1, 0, 10, 10))


def test_a_box_that_is_not_four_integers_is_rejected():
    with pytest.raises(ValueError, match="box"):
        _recipe(box=(0, 0, 10))


def test_a_fractional_box_is_rejected():
    with pytest.raises(ValueError, match="box"):
        _recipe(box=(0.0, 0.0, 10.5, 10.0))


@pytest.mark.parametrize("rotation", [45, -90, 360, 1])
def test_a_rotation_that_is_not_a_quarter_turn_is_rejected(rotation):
    with pytest.raises(ValueError, match="rotation_deg"):
        _recipe(rotation_deg=rotation)


def test_an_empty_source_image_path_is_rejected():
    with pytest.raises(ValueError, match="source_image_path"):
        _recipe(source_image_path="")


# --- VerifiedPairing: absent (every entry in the registry before 第0段) ------


def test_the_crop_fields_default_to_absent():
    pairing = VerifiedPairing(**_base_kwargs())

    assert pairing.crop is None
    assert pairing.final_sha256 is None
    assert not pairing.is_reproducible_crop


# --- VerifiedPairing: present ------------------------------------------------


def test_a_pairing_can_carry_a_recovered_crop_and_its_hash():
    pairing = VerifiedPairing(**_base_kwargs(), crop=_recipe(), final_sha256=SHA)

    assert pairing.crop == _recipe()
    assert pairing.final_sha256 == SHA
    assert pairing.is_reproducible_crop


def test_a_whole_image_entry_carries_neither_field():
    """The 16 entries that point straight at an extracted image have no
    crop to reproduce: the extracted bytes *are* the scored image. They
    keep both fields absent rather than growing a half-filled recipe."""
    pairing = VerifiedPairing(
        **_base_kwargs(image_path="data/verified_pairs/images/4173/p03_embedded_4.jpg")
    )

    assert pairing.crop is None
    assert pairing.final_sha256 is None
    assert not pairing.is_reproducible_crop


# --- VerifiedPairing: validation (both-or-neither, as §7.84's y2 pair) ------


def test_a_crop_without_its_final_hash_is_rejected():
    with pytest.raises(ValueError, match="final_sha256"):
        VerifiedPairing(**_base_kwargs(), crop=_recipe())


def test_a_final_hash_without_a_crop_recipe_is_rejected():
    with pytest.raises(ValueError, match="crop"):
        VerifiedPairing(**_base_kwargs(), final_sha256=SHA)


def test_a_crop_on_an_entry_with_no_image_at_all_is_rejected():
    with pytest.raises(ValueError, match="image_path"):
        VerifiedPairing(
            **_base_kwargs(image_path=None),
            crop=_recipe(),
            final_sha256=SHA,
        )


def test_a_recipe_whose_source_is_the_crop_itself_is_rejected():
    with pytest.raises(ValueError, match="source_image_path"):
        VerifiedPairing(
            **_base_kwargs(),
            crop=_recipe(source_image_path="data/verified_pairs/crops/36305/fig5a.png"),
            final_sha256=SHA,
        )


@pytest.mark.parametrize("bad", ["", "not-a-hash", "E3B0" + "0" * 60, "abc123"])
def test_a_final_hash_that_is_not_lowercase_hex_sha256_is_rejected(bad):
    with pytest.raises(ValueError, match="final_sha256"):
        VerifiedPairing(**_base_kwargs(), crop=_recipe(), final_sha256=bad)
