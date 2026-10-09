"""The nc dataset card's licence (review M3, design §7.88.1): derived from the
rows, never hard-coded. One HF licence id when every row shares one that HF
knows; otherwise ``license: other`` with a name, and the body lists every
licence present."""

from real_chart_bench.usecase.dataset_card import (
    card_licence_metadata,
    render_nc_dataset_card,
)


def _rows(*licences):
    return [{"paper_id": str(i), "license": lic} for i, lic in enumerate(licences)]


def test_single_versioned_nc_licence_uses_its_hf_id():
    assert card_licence_metadata(_rows("cc-by-nc-4.0", "CC BY-NC 4.0")) == {
        "license": "cc-by-nc-4.0"
    }


def test_single_nc_sa_licence_uses_the_sa_id():
    assert card_licence_metadata(_rows("cc-by-nc-sa-4.0")) == {"license": "cc-by-nc-sa-4.0"}


def test_mixed_nc_and_nc_sa_is_other_with_a_name():
    meta = card_licence_metadata(_rows("cc-by-nc-4.0", "cc-by-nc-sa-4.0"))
    assert meta["license"] == "other"
    assert meta["license_name"]


def test_versionless_licence_has_no_hf_id_so_other():
    # Unpaywall/OpenAlex usually report "cc-by-nc" without a version
    assert card_licence_metadata(_rows("cc-by-nc"))["license"] == "other"


def test_card_front_matter_and_body_follow_the_rows():
    card = render_nc_dataset_card(_rows("cc-by-nc", "cc-by-nc", "cc-by-nc-sa"))
    front = card.split("---")[1]
    assert "license: other" in front
    assert "cc-by-nc-4.0" not in card
    assert "`cc-by-nc` (2 papers)" in card
    assert "`cc-by-nc-sa` (1 paper)" in card
    assert "ShareAlike" in card
    assert "non-commercial" in card.lower()


def test_card_without_sa_rows_does_not_mention_sharealike_terms():
    card = render_nc_dataset_card(_rows("cc-by-nc-4.0"))
    assert "license: cc-by-nc-4.0" in card
    assert "ShareAlike" not in card
