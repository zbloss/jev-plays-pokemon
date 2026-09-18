from jev_plays_pokemon.lookup.items import item_name


def test_known_item_returns_display_name():
    assert item_name(0x01) == "MASTER BALL"
    assert item_name(0x04) == "POKé BALL"


def test_known_badge_returns_display_name():
    assert item_name(0x15) == "BOULDERBADGE"


def test_known_hm_and_tm_return_generated_display_name():
    assert item_name(0xC4) == "HM01"
    assert item_name(0xC8) == "HM05"
    assert item_name(0xC9) == "TM01"
    assert item_name(0xFA) == "TM50"


def test_unknown_item_id_does_not_raise():
    assert item_name(0x62) == "Unknown Item (98)"


def test_out_of_range_item_id_does_not_raise():
    assert item_name(9999) == "Unknown Item (9999)"
    assert item_name(-1) == "Unknown Item (-1)"
