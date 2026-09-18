from jev_plays_pokemon.lookup.maps import map_name


def test_known_map_returns_display_name():
    assert map_name(0x00) == "Pallet Town"
    assert map_name(0x0C) == "Route 1"


def test_known_indoor_map_title_cases_floor_suffix():
    assert map_name(0x25) == "Reds House 1F"


def test_unknown_map_id_does_not_raise():
    assert map_name(0xFF) == "Unknown Map (255)"


def test_out_of_range_map_id_does_not_raise():
    assert map_name(9999) == "Unknown Map (9999)"
    assert map_name(-1) == "Unknown Map (-1)"
