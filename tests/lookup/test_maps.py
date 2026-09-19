from jev_plays_pokemon.lookup.maps import MAP_COUNT, is_unused_map, map_name


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


def test_map_count_is_the_first_unknown_map_id():
    assert map_name(MAP_COUNT - 1) == "Agathas Room"  # last known map ID
    assert map_name(MAP_COUNT) == f"Unknown Map ({MAP_COUNT})"


def test_is_unused_map_flags_placeholder_slots():
    assert is_unused_map(0x0B) is True  # UNUSED_MAP_0B
    assert is_unused_map(0x00) is False  # Pallet Town
    assert is_unused_map(0x0C) is False  # Route 1


def test_is_unused_map_treats_out_of_range_ids_as_unused():
    assert is_unused_map(MAP_COUNT) is True
    assert is_unused_map(-1) is True
