from jev_plays_pokemon.lookup.species import species_name


def test_known_species_returns_display_name():
    assert species_name(0x01) == "RHYDON"
    assert species_name(0xB0) == "CHARMANDER"


def test_missingno_slot_returns_its_documented_name():
    assert species_name(0x1F) == "MISSINGNO."


def test_unknown_species_id_does_not_raise():
    assert species_name(0x00) == "Unknown Species (0)"


def test_out_of_range_species_id_does_not_raise():
    assert species_name(9999) == "Unknown Species (9999)"
    assert species_name(-1) == "Unknown Species (-1)"
