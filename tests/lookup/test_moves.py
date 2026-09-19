from jev_plays_pokemon.lookup.moves import move_name


def test_known_move_returns_display_name():
    assert move_name(1) == "POUND"
    assert move_name(85) == "THUNDERBOLT"


def test_last_move_id_returns_its_display_name():
    # Struggle (165) is the highest real move ID in Gen 1 - the full range
    # this ticket calls for is covered end to end.
    assert move_name(165) == "STRUGGLE"


def test_move_none_id_does_not_raise():
    # MOVE_NONE (0) is an empty move slot, not a real move.
    assert move_name(0) == "Unknown Move (0)"


def test_out_of_range_move_id_does_not_raise():
    assert move_name(9999) == "Unknown Move (9999)"
    assert move_name(-1) == "Unknown Move (-1)"
