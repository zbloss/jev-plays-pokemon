from pathlib import Path

import pytest

from jev_plays_pokemon import milestones, rom_maps
from jev_plays_pokemon.milestone_targets import (
    MILESTONE_OBJECT_SELECTIONS,
    resolve_target_coordinates,
)

ROM_PATH = Path(__file__).resolve().parent.parent / "pokemon_red.gb"

_EXPECTED_MILESTONE_IDS = {
    check.milestone.milestone_id for check in milestones._MILESTONES
}


def test_every_milestone_has_exactly_one_object_selection():
    # Pure data - no ROM needed - so this runs even without `pokemon_red.gb`
    # locally, unlike the ROM cross-check below.
    assert set(MILESTONE_OBJECT_SELECTIONS) == _EXPECTED_MILESTONE_IDS


def test_every_selection_carries_a_rationale():
    for milestone_id, selection in MILESTONE_OBJECT_SELECTIONS.items():
        assert selection.rationale.strip(), milestone_id


# `pokemon_red.gb` is gitignored (see test_game_state.py) - pure data-in/
# data-out against the ROM file, same skip pattern as
# tests/test_navigation.py/tests/test_game_state.py/tests/test_rom_maps.py.
@pytest.mark.skipif(not ROM_PATH.exists(), reason=f"{ROM_PATH} not present locally")
def test_milestone_target_literals_match_the_roms_object_records():
    """Regression guard for issue #98's hand-authored milestone -> object
    selection: re-derives every milestone's `(x, y)` from this repo's own
    `pokemon_red.gb` via `rom_maps`/`resolve_target_coordinates` and checks
    it against the literals pinned in `milestones.py`'s `_MILESTONES`, so a
    ROM re-dump or an upstream layout change fails here instead of silently
    drifting."""
    rom = rom_maps.load_rom(ROM_PATH)
    all_maps = rom_maps.parse_all_maps(rom)

    resolved = resolve_target_coordinates(all_maps)
    pinned = {
        check.milestone.milestone_id: (
            check.milestone.target.target_x,
            check.milestone.target.target_y,
        )
        for check in milestones._MILESTONES
    }
    assert resolved == pinned
