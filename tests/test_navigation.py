import io
from pathlib import Path

import numpy as np
import pytest
from pyboy import PyBoy

from jev_plays_pokemon.emulator import boot_past_intro
from jev_plays_pokemon.game_state import extract_game_state
from jev_plays_pokemon.milestones import Milestone, MilestoneTarget
from jev_plays_pokemon.navigation import (
    RAW_BUTTONS,
    NavigationTarget,
    execute_button,
    execute_navigation_macro,
    main_battle_menu_delta_buttons,
    menu_list_delta_buttons,
    read_menu_cursor,
    resolve_navigation_target,
)

ROM_PATH = Path(__file__).resolve().parent.parent / "pokemon_red.gb"

# `pokemon_red.gb` is gitignored (see test_game_state.py) - these tests run
# for real against PyBoy + the ROM wherever it's present, and skip where
# it's not.
pytestmark = pytest.mark.skipif(
    not ROM_PATH.exists(), reason=f"{ROM_PATH} not present locally"
)


def _position(pyboy: PyBoy) -> tuple[int, int, int]:
    memory = pyboy.memory
    return (memory[0xD35E], memory[0xD362], memory[0xD361])


_HOUSE_EXIT_PATH: tuple[str, ...] = (
    "right",
    "up",
    "up",
    "up",
    "up",
    "up",
    "right",
    "right",
    "right",
    "down",
    "down",
    "down",
    "down",
    "down",
    "left",
    "left",
    "left",
    "left",
    "down",
)


def _walk_out_of_the_house(pyboy: PyBoy) -> None:
    """From the bedroom (see `_boot_past_intro`), walk downstairs and out of
    the house into Pallet Town.

    This is needed because PyBoy's Gen1 `game_area_collision()` (see
    `navigation.py`'s docstring) only resolves real walkable/blocked data
    outdoors - every indoor tileset this ticket checked against the real
    ROM (Red's House 1F and 2F) reads back as entirely blocked, which is
    degenerate for the navigation macro's pathfinding tests. `_HOUSE_EXIT_PATH`
    is a fixed sequence of `execute_button` calls (one real tile - or, on
    the stairs, one map transition - per call) found by exploring the real
    ROM from the bedroom out to Pallet Town.
    """
    for direction in _HOUSE_EXIT_PATH:
        execute_button(pyboy, direction)


@pytest.fixture(scope="module")
def bedroom_state() -> bytes:
    pyboy = PyBoy(str(ROM_PATH), window="null")
    pyboy.set_emulation_speed(0)
    boot_past_intro(pyboy)
    buf = io.BytesIO()
    pyboy.save_state(buf)
    pyboy.stop(save=False)
    return buf.getvalue()


@pytest.fixture(scope="module")
def outdoors_state(bedroom_state: bytes) -> bytes:
    pyboy = PyBoy(str(ROM_PATH), window="null")
    pyboy.set_emulation_speed(0)
    pyboy.load_state(io.BytesIO(bedroom_state))
    pyboy.tick(1, False)
    _walk_out_of_the_house(pyboy)
    buf = io.BytesIO()
    pyboy.save_state(buf)
    pyboy.stop(save=False)
    return buf.getvalue()


@pytest.fixture
def pyboy_in_bedroom(bedroom_state: bytes):
    pyboy = PyBoy(str(ROM_PATH), window="null")
    pyboy.set_emulation_speed(0)
    pyboy.load_state(io.BytesIO(bedroom_state))
    pyboy.tick(1, False)
    yield pyboy
    pyboy.stop(save=False)


@pytest.fixture
def pyboy_outdoors(outdoors_state: bytes):
    pyboy = PyBoy(str(ROM_PATH), window="null")
    pyboy.set_emulation_speed(0)
    pyboy.load_state(io.BytesIO(outdoors_state))
    pyboy.tick(1, False)
    yield pyboy
    pyboy.stop(save=False)


def test_raw_button_press_moves_the_player_in_the_pressed_direction(pyboy_in_bedroom):
    before = _position(pyboy_in_bedroom)

    execute_button(pyboy_in_bedroom, "down")

    assert _position(pyboy_in_bedroom) == (before[0], before[1], before[2] + 1)


def test_raw_button_press_on_a_different_axis_moves_that_axis(pyboy_in_bedroom):
    before = _position(pyboy_in_bedroom)

    execute_button(pyboy_in_bedroom, "right")

    assert _position(pyboy_in_bedroom) == (before[0], before[1] + 1, before[2])


def test_raw_button_press_blocked_by_a_wall_does_not_move_the_player(pyboy_in_bedroom):
    before = _position(pyboy_in_bedroom)

    execute_button(
        pyboy_in_bedroom, "up"
    )  # a wall sits north of the bedroom spawn tile

    assert _position(pyboy_in_bedroom) == before


def test_execute_button_rejects_an_unknown_button(pyboy_in_bedroom):
    with pytest.raises(ValueError):
        execute_button(pyboy_in_bedroom, "not-a-button")


def test_execute_button_leaves_the_screen_buffer_rendered(pyboy_in_bedroom):
    # #47: execute_button's ticks render=True now, so the live per-turn loop
    # never leaves screen.ndarray stale (a uniform blank frame) between turns,
    # unlike a render=False tick (see test_emulator.py's stale-buffer test).
    pyboy_in_bedroom.tick(1, False)
    assert float(np.std(pyboy_in_bedroom.screen.ndarray)) == 0.0

    execute_button(pyboy_in_bedroom, "down")

    assert float(np.std(pyboy_in_bedroom.screen.ndarray)) > 0.0


def test_raw_buttons_cover_the_four_directions_and_the_face_buttons():
    assert set(RAW_BUTTONS) == {
        "up",
        "down",
        "left",
        "right",
        "a",
        "b",
        "start",
        "select",
    }


def test_player_grid_anchor_is_always_the_players_own_walkable_tile(pyboy_outdoors):
    """Documents/verifies `navigation.py`'s `_PLAYER_GRID_COL`/`_PLAYER_GRID_ROW`
    constant against the real ROM: whatever tile the player is standing on
    must itself be walkable, so the collision grid's cell at that fixed
    anchor should read as walkable at a real, live outdoor position."""
    collision = pyboy_outdoors.game_area_collision()

    assert collision[8, 8] != 0


def test_navigation_macro_walks_the_player_toward_a_reachable_target(pyboy_outdoors):
    before = extract_game_state(pyboy_outdoors)
    target = NavigationTarget(
        map_id=before.map_id, x=before.player_x + 3, y=before.player_y
    )

    moved = execute_navigation_macro(pyboy_outdoors, target)

    after = extract_game_state(pyboy_outdoors)
    assert moved is True
    assert after.map_id == before.map_id
    assert after.player_x > before.player_x
    assert (after.player_x, after.player_y) != (before.player_x, before.player_y)


def test_navigation_macro_can_reach_a_nearby_target_exactly(pyboy_outdoors):
    before = extract_game_state(pyboy_outdoors)
    target = NavigationTarget(
        map_id=before.map_id, x=before.player_x + 2, y=before.player_y
    )

    execute_navigation_macro(pyboy_outdoors, target)

    after = extract_game_state(pyboy_outdoors)
    assert (after.player_x, after.player_y) == (target.x, target.y)


def test_navigation_macro_is_a_noop_when_player_is_on_a_different_map(pyboy_outdoors):
    before = extract_game_state(pyboy_outdoors)
    target = NavigationTarget(
        map_id=before.map_id + 1, x=before.player_x, y=before.player_y
    )

    moved = execute_navigation_macro(pyboy_outdoors, target)

    after = extract_game_state(pyboy_outdoors)
    assert moved is False
    assert (after.player_x, after.player_y) == (before.player_x, before.player_y)


def test_navigation_macro_reaches_a_milestones_target_end_to_end(pyboy_outdoors):
    """Chains `resolve_navigation_target` and `execute_navigation_macro`
    together end to end - the wiring #20 asks for ("reads its target from
    the current-objective module... and walks the player toward it") -
    against a milestone carrying a real, ROM-verified destination.

    No scripted milestone in `milestones.py` has verified tile coordinates
    yet (see that module's docstring), so this uses a fixture `Milestone`
    pointed at this fixture's own live, reachable position instead of a
    real story milestone; the point is proving the two functions work
    together, not exercising the scripted list itself (that's covered by
    `tests/test_milestones.py`).
    """
    before = extract_game_state(pyboy_outdoors)
    milestone = Milestone(
        milestone_id="test_milestone",
        description="a milestone used only by this test",
        target=MilestoneTarget(
            map_id=before.map_id,
            map_name=before.map_name,
            target_x=before.player_x + 2,
            target_y=before.player_y,
        ),
    )

    target = resolve_navigation_target(milestone)
    assert target is not None
    moved = execute_navigation_macro(pyboy_outdoors, target)

    after = extract_game_state(pyboy_outdoors)
    assert moved is True
    assert (after.player_x, after.player_y) == (target.x, target.y)


_TO_OAKS_LAB_TABLE_PATH: tuple[str, ...] = (
    "down",
    "down",
    "down",
    "down",
    "right",
    "right",
    "right",
    "down",
    "down",
    "down",
    "right",
    "up",
    "down",
    "down",
    "right",
    "right",
    "right",
    "up",
    "down",
    "up",
    "up",
    "up",
    "up",
    "up",
    "up",
    "right",
    "right",
    "up",
)


def test_walking_to_oaks_lab_starter_table_reaches_a_rom_verified_tile(pyboy_outdoors):
    """#90's worked example of the repeatable ROM-verification method (see
    `docs/research/rom-verification-method.md`): a fixed button sequence
    from the outdoor Pallet Town spawn - built and re-verified the same way
    `_HOUSE_EXIT_PATH` was (screenshotting each step against the real ROM,
    see the linked doc) - walks the player through Oak's Lab's front door
    up to the starter Poke Ball table, landing on the one tile that's
    actually interactive: pressing "a" there opens a real dialog box
    (`GameState.dialog_open`), not just a tile that's visually adjacent to
    the table. This is `milestones.py`'s "got_starter" milestone's real,
    ROM-verified `target_x`/`target_y` (Oak's Lab, map 40) - wiring it into
    that milestone's entry is ordinary #84 follow-up work, not done here.
    """
    for direction in _TO_OAKS_LAB_TABLE_PATH:
        execute_button(pyboy_outdoors, direction)

    assert _position(pyboy_outdoors) == (40, 7, 4)

    pyboy_outdoors.button("a", 2)
    dialog_opened = False
    for _ in range(10):
        pyboy_outdoors.tick(30, True)
        if extract_game_state(pyboy_outdoors).dialog_open:
            dialog_opened = True
            break
    assert dialog_opened


# #99: per-milestone boot verification, batch 1 (parent #96). Unlike
# got_starter's single fixed button tuple above, these targets sit past
# Pallet Town, so reaching them drives the navigation macro itself
# (`_walk_toward`) rather than hand-recording a raw button sequence - the
# on-screen A* already re-plans around real obstacles each step, and a
# fixed tuple would be far more brittle over this much distance.

_EVENT_FLAGS_START_ADDRESS = 0xD747  # matches game_state.py's own constant
_EVENT_FOLLOWED_OAK_INTO_LAB_BIT = 0  # pret/pokered's event_constants.asm
_GRASS_RATE_ADDRESS = 0xD887  # wGrassRate, right after wEventFlags's 320
# bytes (0xD747 + flag_array(2560 events) == 0xD887) - re-derives the
# already-verified 0xD747/0xD886 pair from a different anchor, cross-
# checking the byte-counted offset chain below it.
_WATER_RATE_ADDRESS = 0xD8A4  # wGrassRate(1) + wGrassMons(20) + `ds 8`
_VIRIDIAN_MART_CUR_SCRIPT_ADDRESS = 0xD60D  # wViridianMartCurScript; byte-
# counted the same way from wCurMap (0xD35E, ram/wram.asm's "Main Data"
# section) through to ram/wram.asm's "Game Progress Flags" CurScript
# block - the chain reproduces wObtainedBadges (0xD356) and wEventFlags
# (0xD747) exactly, both already independently verified in this repo.
_SCRIPT_VIRIDIANMART_NOOP = 2

_DIALOG_TEXT_ROW = 14
_DIALOG_TEXT_BLANK_TILE = 383
_DIALOG_TEXT_COLUMNS = range(1, 19)


def _bypass_oaks_route_1_interception(pyboy: PyBoy) -> None:
    """Sets pret/pokered's EVENT_FOLLOWED_OAK_INTO_LAB directly rather than
    playing through Oak's mandatory Route 1 interception scripts/
    PalletTown.asm's `PalletTownDefaultScript` gates the entire scene on
    it. Without this flag (or actually playing the scene, as
    `test_walking_to_oaks_lab_oak1_reaches_a_rom_verified_tile` below
    does, since that milestone needs the scene's side effects), Oak
    physically blocks Route 1 - no milestone past Pallet Town is
    reachable at all - which is exactly the kind of prerequisite story
    state this issue says to set directly rather than play through.
    """
    pyboy.memory[_EVENT_FLAGS_START_ADDRESS] |= 1 << _EVENT_FOLLOWED_OAK_INTO_LAB_BIT


def _disable_wild_encounters(pyboy: PyBoy) -> None:
    """Zeroes wGrassRate/wWaterRate: pret/pokered's
    `engine/battle/wild_encounters.asm` (`TryDoWildEncounter`) only starts
    a wild battle if a random byte comes up less than this rate, so 0
    means never. The engine reloads each map's own real rate on every
    map transition, so this needs reapplying after crossing into a new
    map, not just once. A random wild encounter partway through one of
    these walks isn't part of what the test below is checking, and would
    turn an otherwise-deterministic route flaky.
    """
    pyboy.memory[_GRASS_RATE_ADDRESS] = 0
    pyboy.memory[_WATER_RATE_ADDRESS] = 0


def _walk_toward(pyboy: PyBoy, x: int, y: int, max_calls: int = 20) -> None:
    """Drives the navigation macro toward `(x, y)` on the player's current
    map, calling `execute_navigation_macro` repeatedly (it only takes one
    step, or a short run, per call) until it arrives, stops making
    progress (a real obstacle its on-screen A* can't route around, or a
    map transition happened mid-step), or `max_calls` is exhausted.
    """
    for _ in range(max_calls):
        state = extract_game_state(pyboy)
        if (state.player_x, state.player_y) == (x, y):
            return
        target = NavigationTarget(map_id=state.map_id, x=x, y=y)
        if not execute_navigation_macro(pyboy, target):
            return


def _cross_map_edge(pyboy: PyBoy, direction: str, max_presses: int = 5) -> None:
    """Presses `direction` until the player's `map_id` changes - crossing a
    map-edge connection or walking through a door - then settles a few
    extra frames (see `rom-verification-method.md`'s post-warp gotcha: a
    position read immediately after a transition can be stale) and
    reapplies `_disable_wild_encounters` (the new map just reloaded its
    own real rate).
    """
    start_map = extract_game_state(pyboy).map_id
    for _ in range(max_presses):
        execute_button(pyboy, direction)
        if extract_game_state(pyboy).map_id != start_map:
            break
    pyboy.tick(90, True)
    _disable_wild_encounters(pyboy)


def _dialog_text_visible(pyboy: PyBoy) -> bool:
    """A broader "is a dialog box actually showing text" check than
    `GameState.dialog_open`'s continuation-arrow test. Verified (frame-by-
    frame tilemap scan across 400 real frames) that Gen 1's plain,
    single-page `<DONE>`-terminated NPC lines - e.g. Viridian Mart's
    COOLTRAINER_M, "No! POTIONs are all sold out." - never draw the
    arrow tile `dialog_open` looks for at all, even though the box still
    doesn't close until a button is pressed; only longer/multi-page
    dialogs (like the starter table's, above) do. This instead checks
    the dialog text row for anything other than the blank tile, which
    both styles share, so it still confirms interactivity for a tile
    whose real text happens to be a one-page remark.
    """
    tilemap = pyboy.tilemap_window
    return any(
        tilemap[col, _DIALOG_TEXT_ROW] != _DIALOG_TEXT_BLANK_TILE
        for col in _DIALOG_TEXT_COLUMNS
    )


def test_walking_to_oaks_lab_oak1_reaches_a_rom_verified_tile(pyboy_outdoors):
    """#99's boot verification for the `got_pokedex` milestone: Oak's Lab's
    OAKSLAB_OAK1 object (`milestone_targets.py`), map 40 tile (5, 2).

    OAK1 only becomes a real, present sprite once Oak's own Route 1
    interception scene has played out (`pret/pokered`'s `OaksLab.asm`:
    `OaksLabDefaultScript` gates showing him at all on
    `EVENT_OAK_APPEARED_IN_PALLET`, which that scene sets) - unlike
    `got_oaks_parcel`'s test below, this milestone's own target tile
    depends on that scene's side effects, so it's played out for real
    here (walking toward Route 1's trigger tile is enough; Oak takes
    over from there) rather than bypassed.
    """
    _walk_toward(pyboy_outdoors, 10, 1)
    pyboy_outdoors.tick(60, True)
    for _ in range(15):
        execute_button(pyboy_outdoors, "a")
    pyboy_outdoors.tick(90, True)
    for _ in range(15):
        execute_button(pyboy_outdoors, "a")
    pyboy_outdoors.tick(90, True)
    for _ in range(50):
        execute_button(pyboy_outdoors, "a")

    assert extract_game_state(pyboy_outdoors).map_id == 40  # Oak's Lab

    _walk_toward(pyboy_outdoors, 5, 2)

    pyboy_outdoors.button("a", 2)
    dialog_opened = False
    for _ in range(10):
        pyboy_outdoors.tick(30, True)
        if extract_game_state(pyboy_outdoors).dialog_open:
            dialog_opened = True
            break
    assert dialog_opened


_TO_VIRIDIAN_MART_DOOR_PATH: tuple[str, ...] = ("down", "right", "right", "up")


def test_walking_to_viridian_mart_cooltrainer_reaches_a_rom_verified_tile(pyboy_outdoors):
    """#99's boot verification for the `got_oaks_parcel` milestone: Viridian
    Mart's VIRIDIANMART_COOLTRAINER_M object (`milestone_targets.py`), map
    42 tile (3, 3).

    Two pieces of prerequisite story state are set directly rather than
    played through, per this issue's own guidance:
    - `_bypass_oaks_route_1_interception` - this milestone has nothing to
      do with Oak's scene (unlike `got_pokedex`, above).
    - `wViridianMartCurScript` set to its own NOOP step - the mart's
      script (`pret/pokered`'s `ViridianMart.asm`) otherwise auto-fires
      an unskippable "you came from Pallet Town... here, take OAK's
      PARCEL" cutscene the instant the map loads, regardless of which
      tile the player ever stands on or interacts with - independent of
      what this test is checking (COOLTRAINER_M's own interactivity).
    """
    _bypass_oaks_route_1_interception(pyboy_outdoors)
    _disable_wild_encounters(pyboy_outdoors)
    pyboy_outdoors.memory[_VIRIDIAN_MART_CUR_SCRIPT_ADDRESS] = _SCRIPT_VIRIDIANMART_NOOP

    _walk_toward(pyboy_outdoors, 10, 1)
    _cross_map_edge(pyboy_outdoors, "up")
    assert extract_game_state(pyboy_outdoors).map_id == 12  # Route 1

    _walk_toward(pyboy_outdoors, 12, 0)
    _cross_map_edge(pyboy_outdoors, "up")
    assert extract_game_state(pyboy_outdoors).map_id == 1  # Viridian City

    _walk_toward(pyboy_outdoors, 29, 19)
    for direction in _TO_VIRIDIAN_MART_DOOR_PATH:
        execute_button(pyboy_outdoors, direction)
    pyboy_outdoors.tick(90, True)

    assert extract_game_state(pyboy_outdoors).map_id == 42  # Viridian Mart

    _walk_toward(pyboy_outdoors, 3, 3)

    pyboy_outdoors.button("a", 2)
    dialog_visible = False
    for _ in range(10):
        pyboy_outdoors.tick(30, True)
        if _dialog_text_visible(pyboy_outdoors):
            dialog_visible = True
            break
    assert dialog_visible


def _milestone(target_x: int | None = None, target_y: int | None = None) -> Milestone:
    return Milestone(
        milestone_id="test_milestone",
        description="a milestone used only by this test",
        target=MilestoneTarget(
            map_id=40, map_name="Oaks Lab", target_x=target_x, target_y=target_y
        ),
    )


def test_resolve_navigation_target_returns_none_when_there_is_no_current_milestone():
    assert resolve_navigation_target(None) is None


def test_resolve_navigation_target_returns_none_without_verified_tile_coordinates():
    assert resolve_navigation_target(_milestone()) is None


def test_resolve_navigation_target_returns_the_milestones_tile_coordinates():
    milestone = _milestone(target_x=4, target_y=5)

    target = resolve_navigation_target(milestone)

    assert target == NavigationTarget(map_id=40, x=4, y=5)


# #76: battle-menu cursor delta helpers - pure, no PyBoy required.


def test_menu_list_delta_buttons_presses_down_for_a_positive_delta():
    assert menu_list_delta_buttons(0, 3) == ("down", "down", "down")


def test_menu_list_delta_buttons_presses_up_for_a_negative_delta():
    assert menu_list_delta_buttons(3, 1) == ("up", "up")


def test_menu_list_delta_buttons_is_empty_when_already_at_the_target():
    assert menu_list_delta_buttons(2, 2) == ()


def test_menu_list_delta_buttons_never_hardcodes_a_fixed_sequence_across_turns():
    """#76's whole point: the same target reached from two different starting
    cursor positions (e.g. wherever a previous turn left it) presses a
    different number of buttons, since it's a live delta, not a fixed one."""
    turn_one = menu_list_delta_buttons(0, 2)
    turn_two = menu_list_delta_buttons(2, 2)
    turn_three = menu_list_delta_buttons(3, 2)

    assert turn_one == ("down", "down")
    assert turn_two == ()
    assert turn_three == ("up",)


def test_main_battle_menu_delta_buttons_toggles_column_only():
    # FIGHT(0) -> PKMN(1): same row, one column over.
    assert main_battle_menu_delta_buttons(0, 1) == ("right",)
    assert main_battle_menu_delta_buttons(1, 0) == ("left",)


def test_main_battle_menu_delta_buttons_toggles_row_only():
    # FIGHT(0) -> ITEM(2): same column, one row down.
    assert main_battle_menu_delta_buttons(0, 2) == ("down",)
    assert main_battle_menu_delta_buttons(2, 0) == ("up",)


def test_main_battle_menu_delta_buttons_toggles_both_axes_for_the_diagonal():
    # FIGHT(0) -> RUN(3): opposite corner.
    assert main_battle_menu_delta_buttons(0, 3) == ("right", "down")
    assert main_battle_menu_delta_buttons(3, 0) == ("left", "up")


def test_main_battle_menu_delta_buttons_is_empty_when_already_at_the_target():
    assert main_battle_menu_delta_buttons(1, 1) == ()


def test_read_menu_cursor_reads_the_live_wcurrentmenuitem_byte(pyboy_in_bedroom):
    pyboy_in_bedroom.memory[0xCC26] = 2

    assert read_menu_cursor(pyboy_in_bedroom) == 2
