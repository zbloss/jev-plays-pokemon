import io
from dataclasses import replace
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
    _travel_graph,
    execute_button,
    execute_navigation_macro,
    main_battle_menu_delta_buttons,
    menu_list_delta_buttons,
    read_menu_cursor,
    resolve_navigation_target,
)
from jev_plays_pokemon.rom_maps import load_rom, parse_all_maps
from jev_plays_pokemon.tileset_collision import (
    crosses_blocked_pair,
    is_walkable,
    parse_tile_pair_collisions,
    parse_tileset_headers,
    raw_tile_id,
)
from jev_plays_pokemon.travel_graph import next_hop

ROM_PATH = Path(__file__).resolve().parent.parent / "pokemon_red.gb"

# `pokemon_red.gb` is gitignored (see test_game_state.py) - these tests run
# for real against PyBoy + the ROM wherever it's present, and skip where
# it's not.
pytestmark = pytest.mark.skipif(
    not ROM_PATH.exists(), reason=f"{ROM_PATH} not present locally"
)

# Map IDs, matching `travel_graph.py`'s own (`constants/map_constants.asm`
# `const_def` position) - used to exercise #102's cross-map routing against
# real, known hops in the milestone travel graph.
_MAP_PALLET_TOWN = 0
_MAP_OAKS_LAB = 40
# Silph Co 2F: one of the 7 maps #96's own "Out of Scope" section names as
# never getting a travel-graph route (its parsed ROM records disagreed with
# upstream pret/pokered during that ticket's research) - a permanently safe
# "no known route yet" example, unlike map 65 (Cerulean Gym), which #99
# brought into MILESTONE_MAP_IDS.
_MAP_NOT_YET_ROUTED = 207


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


def test_navigation_macro_is_a_noop_when_the_target_map_has_no_known_route(
    pyboy_outdoors,
):
    """#102's graceful-degradation case: a target map the travel graph
    doesn't (yet) know how to reach from here stays a no-op, exactly like
    pre-#102 cross-map behavior - unlike a target map the graph *does* know
    a route to (see `test_navigation_macro_routes_across_maps_via_a_known_hop`
    below), which is no longer a no-op."""
    before = extract_game_state(pyboy_outdoors)
    target = NavigationTarget(map_id=_MAP_NOT_YET_ROUTED, x=4, y=2)

    moved = execute_navigation_macro(pyboy_outdoors, target)

    after = extract_game_state(pyboy_outdoors)
    assert moved is False
    assert (after.player_x, after.player_y) == (before.player_x, before.player_y)


# The first 14 presses of `_TO_OAKS_LAB_TABLE_PATH` below (same fixed,
# ROM-verified sequence, from the same `pyboy_outdoors` spawn) - far enough
# from the house to clear a real Pallet Town obstacle (a fence/hedge tile
# pair) this ticket found `game_area_collision()` reports as walkable when
# the real game engine doesn't allow crossing it, a pre-existing limitation
# of PyBoy 2.2.0's exposed collision data unrelated to #102's own routing
# logic - and still short of Oak's Lab's own door, so the macro is the one
# actually crossing the map boundary below, not this fixed prefix.
_NEAR_OAKS_LAB_DOOR_PATH: tuple[str, ...] = (
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
)


def test_navigation_macro_routes_across_maps_via_a_known_hop(pyboy_outdoors):
    """#102's core acceptance criterion: booted on a map that isn't the
    current milestone's target map, running the macro across the resulting
    travel-graph route - over multiple invocations, since #102 re-derives
    the next hop fresh every call rather than committing to one upfront
    route - ends the player up on the target map, at or closer to the
    milestone's own tile than wherever they first land there.

    A deliberately small `max_steps` (2, versus the real default of 128)
    forces the crossing to actually span several `execute_navigation_macro`
    calls rather than finishing within a single one - from this test's
    starting position, one call at the real default is enough to walk the
    whole remaining route, which would leave every call after the first a
    no-op and never actually exercise re-deriving the route across calls.

    "At or closer" rather than "exactly reaches" because Oak's Lab is an
    indoor map: PyBoy's `game_area_collision()` is only verified accurate
    outdoors (see `_walk_out_of_the_house`'s docstring above), so last-mile
    A* may not be able to walk the player any further once inside - this
    still proves #102's routing itself (leaving Pallet Town via the correct
    door, landing on the target map) without depending on that separate,
    pre-existing indoor-collision limitation.
    """
    for direction in _NEAR_OAKS_LAB_DOOR_PATH:
        execute_button(pyboy_outdoors, direction)
    before = extract_game_state(pyboy_outdoors)
    assert before.map_id == _MAP_PALLET_TOWN
    milestone = Milestone(
        milestone_id="test_cross_map_milestone",
        description="a milestone used only by this test",
        target=MilestoneTarget(
            map_id=_MAP_OAKS_LAB, map_name="Oaks Lab", target_x=8, target_y=3
        ),
    )
    target = resolve_navigation_target(milestone, before.map_id)
    assert target is not None

    landing_distance: int | None = None
    crossed_on_invocation: int | None = None
    for invocation in range(10):
        state = extract_game_state(pyboy_outdoors)
        if state.map_id == target.map_id and landing_distance is None:
            landing_distance = abs(state.player_x - target.x) + abs(
                state.player_y - target.y
            )
            crossed_on_invocation = invocation
        execute_navigation_macro(pyboy_outdoors, target, max_steps=2)

    # Confirms the crossing genuinely took more than one call - otherwise
    # the loop above wouldn't actually be exercising #102's cross-call
    # statelessness (re-deriving the next hop fresh every invocation).
    assert crossed_on_invocation is not None
    assert crossed_on_invocation > 0

    after = extract_game_state(pyboy_outdoors)
    assert after.map_id == target.map_id
    assert landing_distance is not None
    final_distance = abs(after.player_x - target.x) + abs(after.player_y - target.y)
    assert final_distance <= landing_distance


def test_navigation_macro_stops_immediately_if_dialog_is_already_open(
    pyboy_outdoors, monkeypatch
):
    """The dialog/battle early-break (see `execute_navigation_macro`'s own
    docstring) must fire before any button press - a dialog open at entry
    should leave the player exactly where they started, not walk toward the
    target regardless."""
    before = extract_game_state(pyboy_outdoors)
    target = NavigationTarget(
        map_id=before.map_id, x=before.player_x + 3, y=before.player_y
    )
    monkeypatch.setattr(
        "jev_plays_pokemon.navigation.extract_game_state",
        lambda pyboy: replace(before, dialog_open=True),
    )

    moved = execute_navigation_macro(pyboy_outdoors, target)

    after = extract_game_state(pyboy_outdoors)
    assert moved is False
    assert (after.player_x, after.player_y) == (before.player_x, before.player_y)


def test_navigation_macro_stops_immediately_if_a_battle_is_already_active(
    pyboy_outdoors, monkeypatch
):
    """Same early-break as above, for `battle.in_battle` - covers the
    sight-triggered-trainer case the docstring calls out explicitly."""
    before = extract_game_state(pyboy_outdoors)
    target = NavigationTarget(
        map_id=before.map_id, x=before.player_x + 3, y=before.player_y
    )
    in_battle = replace(before, battle=replace(before.battle, in_battle=True))
    monkeypatch.setattr(
        "jev_plays_pokemon.navigation.extract_game_state", lambda pyboy: in_battle
    )

    moved = execute_navigation_macro(pyboy_outdoors, target)

    after = extract_game_state(pyboy_outdoors)
    assert moved is False
    assert (after.player_x, after.player_y) == (before.player_x, before.player_y)


def test_navigation_macro_stops_as_soon_as_a_battle_starts_mid_walk(
    pyboy_outdoors, monkeypatch
):
    """The early-break isn't just an entry check - it must also catch a
    sight-triggered trainer partway through a real, in-progress walk (the
    scenario the function's own docstring names), stopping well short of
    the target rather than burning the rest of `max_steps` on stale
    collision data."""
    before = extract_game_state(pyboy_outdoors)
    target = NavigationTarget(
        map_id=before.map_id, x=before.player_x + 10, y=before.player_y
    )
    real_extract_game_state = extract_game_state
    call_count = {"n": 0}

    def fake_extract_game_state(pyboy):
        call_count["n"] += 1
        state = real_extract_game_state(pyboy)
        if call_count["n"] >= 3:
            return replace(state, battle=replace(state.battle, in_battle=True))
        return state

    monkeypatch.setattr(
        "jev_plays_pokemon.navigation.extract_game_state", fake_extract_game_state
    )

    moved = execute_navigation_macro(pyboy_outdoors, target)

    after = extract_game_state(pyboy_outdoors)
    assert moved is True
    assert after.player_x - before.player_x < 10


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

    target = resolve_navigation_target(milestone, before.map_id)
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
_EVENT_GOT_POKEDEX = 37  # the same `event_constants.asm` ordinal
# `milestones.py`'s own `_EVENT_GOT_POKEDEX` carries, and the reason
# `_load_pewter_gym_interior_fixture` already re-applies it on load: before it
# is set, `scripts/ViridianCity.asm`'s `ViridianCityCheckGotPokedexScript`
# treats tile (19, 9) - the gap beside the sleeping Old Man, on the only street
# that reaches the Gym's pocket from the north - as a personal insult, answering
# any step onto it with "You can't go through here!" and a simulated D-pad-down.
_EVENT_BEAT_BROCK_BIT = 119  # pret/pokered's event_constants.asm: Pewter
# City events start at `const_next $68` (=104) for EVENT_BOUGHT_MUSEUM_TICKET,
# +1 for EVENT_GOT_OLD_AMBER, `const_skip 8` to EVENT_BEAT_PEWTER_GYM_TRAINER_0
# (114), `const_skip 3` to EVENT_GOT_TM34 (118), +1 for EVENT_BEAT_BROCK (119).
# #113: this, not `wObtainedBadges`'s Boulder Badge bit, is what actually
# gates Pewter City's own east-exit escort - see
# `_load_pewter_to_route3_fixture`'s docstring for the full story.
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

_PARTY_COUNT_ADDRESS = 0xD163
_PARTY_SPECIES_LIST_ADDRESS = 0xD164
_PARTY_MON_1_ADDRESS = 0xD16B  # matches game_state.py's own
# _PARTY_MON_BASE_ADDRESS

_BADGES_ADDRESS = 0xD356  # matches game_state.py's own _BADGES_ADDRESS
# Bit position per badge, matching game_state.py's own _BADGE_ITEM_IDS order
# (BOULDERBADGE..EARTHBADGE, items.py 0x15..0x1C) - one bit per badge, set
# directly rather than earned by playing the earlier gyms, per #100's own
# "set prerequisite story state directly" instruction for milestones that
# sit behind several earlier badges.
_BADGE_BITS: dict[str, int] = {
    "BOULDERBADGE": 0,
    "CASCADEBADGE": 1,
    "THUNDERBADGE": 2,
    "RAINBOWBADGE": 3,
    "SOULBADGE": 4,
    "MARSHBADGE": 5,
    "VOLCANOBADGE": 6,
    "EARTHBADGE": 7,
}


def _set_badges(pyboy: PyBoy, badges: tuple[str, ...]) -> None:
    """Sets exactly `badges` on `wObtainedBadges`, directly - see
    `_BADGE_BITS`'s own docstring for why this is set rather than earned."""
    mask = 0
    for badge in badges:
        mask |= 1 << _BADGE_BITS[badge]
    pyboy.memory[_BADGES_ADDRESS] = mask


# `MON_EXP` for `_give_overpowered_party` (offset 14 in the 44-byte party
# struct, per game_state.py's own party_struct comments): 1,059,860 is
# `pret/pokered`'s Medium-Slow growth curve (`data/growth_rates.asm`,
# `growth_rate 6, 5, -15, 100, 140` = (6/5)n^3 - 15n^2 + 100n - 140) at
# n=100. See `_give_overpowered_party`'s docstring for why a level-
# *consistent* total is required and a maxed one actively breaks fights.
_EXP_LEVEL_100 = 1_059_860


def _give_overpowered_party(pyboy: PyBoy) -> None:
    """Writes one absurdly-strong party Pokemon directly into WRAM.

    Milestones past Pewter City sit behind Route 2's mandatory Viridian
    Forest crossing, which cannot be walked through blind without ever
    risking a trainer's sight-triggered battle (`pret/pokered`'s Bug
    Catchers) - trainer battles, unlike wild ones, can't be fled, so
    surviving them is a precondition for the walk itself, not something
    these tests are trying to verify. This is exactly the kind of
    prerequisite state the issue says to set directly.

    HP/Attack/Defense/Speed/Special are all set to 999 - not higher:
    999 survived every fight encountered while building this; 65000 was
    tried and once produced a real, reproducible battle-engine freeze
    (a "but it failed!" exchange that never advanced across hundreds of
    presses), so this deliberately stays well clear of extreme values.

    `_EXP_LEVEL_100` instead of a maxed `$FFFFFF`, and for a sharper reason
    than "don't let EXP creep up": Gen 1 recomputes a Pokemon's level from
    its *total* EXP the first time any is gained, and its level-from-EXP
    lookup only covers the 100 table entries - an EXP total past level
    100's threshold makes that recomputation land on a nonsense low level.
    Measured against Pewter Gym's own trainer: with `$FFFFFF` the party
    mon read L100 999/999 going in, became a real level-6 28/28 spread
    mid-fight the instant it scored its first KO, lost, and blacked the
    player out to Pallet Town (map 0, tile (5, 6)) - which from the outside
    just looked like `_advance_past_any_encounter` failing to walk the
    gym. With a level-consistent total the same fight ends in one turn and
    the player never leaves the map. 1,059,860 is `pret/pokered`'s
    Medium-Slow curve (`data/growth_rates.asm`: (6/5)n^3 - 15n^2 + 100n -
    140) at n=100; it is at or above level 100's requirement for every one
    of the six Gen 1 growth curves except Slow, where it still lands at
    ~97, so it stays correct whatever species this is pointed at.

    `_resolve_any_battle` re-applies this after every battle for the same
    reason - the recalculation happens the instant EXP is gained, and with
    a consistent total it now recomputes to the same strong level instead
    of a broken one.
    """
    pyboy.memory[_PARTY_COUNT_ADDRESS] = 1
    pyboy.memory[_PARTY_SPECIES_LIST_ADDRESS] = 1  # RHYDON's internal index
    pyboy.memory[_PARTY_SPECIES_LIST_ADDRESS + 1] = 0xFF  # list terminator

    stat = (999).to_bytes(2, "big")
    mon = bytearray(44)  # PARTYMON_STRUCT_LENGTH, per game_state.py's own
    # party_struct offset comments
    mon[0] = 1  # species
    mon[1:3] = stat  # current HP
    mon[4] = 0  # status: healthy
    mon[8] = 33  # move 1: TACKLE (real, damaging, nonzero)
    mon[14:17] = _EXP_LEVEL_100.to_bytes(3, "big")  # EXP: see above
    mon[29] = 35  # move 1 PP
    mon[33] = 100  # level
    mon[34:36] = stat  # max HP
    mon[36:38] = stat  # attack
    mon[38:40] = stat  # defense
    mon[40:42] = stat  # speed
    mon[42:44] = stat  # special
    for offset, value in enumerate(mon):
        pyboy.memory[_PARTY_MON_1_ADDRESS + offset] = value


def _in_battle(pyboy: PyBoy) -> bool:
    battle = extract_game_state(pyboy).battle
    return bool(battle and battle.in_battle)


def _resolve_any_battle(pyboy: PyBoy, max_presses: int = 2000) -> None:
    """Mashes A through a wild or trainer battle already in progress,
    relying on `_give_overpowered_party`'s stats for a guaranteed win, and
    re-applies that party afterward (see its own docstring for why).

    Settles 30 extra frames after every press - confirmed necessary by a
    direct A/B test: without it, a fight where our own Pokemon gets put
    to sleep can run 100,000+ presses without ever ending, because each
    press lands mid print/animation and never actually completes a turn -
    pret/pokered only decrements the sleep counter when a move is
    actually attempted, so a turn that never completes never wakes it up.
    With the extra settle, the same fight resolves normally. Even a
    "normal" trainer fight can still take several hundred presses (real
    turns, not a stall) if our own Pokemon keeps getting put back to
    sleep, hence the generous default budget.
    """
    for _ in range(max_presses):
        if not _in_battle(pyboy):
            _give_overpowered_party(pyboy)
            return
        execute_button(pyboy, "a")
        pyboy.tick(30, True)
    raise AssertionError(f"battle did not resolve within {max_presses} presses")


# How many 30-frame windows `_advance_past_any_encounter` waits (without
# pressing anything) after a dialog closes before it believes nothing is
# coming. 8 = 240 frames, several times the ~30-frame gap measured between
# Pewter Gym's trainer closing its greeting and `wIsInBattle` going nonzero.
_POST_DIALOG_SETTLE_CHECKS = 8


def _advance_past_any_encounter(pyboy: PyBoy) -> bool:
    """Viridian Forest's Bug Catchers (and Pewter Gym's own trainer) are
    sight-triggered: walking into view opens a "Hey, wait up!"-style
    greeting dialog *before* the battle itself actually starts, so a
    dialog box goes up a few steps before `BattleState.in_battle` does.
    A step-by-step walker that only checks `_in_battle` can stall here
    indefinitely (the dialog blocks movement, so every direction reads as
    "blocked" without this). If either is showing, mash through into the
    fight and resolve it; otherwise this is a no-op.

    Uses `_dialog_text_visible`, not `GameState.dialog_open`, for both
    checks - confirmed necessary the hard way: a trainer's own greeting
    and post-battle quote are exactly the short, single-page kind
    `_dialog_text_visible`'s own docstring describes, which never draw
    `dialog_open`'s continuation-arrow tile at all. Missing that left a
    real, still-open text box undetected (`dialog_open=False` even while
    text was visibly on screen) - and since arrow presses don't dismiss a
    Gen 1 dialog box, only A/B do, every direction then reads as
    "blocked" with no way to tell that from a genuine dead end.

    Stops the instant both checks read clear and presses nothing after
    that (#113). The original version mashed "a" 20 times unconditionally
    once anything was detected, breaking early only when `_in_battle` went
    true - which is wrong for the short, single-page "I already beat you,
    here's my canned line" dialog a *revisited* stationary trainer shows,
    because there is no battle to break into: presses 7-20 landed after
    the dialog had already closed, each one re-opening it by re-talking to
    the NPC standing right there. Verified directly against Route 3's
    trainers that way - the flag genuinely said "beaten", the dialog
    genuinely did clear, and the loop was the whole problem.

    Returns whether a battle was actually fought and finished. A caller that
    just pressed "up" into a tile needs that distinction: a tile answered by a
    trainer who then gets beaten is a tile that is free one dialog later, while
    a tile answered by an NPC who cannot be fought stays answered, and a walker
    that learns from the first one blacklists a route it could simply have won.

    "Both checks read clear" is not the same moment as "nothing is coming",
    though, and the first cut of that fix got this wrong: a sight-triggered
    trainer's greeting closes on its last press and only *then* starts its
    battle, so for about a screen-fade's worth of frames both checks
    genuinely read false with a fight on its way in. Returning there handed
    the battle to the caller still opening, which is what
    `test_walking_to_pewter_gym_brock_reaches_a_rom_verified_tile` was then
    failing on. The settle loop below waits that window out - pressing
    nothing, since pressing is exactly what #113 had to stop - and only
    returns once a battle has had every chance to show up.

    Neither check can be trusted as the *gate*, for the reason recorded in
    `_dialog_text_visible`'s own docstring: on any map tall enough to fill the
    window's bottom rows it reads True with nothing on screen, so gating on it
    mashed the full 20 + 10 presses into every single step of a walk across
    Route 3 or Route 4. "A press the text row doesn't answer" is the close
    signal instead, which is equally right on a short map (its row just stays
    blank) - so nothing here depends on how tall the current map happens to be.
    """
    if not (_dialog_text_visible(pyboy) or _in_battle(pyboy)):
        return False
    if not _in_battle(pyboy):
        # One press, sampled: a text row that did not move was the map's own
        # tiles, not a box (see `_dialog_text_visible`'s own docstring for why
        # the check alone can't tell those apart on a tall map).
        row = _dialog_row_tiles(pyboy)
        execute_button(pyboy, "a")
        pyboy.tick(20, True)
        if not _in_battle(pyboy) and _dialog_row_tiles(pyboy) == row:
            return False
    for _ in range(20):
        if _in_battle(pyboy):
            break
        row = _dialog_row_tiles(pyboy)
        execute_button(pyboy, "a")
        pyboy.tick(20, True)
        if not _in_battle(pyboy) and _dialog_row_tiles(pyboy) == row:
            # #113: stop here. Pressing "a" any further re-talks to the
            # stationary NPC this player is left standing adjacent to and
            # facing, re-opening the very dialog just closed - which from
            # outside looks like an unresolvable stuck loop. A trainer's
            # "already beaten" canned line takes exactly this branch (no
            # battle to break into), so the old unconditional mash kept
            # pressing 14+ more times after the dialog was already gone.
            closed_row = _dialog_row_tiles(pyboy)
            for _ in range(_POST_DIALOG_SETTLE_CHECKS):
                # Waiting is safe where pressing isn't. Confirmed against
                # Pewter Gym's own trainer: the greeting closes on press 12
                # and `wIsInBattle` goes nonzero 30 frames later, so this is
                # the window its fight opens in - and the reason a single
                # 30-frame re-check was not enough to be sure. A row that
                # starts moving again reads the same way: some new box is
                # coming up, which is not an empty screen.
                pyboy.tick(30, True)
                if _in_battle(pyboy) or _dialog_row_tiles(pyboy) != closed_row:
                    break
            else:
                return False
    # Every way into `_resolve_any_battle` from here has just seen
    # `wIsInBattle` go nonzero - the loop above only leaves it two ways, a
    # `break` on the flag or a fall-through after a battle was already up -
    # so this is the answer the caller wants, read before the fight is gone.
    fought = _in_battle(pyboy)
    _resolve_any_battle(pyboy)
    pyboy.tick(60, True)  # let the fade back to the overworld finish
    # before any caller reads game_area_collision() again - confirmed
    # necessary: without it, the very next on-screen A* replan can read
    # a stale/transitional collision buffer and give up immediately.
    for _ in range(10):
        # a trainer's post-battle quote ("Oh, I lost...") can still be
        # showing right after `BattleState.in_battle` already reads
        # False - confirmed to otherwise silently block all movement
        # afterward (arrow presses don't dismiss a dialog box in Gen 1,
        # only A/B do, so every direction looks "blocked" until this is
        # cleared). Sampled the same way as above: a press the text row
        # doesn't answer is the quote being gone, not a page to advance.
        row = _dialog_row_tiles(pyboy)
        execute_button(pyboy, "a")
        pyboy.tick(20, True)
        if _dialog_row_tiles(pyboy) == row:
            break
    return fought


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


def _set_event_flag(pyboy: PyBoy, flag: int) -> None:
    """Sets one `wEventFlags` bit by its `event_constants.asm` ordinal.

    `game_state.py`'s `_read_event_flags` numbers flags `offset * 8 + bit` from
    `_EVENT_FLAGS_START_ADDRESS` (`milestones.py`'s module docstring records the
    same convention), so an ordinal is a byte offset and a bit within it - no
    hand-counted bit position to get wrong, which is the whole reason
    `milestones.py` carries the ordinals in the first place.
    """
    byte, bit = divmod(flag, 8)
    address = _EVENT_FLAGS_START_ADDRESS + byte
    pyboy.memory[address] |= 1 << bit


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


def _direction_toward(
    from_x: int, from_y: int, player_x: int, player_y: int
) -> str | None:
    """Which of the four raw directions moves the player toward
    `(from_x, from_y)` - a hop's own source tile - along whichever single
    axis it actually differs on (a hop tile only ever differs from the
    approach on one axis: the map-edge/door's own perpendicular axis).
    `None` if already aligned on both (nothing to nudge)."""
    if from_y < player_y:
        return "up"
    if from_y > player_y:
        return "down"
    if from_x < player_x:
        return "left"
    if from_x > player_x:
        return "right"
    return None


def _nudge_across_hop(
    pyboy: PyBoy, hop, player_x: int, player_y: int, max_offset: int = 3
) -> None:
    """Crosses a stalled hop with a raw directional press toward its own
    tile (`_cross_map_edge`) - and, since that exact tile can itself sit on
    a real, physically-blocked tile a few tiles off from the actual
    walkable gap in the map edge's tree/fence line (confirmed directly:
    Route 1's own north edge into Viridian City blocks dead straight-on at
    the hop's exact computed x, a couple of tiles either side crosses
    fine - `connection_hop`'s own docstring already notes any tile in a
    connection's overlap range is an equally valid crossing, not just the
    midpoint it picks), searches a short lateral offset either side (still
    pressing the same primary direction to actually cross) before giving
    up.
    """
    direction = _direction_toward(hop.from_x, hop.from_y, player_x, player_y)
    if direction is None:
        return
    start_map = extract_game_state(pyboy).map_id

    def crossed() -> bool:
        _cross_map_edge(pyboy, direction)
        return extract_game_state(pyboy).map_id != start_map

    if crossed():
        return
    lateral_a, lateral_b = (
        ("left", "right") if direction in ("up", "down") else ("up", "down")
    )
    for lateral, opposite in ((lateral_a, lateral_b), (lateral_b, lateral_a)):
        for _ in range(max_offset):
            execute_button(pyboy, lateral)
            if crossed():
                return
        for _ in range(max_offset):
            execute_button(pyboy, opposite)  # back to center before the other side


def _walk_to_milestone_target(
    pyboy: PyBoy, map_id: int, x: int, y: int, max_calls: int = 40
) -> None:
    """Like `_walk_toward`, but for a target that may sit on a different map
    than wherever the player currently is - #102's cross-map routing (see
    `travel_graph.MILESTONE_MAP_IDS`) does the actual map-crossing, one hop
    per `execute_navigation_macro` call, as long as every map on the route
    is in that scoped graph. A higher default `max_calls` than
    `_walk_toward`'s, since a milestone's target can sit several maps and
    screens away rather than a single nearby tile.

    A hop's own tile can itself misread as collision-blocked in PyBoy's
    on-screen `game_area_collision()` - the same pre-existing limitation
    `_NEAR_OAKS_LAB_DOOR_PATH`'s comment documents for Pallet Town's fence
    tile (confirmed directly: the on-screen A* walks right up to within a
    tile or two of a hop tile, then reports no further progress, even
    though the real game lets the player step onto it) - so when
    `execute_navigation_macro` stops making progress while still short of
    `map_id`, this nudges across the map edge (`_nudge_across_hop`) rather
    than treating "the on-screen A* gave up" as "no route exists."
    """
    for _ in range(max_calls):
        state = extract_game_state(pyboy)
        if state.dialog_open or state.battle.in_battle:
            return
        if state.map_id == map_id and (state.player_x, state.player_y) == (x, y):
            return
        target = NavigationTarget(map_id=map_id, x=x, y=y)
        if execute_navigation_macro(pyboy, target):
            continue
        if state.map_id == map_id:
            return
        hop = next_hop(_travel_graph(), state.map_id, map_id)
        if hop is None:
            return
        before = extract_game_state(pyboy)
        _nudge_across_hop(pyboy, hop, before.player_x, before.player_y)
        if extract_game_state(pyboy).map_id == before.map_id:
            return  # the nudge search exhausted every offset without crossing


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

    The tile row alone is not enough, and #99's Route 4 crossing is where
    that was caught: the row only reads blank when the window's bottom rows
    fall outside the map, which is true of Pallet Town from the starting
    position and of small interior maps, but not of any map tall enough to
    fill the window. Route 4 is 18 tiles tall - exactly the window's height -
    so its own fence and ground tiles occupy the text row permanently, and the
    check returned True with nothing at all on screen (screenshot-confirmed,
    with the player walking normally throughout). Read this as "text *could* be
    showing", never as "a box is open": `_advance_past_any_encounter` used to
    gate on it directly, which turned every single step of a walk across such a
    map into 20 + 10 mashed "a" presses into whatever tile the player happened
    to face - and re-talking to the NPC standing there is exactly the failure
    #113's note above had already identified as the thing to stop doing.

    A memory flag would settle it, and the obvious candidate was ruled out:
    reading 0xFF8C does distinguish the two states here (0x06 with Viridian
    City's "The GYM's doors are locked..." box up, 0x00 on Route 4 with
    nothing up), but counting `ram/hram.asm`'s declarations from its $FF80
    start puts that byte on scratch (`hSpriteHeight` in one count,
    `hMultiplicand`/`hMultiplier` in a union-aware one), so its agreement with
    the dialog was coincidence and it is not safe to rely on. What _advance_
    past_any_encounter does instead - press once and see whether the text row
    moved - is confirmed by the same two states and asks nothing of any
    address.
    """
    tilemap = pyboy.tilemap_window
    return any(
        tilemap[col, _DIALOG_TEXT_ROW] != _DIALOG_TEXT_BLANK_TILE
        for col in _DIALOG_TEXT_COLUMNS
    )


def _dialog_row_tiles(pyboy: PyBoy) -> tuple[int, ...]:
    """The dialog text row itself, for comparing it across a press: see
    `_advance_past_any_encounter`'s use."""
    tilemap = pyboy.tilemap_window
    return tuple(tilemap[col, _DIALOG_TEXT_ROW] for col in _DIALOG_TEXT_COLUMNS)


# ## Walking a whole map by the ROM's own collision data
#
# `execute_navigation_macro` plans with PyBoy's `game_area_collision()` over the
# on-screen window only, and for the single-screen hops every test above needs,
# that is enough. It is not enough to cross a town. Two separate failures, both
# confirmed by reading the tile grid under the player at the tile a walk stopped
# on:
#
# * A sprite is solid in-game and invisible to `game_area_collision()`. The
#   approach to Viridian Gym's door stops dead at `(19, 9)` with nothing on
#   screen except that the Old Man asleep at `(18, 9)` has started talking - and
#   `pret/pokered`'s `scripts/ViridianCity.asm` (`ViridianCityOldManSleepyText`)
#   shows what talking to him does: `PrintText`, then
#   `call ViridianCityMovePlayerDownScript`, then
#   `SCRIPT_VIRIDIANCITY_PLAYER_MOVING_DOWN`. He is not an obstacle the A* can be
#   nudged past; bump him and the ROM answers with text and *pushes the player one
#   tile back the way they came*, which is exactly how a walk aimed at `(32, 8)`
#   finished at `(32, 10)`. #113 wrote the same lesson up for Route 3's stationary
#   trainers.
# * `execute_navigation_macro` refuses to move at all while
#   `GameState.dialog_open` reads `True` - correct behaviour in a game where a
#   text box owns the D-pad, and the reason a walk that keeps getting spoken to by
#   an NPC makes no progress at all.
#
# So the helpers below plan on the ROM's own data instead: `tileset_collision`'s
# per-tile decode for the terrain, every `rom_maps` object record as solid (the
# second obstacle layer, which no terrain decode can see), and - on top of both -
# every tile this particular walk has tried to enter and failed to enter, because
# a sprite that wanders, a one-way ledge, or an Old Man who shoves you south is
# not in any of the ROM's static tables.

_MAP_GEOMETRY: dict[
    int, tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]
] = {}
_ROM_CACHE: list = []


def _rom_parse() -> tuple[bytes, dict, tuple, frozenset[tuple[int, int, int]]]:
    """The ROM bytes, its parsed maps, its parsed tileset headers, and its
    blocked tile-pair crossings - parsed once per module, since every one of
    these walks asks for the same four."""
    if not _ROM_CACHE:
        rom = load_rom(ROM_PATH)
        _ROM_CACHE.extend(
            (
                rom,
                parse_all_maps(rom),
                parse_tileset_headers(rom),
                parse_tile_pair_collisions(rom),
            )
        )
    return _ROM_CACHE[0], _ROM_CACHE[1], _ROM_CACHE[2], _ROM_CACHE[3]


# Maps whose tiles throw the player, by the raw tile IDs that do it.
# `pret/pokered`'s `engine/overworld/spinners.asm` picks its table with
# `ld a, [wCurMapTileset]` / `cp FACILITY` - every tileset but `FACILITY` gets
# `GymSpinnerArrows`, whose four IDs are `$3c`, `$3d`, `$4c`, `$4d` - and
# `scripts/ViridianGym.asm` / `scripts/RocketHideoutB2F.asm` /
# `scripts/RocketHideoutB3F.asm` are what call it. Keyed by map rather than
# tileset deliberately: `$3C` is in the Overworld tileset's own passable list,
# so a tileset-keyed rule would wall off ordinary Viridian City ground.
# Giovanni's room is reachable without touching a single one (`(16, 16)` to his
# tile `(2, 1)` is 39 steps that cross none), which is why treating these as
# walls is enough - no walk planned here needs a spinner's ride.
_SPINNER_TILE_IDS: dict[int, frozenset[int]] = {
    45: frozenset({0x3C, 0x3D, 0x4C, 0x4D}),
}


def _map_obstacles(
    map_id: int,
) -> tuple[frozenset[tuple[int, int]], frozenset[tuple[int, int]]]:
    """`(solid, warps)` for a map: every tile its own tileset collision lists
    report as impassable, plus every object record's tile, plus the spinner
    tiles above; and separately its warp tiles, which are walkable in the sense
    that the player can stand on them and unusable in the sense that no route
    may be planned *through* them.

    The decode reads a stair or door tile as solid, so warp tiles are lifted out
    of the solid set here and put back by `_route_across_map` for every tile
    except the walk's own start and goal. Mt Moon's first crossing showed why
    both halves of that matter: with its exit warp at `(14, 35)` left plain
    walkable, the planner routed the cave crossing *through* it and the player
    left the dungeon mid-walk. Spinners are the same trap from the other side -
    the decode calls them walkable, which is true, and the game then moves the
    player to wherever the arrow points, so a plan that steps on one is not a
    plan the walk can follow.
    """
    if map_id not in _MAP_GEOMETRY:
        rom, maps, headers, _pairs = _rom_parse()
        rmap = maps[map_id]
        width = rmap.width_blocks * 2
        height = rmap.height_blocks * 2
        solid = {(o.x, o.y) for o in rmap.objects}
        solid |= {
            (x, y)
            for y in range(height)
            for x in range(width)
            if is_walkable(rom, rmap, headers, x, y) is not True
        }
        spinners = _SPINNER_TILE_IDS.get(map_id)
        if spinners is not None:
            solid |= {
                (x, y)
                for y in range(height)
                for x in range(width)
                if raw_tile_id(rom, rmap, headers, x, y) in spinners
            }
        warps = {(w.x, w.y) for w in rmap.warps}
        _MAP_GEOMETRY[map_id] = (
            frozenset(solid - warps),
            frozenset(warps),
        )
    return _MAP_GEOMETRY[map_id]


def _route_across_map(
    map_id: int,
    start: tuple[int, int],
    goal: tuple[int, int],
    learned: frozenset[tuple[int, int]],
) -> tuple[tuple[int, int], ...] | None:
    """Shortest walkable tile sequence from `start` to `goal` on one map, or
    `None` if there is none. Breadth-first over the four plain directions - the
    same shape as `travel_graph.find_route`, one tile at a time instead of one
    map at a time.

    A step is refused if either tile is one the tileset's collision lists call
    impassable *or* the two tiles are one of the ROM's blocked elevation pairs -
    the first of the two checks `pret/pokered`'s `home/overworld.asm`'s
    `CollisionCheckOnLand` runs before it lets an ordinary step happen. The
    second half is an edge test rather than a tile test, so it cannot live in
    `_map_obstacles`' solid set and is checked per step here. Mt Moon B2F is
    what forced it: `(24, 12)` is walkable, and every one of the twelve steps
    its corridor would take from the `(25, 9)` landing is a `CAVERN $20 -> $05`
    crossing the ROM refuses - so a plan that could not see the rule sent the
    walk pressing into a wall the map was never going to open, press after
    press, and blacklisted tiles that were never the problem.
    """
    solid, warps = _map_obstacles(map_id)
    blocked = (solid | learned | warps) - {start, goal}
    rom, maps, headers, pairs = _rom_parse()
    rmap = maps[map_id]
    width, height = rmap.width_blocks * 2, rmap.height_blocks * 2
    seen = {start}
    frontier = [start]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    while frontier:
        nxt = []
        for x, y in frontier:
            for dx, dy in ((0, -1), (0, 1), (-1, 0), (1, 0)):
                step = (x + dx, y + dy)
                if not (0 <= step[0] < width and 0 <= step[1] < height):
                    continue
                if step in seen or step in blocked:
                    continue
                if crosses_blocked_pair(rom, rmap, headers, pairs, (x, y), step):
                    continue
                seen.add(step)
                came_from[step] = (x, y)
                if step == goal:
                    path = [step]
                    while path[-1] != start:
                        path.append(came_from[path[-1]])
                    return tuple(reversed(path))
                nxt.append(step)
        frontier = nxt
    return None


_DIRECTION_OFFSETS = {
    "up": (0, -1),
    "down": (0, 1),
    "left": (-1, 0),
    "right": (1, 0),
}


# How many samples in a row a new tile has to be read on before a press is taken
# to have landed there. One is the bug this exists to avoid (see
# `_press_and_settle`); two more is enough to sit out a step's own animation.
_SETTLED_REPEATS = 2

# How long (in presses of the walk that refused it) a *first* refusal is held
# against a tile, and how many refusals make one permanent. See `_walk_tiles`.
_LEARNED_PRESSES = 25
_LEARNED_LIMIT = 2

# How many 20-frame windows `_press_and_settle` waits out, pressing nothing, for
# a press that so far has no answer at all. 20 = 400 frames, which is the
# sight-triggered trainer's whole measured sequence from the "!" to the fight -
# see the wait inside `_press_and_settle` for the numbers.
_SCRIPT_SETTLE_CHECKS = 20

# How many "a" presses a walk is willing to spend answering whatever its
# directional press walked into. Measured on Viridian Gym's corridor trainer:
# twelve, of which the first three move nothing at all.
_ANSWER_PRESS_LIMIT = 25


def _fight_what_is_in_the_way(pyboy: PyBoy) -> bool:
    """Answers whatever a directional press walked into, and fights it. Returns
    whether a battle actually happened.

    `_advance_past_any_encounter` cannot be the one to do this, and calling it
    here is what let a walk stall for press after press on the only corridor
    into Viridian Gym's northwest pocket: its first move is one "a" press
    sampled, and on any map tall enough for the window's bottom rows to be the
    map's own tiles - the case its own docstring says makes
    `_dialog_text_visible` useless as a gate - a trainer who has spotted the
    player and is still walking over answers that probe with an unchanged row,
    so it returns having done nothing. Measured directly at the stall: the
    trainer's text row first moves on the third press and `wIsInBattle` goes
    nonzero on the twelfth, with presses nought through two reading nothing but
    map tiles.

    So this presses "a" a fixed, generous number of times instead and only
    looks for a battle. It gives up without fighting anything on a tile whose
    answer is an NPC who cannot be fought, which is what `_walk_tiles` needs to
    hear in order to hold that refusal against the tile - #113's lesson about
    re-talking to a stationary NPC still stands, it is just not the right answer
    for an NPC who is about to battle you and disappear.
    """
    for _ in range(_ANSWER_PRESS_LIMIT):
        if _in_battle(pyboy):
            break
        execute_button(pyboy, "a")
        pyboy.tick(20, True)
    if not _in_battle(pyboy):
        return False
    _resolve_any_battle(pyboy)
    pyboy.tick(60, True)  # let the fade back to the overworld finish
    # his "Oh, I lost..." quote is still up when `wIsInBattle` is already 0
    _advance_past_any_encounter(pyboy)
    return True


def _press_and_settle(pyboy: PyBoy, direction: str) -> tuple[int, int, int, bool]:
    """Presses `direction`, waits for whatever that press can start to finish,
    and returns `((map, x, y), fought)` - where the player comes to rest, and
    whether the press's own story gate (a text box and/or a battle) had to be
    fought through to get there.

    Waiting for a fixed number of frames is not enough, and neither is asking
    either available dialog check. One tile of walking does not always finish
    inside a short tick budget, so a press that really worked gets scored as a
    failure - which, once failures are blacklisted, lets a walker declare the tile
    it is standing on impassable and quit. And of the two checks,
    `GameState.dialog_open` only sees the continuation arrow (which single-page
    NPC lines never draw at all, per `_dialog_text_visible` above) while
    `_dialog_text_visible` reads `True` permanently on any map tall enough to fill
    the window's bottom rows. What works is the *difference* `_advance_past_any_encounter`
    already relied on: sample the text row, press, and see whether it changed.

    Returning where the player ended up - rather than whether they moved - is
    what makes the caller's verdict about a tile mean anything, because two
    things here move `wX`/`wY` without ever being a step. Gen 1 writes them at
    the *start* of a tile step and puts them back when the step collides, so one
    differing sample is not evidence of a step; and a map script can move the
    player a whole tile in a direction nobody pressed (Viridian City's
    `ViridianCityCheckGotPokedexScript` does exactly that, quoted in
    `test_walking_to_viridian_gym_giovanni_reaches_a_rom_verified_tile`). A
    walker that stops looking at the first difference believes both, blacklists
    nothing, and presses into the same refusal forever - which is precisely how
    the first version of this walker spent sixty presses going nowhere on
    `(19, 9)`. So a tile only counts once it has been *stayed* on.
    """
    for _ in range(3):
        before = extract_game_state(pyboy)
        row = _dialog_row_tiles(pyboy)
        execute_button(pyboy, direction)
        resting: tuple[int, int] | None = None
        repeats = 0
        answered = False
        for _ in range(40):
            pyboy.tick(10, True)
            now = extract_game_state(pyboy)
            if now.map_id != before.map_id:
                return now.map_id, now.player_x, now.player_y, False
            tile = (now.player_x, now.player_y)
            if tile != (before.player_x, before.player_y):
                if tile == resting:
                    repeats += 1
                    if repeats >= _SETTLED_REPEATS:
                        return now.map_id, tile[0], tile[1], False
                else:
                    resting, repeats = tile, 0
                continue
            if _in_battle(pyboy):
                answered = True
                break
            if _dialog_row_tiles(pyboy) != row:
                answered = True
                break
        if resting is not None:
            # Something moved the player and then moved them back inside one
            # press; that is an answer about the pressed tile, not noise.
            break
        if not answered:
            # Nothing has answered the press at all yet - which is also what a
            # sight-triggered trainer looks like for the second or so between
            # spotting the player and arriving to talk. Measured on Viridian
            # Gym's corridor: the "!" goes up on the press, his text row first
            # moves about 200 frames later, and `wIsInBattle` goes nonzero
            # about 120 frames after that. Clearing the screen without waiting
            # for him finds an empty one, so the press gets re-attempted into a
            # script that still owns the map and the walk reads the whole
            # corridor - the only way into Giovanni's half of the room - as a
            # wall. Waiting is what makes the clear below able to see the
            # dialog this press is on its way to producing.
            for _ in range(_SCRIPT_SETTLE_CHECKS):
                pyboy.tick(20, True)
                if _in_battle(pyboy) or _dialog_row_tiles(pyboy) != row:
                    break
        # Either a text box ate the press, or a battle did, or the tile really is
        # solid - and the first two are indistinguishable from the third from
        # outside, so answer whatever is standing there and give the same press
        # another chance before anyone is called a wall.
        if _fight_what_is_in_the_way(pyboy):
            _disable_wild_encounters(pyboy)
            after = extract_game_state(pyboy)
            return after.map_id, after.player_x, after.player_y, True
        _disable_wild_encounters(pyboy)
    after = extract_game_state(pyboy)
    return after.map_id, after.player_x, after.player_y, False


def _walk_tiles(
    pyboy: PyBoy,
    x: int,
    y: int,
    max_steps: int = 200,
    map_id: int | None = None,
) -> tuple[int, int, int]:
    """Walks the player, one real tile per press, to `(x, y)` on whatever map
    they are currently on, and returns the final `(map, x, y)`. Give it
    `map_id` and it stops the moment the player is anywhere else - a walk that
    wanders onto a warp tile it wasn't aimed at has left the map it was planning
    on, and continuing to press toward a tile on a map it is no longer on is how
    a walker ends up in Pallet Town.

    One refusal is held against a tile for `_LEARNED_PRESSES` presses and then
    forgotten, because most of the reasons a tile refuses a single press - a
    sprite mid-stride, an NPC that has stepped into the gap, a page of text still
    on the screen - are gone within a few, and a walk that remembers those
    forever has usually just blacklisted the one tile it had no way round: a
    traced run up the west side of Viridian City lost its whole route that way,
    with one press at `(13, 4)` coming back without moving, `(14, 4)` becoming a
    permanent wall, and the walk spending the next 178 presses walking one tile
    backwards and forwards between `(6, 4)` and `(7, 4)`. A tile refused
    `_LEARNED_LIMIT` times is held for the rest of the walk instead, because the
    other kind of refusal does not expire: `(19, 9)` answered the same press the
    same way eighty presses apart, and a walker that forgets the sleeping Old Man
    every 25 presses spends the whole budget rediscovering him. That same trace is
    why the plan-less greedy press stops on a repeat of itself: a press this
    walker cannot improve on is not a discovery in progress, it is the walk
    standing in a hole it has already dug, and pressing it again only spends the
    step budget.

    The plan is re-run from the player's *live* position before every single
    press, and the tile a press failed to put the player *on* is added to that
    walk's own set of blocked tiles. "Put the player on" is the whole test - not
    "the player moved" - because the ROM
    moves `wX`/`wY` for reasons that are not the pressed step, a collided step
    included, and a map script that shoves the player back down the tile they
    were standing on (Viridian City's `ViridianCityCheckGotPokedexScript`, in
    `test_walking_to_viridian_gym_giovanni_reaches_a_rom_verified_tile` below)
    moves them a whole tile. Re-planning is what makes the learned set worth
    having: a sprite that wanders, a sight-triggered trainer who walked over, an
    NPC who answers a bump with text, and a scripted refusal like that one all
    look like the same thing from inside - a tile that will not have you - and
    the route goes around it from the next press on. For `(19, 9)` that means the
    walk stops trying to go up column 19 at all: with that one tile blacklisted
    the only complete route the planner can still find runs round the west side
    of town, 81 steps where the direct way was 57.

    When the plan runs out entirely the walker presses toward the goal anyway
    rather than giving up. That is deliberate and it is the only way through the
    two things the static picture cannot express at all: Gen 1 ledges, which let
    the player step off and not back on (`tileset_collision.py`'s own "What this
    doesn't model" section says it does not model them), and object sprites that
    this very walk has already removed - Mt Moon B2F's `(5, 7)` stair is reached
    only through the tile a fossil sits on, and `_map_obstacles` reads object
    records out of the ROM, so it keeps calling a sprite that is already in the
    player's bag a wall.
    """
    strikes: dict[tuple[int, int], int] = {}
    expires: dict[tuple[int, int], int] = {}
    pressed_before: set[tuple[tuple[int, int], str]] = set()

    def refused_tiles(press: int) -> frozenset[tuple[int, int]]:
        return frozenset(
            tile
            for tile, count in strikes.items()
            if count >= _LEARNED_LIMIT or expires.get(tile, -1) > press
        )

    for press in range(max_steps):
        state = extract_game_state(pyboy)
        pos = (state.player_x, state.player_y)
        if pos == (x, y) or (map_id is not None and state.map_id != map_id):
            break
        path = _route_across_map(state.map_id, pos, (x, y), refused_tiles(press))
        if path is not None and len(path) > 1:
            step_tile = path[1]
            direction = next(
                name
                for name, (dx, dy) in _DIRECTION_OFFSETS.items()
                if (pos[0] + dx, pos[1] + dy) == step_tile
            )
        else:
            direction = _press_at_the_goal(pos, (x, y), refused_tiles(press))
            if (pos, direction) in pressed_before:
                break
            pressed_before.add((pos, direction))
        dx, dy = _DIRECTION_OFFSETS[direction]
        aimed = (pos[0] + dx, pos[1] + dy)
        landed_map, landed_x, landed_y, fought = _press_and_settle(pyboy, direction)
        if (landed_map, landed_x, landed_y) == (state.map_id, *aimed):
            continue
        if fought:
            # Nothing refused this press - a trainer stood on the tile, the
            # press walked into him, and he is not standing there any more.
            # Learning from that blacklists a corridor the walk can only
            # finish by winning, which is what Viridian Gym's own is.
            continue
        strikes[aimed] = strikes.get(aimed, 0) + 1
        expires[aimed] = press + _LEARNED_PRESSES
    final = extract_game_state(pyboy)
    return final.map_id, final.player_x, final.player_y


def _press_at_the_goal(
    pos: tuple[int, int],
    goal: tuple[int, int],
    learned: frozenset[tuple[int, int]],
) -> str:
    """Which way to press when `_route_across_map` has no plan: along whichever
    axis is further off, preferring a direction whose tile this walk has not
    already given up on. Only falls back to an exhausted direction once every
    tile around the player has refused them, which is a genuinely boxed-in
    player and not something a press can fix."""
    dx, dy = goal[0] - pos[0], goal[1] - pos[1]
    preferred = ["up", "down"] if abs(dy) >= abs(dx) else ["left", "right"]
    order = preferred + [name for name in _DIRECTION_OFFSETS if name not in preferred]
    free = [
        name
        for name in order
        if (
            pos[0] + _DIRECTION_OFFSETS[name][0],
            pos[1] + _DIRECTION_OFFSETS[name][1],
        )
        not in learned
    ]
    return free[0] if free else order[0]


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


def test_walking_to_viridian_mart_cooltrainer_reaches_a_rom_verified_tile(
    pyboy_outdoors,
):
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


_PEWTER_GYM_INTERIOR_STATE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "pewter_gym_interior.state"
)


def _apply_fixture_prerequisites(pyboy: PyBoy) -> None:
    """Re-applies on load the two pieces of prerequisite state every captured
    fixture relies on, rather than trusting what its bytes happen to carry.

    A `.state` file is a point-in-time snapshot of the *capture session's*
    helpers, so anything those helpers write is frozen at whatever they
    believed then: `_disable_wild_encounters` has to be reapplied after every
    map transition anyway (see its own docstring), and `_give_overpowered_party`
    is rewritten here so that fixing that helper also fixes fixtures captured
    before the fix. `pewter_gym_interior.state` in particular was captured
    when it wrote a maxed `$FFFFFF` EXP total - which, per
    `_give_overpowered_party`'s own docstring, doesn't just fail to help, it
    actively breaks the first fight afterwards - and re-capturing three
    hand-verified fixtures to pick up a WRAM tweak is not worth it when the
    load path can normalize it in one line.
    """
    _disable_wild_encounters(pyboy)
    _give_overpowered_party(pyboy)


def _load_pewter_gym_interior_fixture(pyboy: PyBoy) -> None:
    """Loads a captured save state already past Route 2's boulder maze,
    Viridian Forest, Pewter City's own street layout, and Pewter Gym's own
    door - landing just inside the gym itself (map 54, tile (4, 13)) -
    with `_bypass_oaks_route_1_interception`'s own EVENT_FOLLOWED_OAK_INTO_LAB
    flag set, along with EVENT_GOT_POKEDEX (which otherwise plants a
    sleeping old man blocking the only road out of Viridian City), wild
    encounters already disabled, and `_give_overpowered_party`'s own
    party already in place.

    None of the ground covered to reach this point is what this test
    verifies (the milestone under test is Brock's own tile, past all of
    it) - but getting here by actually walking it, tried first, wasn't
    reliable enough to keep:
    - Blind `_explore_toward` search across Viridian Forest worked in
      isolated tries but wasn't reliably fast: a Bug Catcher fight's exact
      frame length varies run to run, which cascades into wildly
      different amounts of backtracking - one otherwise identical run
      took over 3x as long and covered a fraction of the distance.
    - A same-map RAM position write (safe for ordinary movement, since it
      never needs the engine's map-load routine) was tried next for both
      mazes and rejected outright: an exhaustive grid search over every
      tile near a real, ROM-confirmed warp coordinate found that none of
      them trigger the actual map transition after a teleport - Gen 1's
      warp-detection apparently depends on some internal step/animation
      state a raw position write never sets up, even though ordinary
      non-warp movement from the same teleported position works
      completely normally afterward. Repeated teleporting also once left
      the emulator reading a corrupted, invalid map ID.
    - Even Pewter City's own streets, crossed with the same
      `_explore_toward` search once past both mazes, turned out just as
      unreliable: repeated runs from the identical starting tile landed at
      wildly different, unrelated dead ends without ever reaching the
      gym's door, despite the town not being maze-like at all.
    - `_walk_toward` chained through hand-found waypoints got close but
      still couldn't finish: like Viridian Mart's own door
      (`_TO_VIRIDIAN_MART_DOOR_PATH`, above), the gym's door tile itself
      reads as collision-blocked in PyBoy's static map (real Gen 1 doors
      are drawn as part of the building wall), so the on-screen A* refuses
      to route onto it at all - confirmed by dumping the collision grid
      at the tiles adjacent to it. Unlike the mart, no single nearby
      raw-button detour was found within a reasonable amount of manual
      probing; the gym building's footprint blocks the direct approaches
      tried from every side but one, which itself needs a longer detour
      than seemed worth hand-deriving.

    A captured save state sidesteps all of this: it already reflects real
    play on the other side of every warp and door above, so nothing here
    needs to find or trigger one - only the gym's own trainer fight and
    Brock's tile itself are still walked for real. Captured once from a
    real, manually-verified crossing of the whole route; regenerate by
    replaying that crossing again and re-saving over this file if the ROM
    or PyBoy's save-state format ever changes.
    """
    with _PEWTER_GYM_INTERIOR_STATE_PATH.open("rb") as f:
        pyboy.load_state(f)
    pyboy.tick(1, False)
    _apply_fixture_prerequisites(pyboy)


def test_walking_to_pewter_gym_brock_reaches_a_rom_verified_tile(pyboy_outdoors):
    """#99's boot verification for the `boulder_badge` milestone: Pewter
    Gym's PEWTERGYM_BROCK object (`milestone_targets.py`), map 54 tile
    (4, 1).

    Reaching Pewter Gym's interior at all needs crossing Route 2's boulder
    maze, Viridian Forest, Pewter City's own streets, and the gym's own
    door, past a couple of prerequisite gates (Oak's Route 1 interception
    and the Viridian old man - see `_load_pewter_gym_interior_fixture`'s
    docstring for the event flags involved) and with
    `_give_overpowered_party` in place (Viridian Forest's Bug
    Catchers can't be fled from if bumped into, unlike a wild encounter,
    so crossing it at all requires being able to win a fight no matter
    what) - none of which is what this milestone verifies, so
    `_load_pewter_gym_interior_fixture` starts the test already on the
    other side of all of it (see its own docstring for why a live
    crossing isn't done here instead).

    Brock's own gym has one trainer guarding the way to him
    (`pret/pokered`'s `PewterGymTrainerHeader0`) who is also sight-
    triggered; that fight is resolved the same way as the forest's before
    reaching Brock himself. Talking to Brock before beating him
    immediately starts their gym battle (`PewterGymBrockText`'s
    `.beforeBeat` branch calls `EngageMapTrainer` directly, no sight line
    needed) - but the pre-battle dialog it shows on the way in already
    satisfies this test's own check (`GameState.dialog_open`), the same
    as `got_starter`'s worked example above not needing to complete the
    starter selection either.
    """
    _load_pewter_gym_interior_fixture(pyboy_outdoors)
    assert extract_game_state(pyboy_outdoors).map_id == 54  # Pewter Gym

    _walk_toward(pyboy_outdoors, 4, 6)
    _advance_past_any_encounter(pyboy_outdoors)  # PewterGymTrainerHeader0

    _walk_toward(pyboy_outdoors, 4, 1)

    pyboy_outdoors.button("a", 2)
    dialog_opened = False
    for _ in range(10):
        pyboy_outdoors.tick(30, True)
        if extract_game_state(pyboy_outdoors).dialog_open:
            dialog_opened = True
            break
    assert dialog_opened


_PEWTER_TO_ROUTE3_STATE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "pewter_to_route3.state"
)
_MAP_ROUTE_3 = 14


def _load_pewter_to_route3_fixture(pyboy: PyBoy) -> None:
    """#113: loads a captured save state that has already crossed Pewter
    City's real, ROM-verified connection to Route 3 - landing at Route 3's
    own tile (0, 8), matching `travel_graph.py`'s own computed hop
    (`Hop(from_map=PewterCity, from_x=39, from_y=16, to_map=Route3,
    to_x=0, to_y=8)`) - with `_give_overpowered_party`'s own party already
    in place and wild encounters already disabled (reapplied below anyway,
    per `_disable_wild_encounters`'s own docstring on why every map
    transition needs it reapplied).

    Captured from `pewter_gym_interior.state` (past Route 2's boulder maze,
    Viridian Forest, and Pewter's own streets already) rather than a fresh
    cold boot, the same reuse `_load_pewter_gym_interior_fixture` itself
    is built on - none of that ground is what this fixture's own crossing
    exercises.

    ## The real gate: `EVENT_BEAT_BROCK`, not the Boulder Badge

    #113's own issue text names a real, useful lead: walking east out of
    Pewter Gym before beating Brock triggers a scripted escort
    ("You're a trainer! Follow me!") that auto-walks the player back
    toward the gym - confirmed directly to move the player from world
    `(36, 17)` on Pewter City all the way back to `(11, 18)`, a 25-tile
    round-trip - and the issue's own notes guessed this was gated on
    `wObtainedBadges`'s Boulder Badge bit. Setting that bit directly
    (`_set_badges`'s own pattern) does *not* clear it, confirmed directly
    (the escort still fires with the badge bit set) - `pret/pokered`'s
    `scripts/PewterCity.asm` (`PewterCityCheckPlayerLeavingEastScript`)
    settles it precisely:

    ```
    PewterCityCheckPlayerLeavingEastScript:
        CheckEvent EVENT_BEAT_BROCK
        ret nz
        ...
        ld hl, PewterCityPlayerLeavingEastCoords
        call ArePlayerCoordsInArray
        ret nc
        ...
        ld a, TEXT_PEWTERCITY_YOUNGSTER
        ldh [hTextID], a
        jp DisplayTextID

    PewterCityPlayerLeavingEastCoords:
        dbmapcoord 35, 17
        dbmapcoord 36, 17
        dbmapcoord 37, 18
        dbmapcoord 37, 19
        db -1 ; end
    ```

    It's `EVENT_BEAT_BROCK` (`_EVENT_BEAT_BROCK_BIT`, set the same direct
    way `_bypass_oaks_route_1_interception` sets a different event flag)
    that gates it, checked against four specific Pewter City tiles - and
    setting it directly clears the escort completely, the same "set
    prerequisite story state directly" pattern the issue itself names.

    ## Why a captured state, not a recorded button sequence

    Once free of the escort, Pewter City's own east edge has exactly one
    real, walkable crossing onto Route 3: world tiles `(39, 16)` through
    `(39, 19)` - found by an exhaustive, save-state-forked BFS (every
    reachable tile from the gym's exit walked via real button presses,
    not `game_area_collision()`'s static reads, queue emptied naturally
    rather than hitting a depth cap) rather than hand-walked and recorded,
    the same reasoning `_load_pewter_gym_interior_fixture`'s own docstring
    gives for why a captured state beats a fixed button tuple here.

    ## What's past this fixture: a "second blocker" that wasn't

    Reaching any of `cascade_badge`/`got_ss_ticket`/`thunder_badge`/
    `rainbow_badge` (all reachable, per `travel_graph.py`'s own scope,
    only via Route 3 -> Route 4 -> Cerulean City) needs walking further
    east across Route 3 to its own north connection to Route 4 (world tile
    `(59, 0)`, `travel_graph.py`'s own computed hop). The same exhaustive
    BFS approach, extended east from this fixture's own landing tile (and
    resolving every sight-triggered trainer it met along the way with
    `_give_overpowered_party`), found what looked like a real, bounded,
    dead-end region - every tile reachable by ordinary walking caps out at
    world x=22 (out of Route 3's full 70-tile width) across the whole
    y=4-13 band tried, confirmed against a published Route 3 map
    (serebii.net's own `kanto-rb` map image) showing a solid boulder-cluster
    obstacle starting around world x=23 with no gap found in that band.

    **That conclusion was wrong, and is recorded here only as the cautionary
    note it is.** `tileset_collision.py`'s static whole-map decode found a
    real ordinary-walkable path to `(59, 0)`, and it was walked for real to
    capture `_load_route3_to_route4_fixture`'s own state. What the BFS
    actually hit was Route 3's `y=7` wall: the gap it never found is a
    *single tile*, and the route to it goes north and back down rather than
    east along the band it searched. Its own "needs a from-scratch static
    ROM/tileset collision decode" guess about what a future pass would need
    was the correct next step - see that fixture's docstring for the real
    shape and why a 9x10-window, no-memory on-screen search can't find it.
    """
    with _PEWTER_TO_ROUTE3_STATE_PATH.open("rb") as f:
        pyboy.load_state(f)
    pyboy.tick(1, False)
    _apply_fixture_prerequisites(pyboy)


def test_pewter_to_route3_fixture_lands_on_a_rom_verified_tile(pyboy_outdoors):
    """#113's own acceptance criterion: the captured fixture crosses
    Pewter City's real connection to Route 3 and lands at Route 3's own
    tile `(0, 8)` - `travel_graph.py`'s own computed hop tile, not a
    hand-typed guess - with no dialog or battle left open (safe to build
    further milestone tests on top of once #113's own follow-up finds a
    way past the second blocker `_load_pewter_to_route3_fixture`'s
    docstring documents).
    """
    _load_pewter_to_route3_fixture(pyboy_outdoors)

    state = extract_game_state(pyboy_outdoors)
    assert state.map_id == _MAP_ROUTE_3
    assert (state.player_x, state.player_y) == (0, 8)
    assert not state.dialog_open
    assert not state.battle.in_battle


_ROUTE3_TO_ROUTE4_STATE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "route3_to_route4.state"
)
_MAP_ROUTE_4 = 15


def _load_route3_to_route4_fixture(pyboy: PyBoy) -> None:
    """#113's remaining criterion: loads a captured save state that has
    walked the width of Route 3 and crossed its north connection to Route 4,
    landing at Route 4's own tile `(9, 17)` - again `travel_graph.py`'s own
    computed hop (`Hop(from_map=Route3, from_x=59, from_y=0, to_map=Route4,
    to_x=9, to_y=17)`), not a hand-typed guess - with
    `_give_overpowered_party`'s party in place and wild encounters disabled
    (reapplied below, per `_disable_wild_encounters`'s own note that every
    map transition reloads the real rate).

    ## PR #115's Route 3 dead end was not real

    `_load_pewter_to_route3_fixture`'s own docstring records a second
    blocker: an exhaustive live BFS from this fixture's own starting tile
    that found every tile reachable by ordinary walking capping out at
    world x=22, read as a boulder cluster with no gap. `tileset_collision.py`
    (#113's follow-up) decodes the same question statically, from the ROM's
    own block/collision tables over the whole map at once, and finds a real
    ordinary-walkable path from `(0, 8)` all the way to the crossing tile -
    no ledge involved (`tileset_collision.py`'s "What this doesn't model"
    section explains why that direction of error is impossible: a ledge only
    ever makes a tile read *more* blocked, never less, so a tile this module
    calls walkable can't secretly be one).

    Reading the decoded map explains how a live search misses it. Route 3's
    `y=7` is a near-solid boulder wall whose only five openings are
    *single tiles* - `x=11, 27, 37, 49, 59` - and the whole lower band
    (`y=8`-`13`) is cut off from the upper band horizontally at `x=38`-`43`
    on every single row. So going east along the ground from the landing
    tile really is impossible past about x=37, exactly as the old BFS
    reported, and Route 4's connection row (`y=0`) is only walkable at
    `x=57`-`63`. The way through is vertical, not lateral: up through the
    `x=37` gap, east along the `y=4`/`y=5` band (which does span `x=34`-`49`
    where the ground band doesn't), back *down* through the `x=49` gap, east
    along `y=10`/`y=11`, then up the `x=59` column through the last gap. The
    published walkthrough PR #115 cross-checked was right that the real
    route "goes north" and wrong that it needs ledges.

    ## The 8 stationary trainers are a second, separate obstacle layer

    `rom_maps.parse_map(rom, 14).objects` lists 8 records with `.trainer`
    set, at `(10, 6) (14, 4) (16, 9) (19, 5) (23, 4) (22, 9) (24, 6)
    (33, 10)`. A trainer's own sprite tile is invisible to background-tile
    collision - `tileset_collision.py` decodes the map's *terrain*, so it
    still calls those tiles walkable - so each one had to be added to the
    pathfinder's blocked set by hand. Their sight lines are deliberately
    *not* avoided: walking into one is an ordinary, winnable battle (5 of
    the 8 triggered on this route and were fought for real with
    `_give_overpowered_party`/`_resolve_any_battle`), it just can't be the
    tile the path steps onto.

    ## Why a captured state again

    Same reasoning as the two fixtures this one builds on: threading a
    sequence of single-tile gaps across four screens, with an on-screen A*
    that sees a 9x10 window and keeps no memory of where it has been, and
    cannot see any of the 8 trainer tiles above, is exactly the case
    `_walk_toward` is unreliable for (see
    `_load_pewter_gym_interior_fixture`'s docstring for the same conclusion
    on Pewter's streets). Captured by following a distance field computed
    from the static decode - 92 `execute_button` steps from `(0, 8)` to
    `(59, 0)`, re-deriving the next step from the player's *actual* live
    position each time so a battle's displacement can't desync the route -
    then one more "up" to trip the connection. Regenerate the same way, over
    this file, if the ROM or PyBoy's save-state format ever changes.

    One shared helper *was* fixed rather than worked around:
    `_advance_past_any_encounter`'s unconditional 20-press mash re-opened
    the dialog it had just closed by re-talking to the stationary trainer
    standing right there, which looks identical from outside to an
    unresolvable stuck loop. See its own docstring.
    """
    with _ROUTE3_TO_ROUTE4_STATE_PATH.open("rb") as f:
        pyboy.load_state(f)
    pyboy.tick(1, False)
    _apply_fixture_prerequisites(pyboy)


def test_route3_to_route4_fixture_lands_on_a_rom_verified_tile(pyboy_outdoors):
    """#113's acceptance criterion: the fixture crosses Route 3's own north
    connection and lands on Route 4 at `(9, 17)` -
    `travel_graph.py`'s own computed hop tile for that crossing - with no
    dialog or battle left open, so #99's four remaining milestone tests can
    be built on top of it.
    """
    _load_route3_to_route4_fixture(pyboy_outdoors)

    state = extract_game_state(pyboy_outdoors)
    assert state.map_id == _MAP_ROUTE_4
    assert (state.player_x, state.player_y) == (9, 17)
    assert not state.dialog_open
    assert not state.battle.in_battle


_MAP_MT_MOON_B2F = 61


def test_the_walk_planner_never_plans_a_step_the_roms_pair_rule_refuses():
    """A planner-level regression for `crosses_blocked_pair`.

    Mt Moon B2F's `(5, 7)` stair is the only way out of the dungeon towards
    Route 4, and until the pair rule reached `_route_across_map` the planner
    happily routed a walk from the `(25, 9)` stair landing to it - a route
    straight through twelve `CAVERN $20 -> $05` crossings, which is where every
    one of this repo's recorded `down (n, 11) -> (n, 12)` refusals came from.
    The plan was not merely hard to walk; the ROM refuses each of those steps,
    so no amount of pressing walks it. With the rule in place the same query
    answers `None`, and the floor's real trunk road - from `(21, 17)`, to the
    mouth of the fossil corridor at `(12, 7)` - still plans.
    """
    assert _route_across_map(_MAP_MT_MOON_B2F, (25, 9), (5, 7), frozenset()) is None
    assert (
        _route_across_map(_MAP_MT_MOON_B2F, (21, 17), (12, 7), frozenset()) is not None
    )


_MAP_VIRIDIAN_CITY = 1
_MAP_ROUTE_1 = 12
_MAP_VIRIDIAN_GYM = 45

# The seven badges Earth Badge excluded. Not an arbitrary "enough progress"
# stand-in: `pret/pokered`'s `scripts/ViridianCity.asm` tests for exactly this
# value with `cp`, not a mask - `ld a, [wObtainedBadges]` /
# `cp ~(1 << BIT_EARTHBADGE)` - so `wObtainedBadges == 0x7F` is what the ROM
# itself accepts to open the Gym, and `_set_badges` is the same primitive
# `cascade_badge`'s milestone needs for its own reasons.
_GYM_OPENING_BADGES: tuple[str, ...] = (
    "BOULDERBADGE",
    "CASCADEBADGE",
    "THUNDERBADGE",
    "RAINBOWBADGE",
    "SOULBADGE",
    "MARSHBADGE",
    "VOLCANOBADGE",
)


def _bump_into_viridian_gym_door(pyboy: PyBoy, max_bumps: int = 3) -> None:
    """Presses "up" into the Gym's door tile until the warp fires.

    More than one bump can be needed, and the ROM says why: the lock branch
    above ends by latching `SCRIPT_VIRIDIANCITY_PLAYER_MOVING_DOWN` and pushing
    the player back down the one tile they were standing on, so a first bump
    that lands while that script still owns the map answers with its text and
    swallows the press (`xor a` / `ldh [hJoyHeld], a`, verbatim). Measured both
    ways: with the story gate released the second bump warps; with nothing set
    at all, eight bumps in a row produced eight lock dialogs and no warp.
    """
    start_map = extract_game_state(pyboy).map_id
    for _ in range(max_bumps):
        _advance_past_any_encounter(pyboy)
        execute_button(pyboy, "up")
        pyboy.tick(90, True)
        if extract_game_state(pyboy).map_id != start_map:
            return


def _leave_and_reenter_viridian_gym(pyboy: PyBoy) -> None:
    """Walks back out of the Gym and in through the same door, which is how a
    beaten Gym trainer's sprite is removed.

    Gen 1 leaves a defeated trainer standing on whatever tile the fight ended on
    for the rest of the visit. His object record's flag is set by the win, but
    the sprites on a map are only rebuilt from those records by
    `LoadObjectEvents`, which runs on the next load of the map - so the win is
    half the answer, and the reload is the other half. Measured on the corridor
    this is for: `game_area_collision()` reports the tile north of `(10, 4)` as
    walkable and `tileset_collision`'s decode agrees, yet pressing into it sixty
    times produces neither a step nor a battle, because the Hiker whose object
    record is at `(10, 1)` ended the fight on that tile and is in neither of
    those two sources. It is the same blind spot #113 wrote up for Route 3's
    trainers - sprites are a layer the collision grid does not contain - with the
    extra twist that here he is a sprite the walk has already beaten.

    Getting out needs the presses in a particular order. A D-pad press made
    while a text box is up is thrown away, and the box is up because answering
    the NPC standing directly in front of the player is what puts it there -
    #113's finding, and the reason `_fight_what_is_in_the_way` presses "a" at
    all. So the way out is "a" and then the step, which is the ordering
    `vgym_escape.py` measured as the one that moves the player out of that
    corridor ("then down: (10,5) ... moved!"), while "up" - into him - never
    does. It takes the pair more than once, and not because the first is
    ignored: which half of the pair does the work depends on whether his quote
    is already up when the pair starts, so the first "a" can be the press that
    opens it rather than the one that closes it. Measured at (10, 4) both ways -
    one pair and the walk to the door came back (10, 4); the pair repeated, the
    door walk came back (16, 17).
    """
    leaving = extract_game_state(pyboy)
    for _ in range(6):
        execute_button(pyboy, "a")
        pyboy.tick(10, True)
        execute_button(pyboy, "down")
        pyboy.tick(20, True)
        now = extract_game_state(pyboy)
        if (now.player_x, now.player_y) != (leaving.player_x, leaving.player_y):
            break
    assert _walk_tiles(pyboy, 16, 17, max_steps=150, map_id=_MAP_VIRIDIAN_GYM) == (
        _MAP_VIRIDIAN_GYM,
        16,
        17,
    )
    _cross_map_edge(pyboy, "down")
    assert extract_game_state(pyboy).map_id == _MAP_VIRIDIAN_CITY
    _bump_into_viridian_gym_door(pyboy)
    state = extract_game_state(pyboy)
    assert (state.map_id, state.player_x, state.player_y) == (
        _MAP_VIRIDIAN_GYM,
        16,
        17,
    )
    # Both halves of the room's state came back with the map: its own wild rate
    # and the party the overpowered write replaced.
    _disable_wild_encounters(pyboy)
    _give_overpowered_party(pyboy)


def test_walking_to_viridian_gym_giovanni_reaches_a_rom_verified_tile(
    pyboy_outdoors,
):
    """#114's root cause, and #100's boot verification for `earth_badge`:
    Viridian Gym's VIRIDIANGYM_GIOVANNI object (`milestone_targets.py`), map
    45 tile (2, 1).

    #113's static parse put the city's warp record for the Gym at `(32, 7)`,
    and #114's report doubted it. The coordinate is right, and so is the
    assumption behind it - ordinary walking does reach that door. Standing on
    `(32, 8)` due south of it, pressing "up" lands the player on map 45 at
    `(16, 17)`, which is `travel_graph`'s own landing tile for the hop down to
    the tile. What blocks the door is the story, and
    `pret/pokered`'s `scripts/ViridianCity.asm` spells out the check:

        ViridianCityCheckGymOpenScript:
                CheckEvent EVENT_VIRIDIAN_GYM_OPEN
                ret nz
                ld a, [wObtainedBadges]
                cp ~(1 << BIT_EARTHBADGE)
                jr nz, .gym_closed
                SetEvent EVENT_VIRIDIAN_GYM_OPEN
                ret
        .gym_closed
                ld a, [wYCoord]
                cp 8
                ret nz
                ld a, [wXCoord]
                cp 32
                ret nz
                ld a, TEXT_VIRIDIANCITY_GYM_LOCKED
                ldh [hTextID], a
                call DisplayTextID
                ...
                call ViridianCityMovePlayerDownScript

    Those are the city and the tile this test stands on - `[wYCoord] == 8`,
    `[wXCoord] == 32` - and the screenshot of a run with nothing set is the
    script's own words, "The GYM's doors are locked...", with the Gambler at
    `(30, 8)` nearby supplying "This #MON GYM is always closed." Measured:
    eight bumps with nothing set, eight lock dialogs, no warp; the same bump
    after `_GYM_OPENING_BADGES` goes straight through. `EVENT_VIRIDIAN_GYM_OPEN`
    is the other release the script allows, and setting the flag alone works
    too - the badge write is the one used here because it's the condition the
    game itself has to reach to open the Gym at all, and it's state
    `_set_badges` already knows how to write.

    The second gate is one screen south of the door and is the reason two
    exhaustive live BFS passes reported the door itself as unreachable. It is
    the same shape, one map-tile earlier in the route, and again the ROM
    supplies it verbatim - `scripts/ViridianCity.asm`, immediately after the
    block above, run every frame by `ViridianCityDefaultScript`:

        ViridianCityCheckGotPokedexScript:
                CheckEvent EVENT_GOT_POKEDEX
                ret nz
                ld a, [wYCoord]
                cp 9
                ret nz
                ld a, [wXCoord]
                cp 19
                ret nz
                ld a, TEXT_VIRIDIANCITY_OLD_MAN_SLEEPY
                ldh [hTextID], a
                call DisplayTextID
                xor a
                ldh [hJoyHeld], a
                call ViridianCityMovePlayerDownScript

    The sleeping Old Man is therefore not only a sprite standing in the way:
    without `EVENT_GOT_POKEDEX`, standing on `(19, 9)` at all - the one tile the
    route to the Gym crosses - answers with "You can't go through here! This is
    private property!" and a shove back down the tile the player was standing
    on, and the script latches `SCRIPT_VIRIDIANCITY_PLAYER_MOVING_DOWN` to do
    it. This is the same release `_load_pewter_gym_interior_fixture` already
    writes for its own reason on the far side of town, and `_walk_tiles` reaches
    `(32, 8)` with it set.

    Which is also the correction this file owed #114's report. The door tile and
    the tiles south of it are reachable by ordinary walking, and the note that
    they were not read a correct BFS result backwards: `game_area_collision()`
    does not see sprites (#113 wrote that up for Route 3's trainers), so a
    walker that trusts it plans straight through a standing NPC, presses into
    him, never moves, and marks everything behind him unreachable. Measured
    against `tileset_collision`'s own decode with the town's object records
    added to the solid set: the shortest route from Route 1's north edge to
    `(32, 8)` is 57 steps; blocking any *one* of the town's nine object tiles
    leaves it at the same 57, so the Gambler at `(30, 8)` is not the single
    chokepoint that note made him out to be; and blocking `(19, 8)` - the tile
    north of the Old Man's - is the one that costs anything, pushing the plan to
    81 steps round the west side of town. The walk below takes the 57.

    Inside, the room has two mechanics of its own, both recorded where they are
    handled: `_SPINNER_TILE_IDS` keeps the route off `GymSpinnerArrows`, and the
    object records put sight-triggered Gym trainers at `(10, 1)` and `(10, 7)`,
    the two ends of the one-tile-wide corridor that - with the `(7, 2)`/`(7, 3)`
    notch it feeds - is the only way into Giovanni's pocket, so every route the
    tile decode finds to him runs up it. Their answer to a walk that does not
    beat them is to stand on the tile being pressed into and talk, which is what
    `_press_and_settle`'s `fought` flag is for; and beating them is only half of
    it, because Gen 1 leaves the sprite standing where the fight ended until the
    next `LoadObjectEvents`. `_leave_and_reenter_viridian_gym` is the walk's
    way of asking for that reload - out of the Gym's door and back in - and the
    second attempt at `(2, 2)` is what it is for.
    """
    _bypass_oaks_route_1_interception(pyboy_outdoors)
    _disable_wild_encounters(pyboy_outdoors)
    _give_overpowered_party(pyboy_outdoors)
    _set_badges(pyboy_outdoors, _GYM_OPENING_BADGES)
    _set_event_flag(pyboy_outdoors, _EVENT_GOT_POKEDEX)

    _walk_toward(pyboy_outdoors, 10, 1)
    _cross_map_edge(pyboy_outdoors, "up")
    assert extract_game_state(pyboy_outdoors).map_id == _MAP_ROUTE_1

    _walk_toward(pyboy_outdoors, 12, 0)
    _cross_map_edge(pyboy_outdoors, "up")
    assert extract_game_state(pyboy_outdoors).map_id == _MAP_VIRIDIAN_CITY

    assert _walk_tiles(
        pyboy_outdoors, 32, 8, max_steps=200, map_id=_MAP_VIRIDIAN_CITY
    ) == (_MAP_VIRIDIAN_CITY, 32, 8)
    _bump_into_viridian_gym_door(pyboy_outdoors)
    state = extract_game_state(pyboy_outdoors)
    assert state.map_id == _MAP_VIRIDIAN_GYM
    # `travel_graph`'s own landing tile for the hop, arrived at through the
    # door rather than written there.
    assert (state.player_x, state.player_y) == (16, 17)

    _advance_past_any_encounter(pyboy_outdoors)
    if _walk_tiles(pyboy_outdoors, 2, 2, max_steps=200, map_id=_MAP_VIRIDIAN_GYM) != (
        _MAP_VIRIDIAN_GYM,
        2,
        2,
    ):
        # The walk beat the Hiker at the head of the corridor and then found his
        # sprite still on the tile it needed: measured at (10, 4), with the tile
        # north of it refused for the rest of the visit. He only leaves with the
        # next load of the map.
        _leave_and_reenter_viridian_gym(pyboy_outdoors)

    assert _walk_tiles(
        pyboy_outdoors, 2, 2, max_steps=220, map_id=_MAP_VIRIDIAN_GYM
    ) == (_MAP_VIRIDIAN_GYM, 2, 2)
    _advance_past_any_encounter(pyboy_outdoors)
    # The last step of that walk comes in along row 2, so the player is standing
    # on `(2, 2)` facing *west* - into an empty tile, where "a" answers nothing
    # at all. A Gen 1 step that collides still turns the player, so one press
    # into Giovanni's own tile is what faces him; the tile is his and the walk
    # cannot take it, which is the point of the press.
    execute_button(pyboy_outdoors, "up")
    pyboy_outdoors.tick(20, True)

    row_before = _dialog_row_tiles(pyboy_outdoors)
    pyboy_outdoors.button("a", 2)
    dialog_visible = False
    for _ in range(10):
        pyboy_outdoors.tick(30, True)
        # `_dialog_text_visible` is not usable as the check on this map - it is
        # 18 tiles tall, exactly the window's height, so its own docstring's
        # caveat ("text *could* be showing") is permanently true here - and
        # neither is a bare `_in_battle`, which is only Giovanni's *second*
        # answer. So: the tile this file trusts, a text row that moved because
        # of the press, plus one of the two things the milestone is expected to
        # do.
        state = extract_game_state(pyboy_outdoors)
        if (state.dialog_open or _in_battle(pyboy_outdoors)) and _dialog_row_tiles(
            pyboy_outdoors
        ) != row_before:
            dialog_visible = True
            break
    assert dialog_visible


def test_milestone_travel_graph_hops_viridian_city_into_viridian_gym():
    """#114's other half: `build_hops` needed one map in `MILESTONE_MAP_IDS`,
    not a new edge type. Viridian City's own warp record for the Gym is real
    ROM data, so once map 45 is in scope at all the hop falls out of the
    parser - and it comes out with the door tile on the city side and the
    landing tile the walk above actually arrives on, on the gym side.
    """
    hop = next_hop(_travel_graph(), _MAP_VIRIDIAN_CITY, _MAP_VIRIDIAN_GYM)
    assert hop is not None
    assert (hop.from_x, hop.from_y) == (32, 7)
    assert (hop.to_x, hop.to_y) == (16, 17)


# #100: per-milestone boot verification, batch 2 (parent #96, sibling of
# #99's batch 1 above). `_walk_to_milestone_target` (unlike #99's
# `_walk_toward`) lets the target sit on a different map than wherever the
# player currently is - #102's cross-map routing does the actual crossing,
# one travel-graph hop per `execute_navigation_macro` call, as long as
# every map on the route is in `travel_graph.MILESTONE_MAP_IDS` (extended
# per milestone below).
#
# earth_badge is covered above, by `test_walking_to_viridian_gym_giovanni_
# reaches_a_rom_verified_tile` - see that test's own docstring for the two
# things that were actually in the way there, and for why the "unreachable
# door" note this block used to carry was wrong.
#
# cascade_badge/got_ss_ticket/thunder_badge/rainbow_badge (#99's original
# remaining 4, all reachable only via Route 3 -> Route 4 -> Cerulean City
# per `travel_graph.py`'s own scope) are NOT covered yet. Their blocker is
# the same second obstacle layer one screen further on: Route 4's own
# decoded walkability is right and useless on its own, because a wall at
# x=20-23 splits the map into the pocket Route 3's south connection lands in
# and the eastern side that owns the Cerulean connection, and the only thing
# joining them is Route 4's pair of warps into Mt Moon - a three-floor
# dungeon. `travel_graph.py`'s own "A known false edge" section documents
# what that does to a route this module returns; the crossing itself is
# scripted work rather than graph routing.


def _milestone(target_x: int | None = None, target_y: int | None = None) -> Milestone:
    return Milestone(
        milestone_id="test_milestone",
        description="a milestone used only by this test",
        target=MilestoneTarget(
            map_id=40, map_name="Oaks Lab", target_x=target_x, target_y=target_y
        ),
    )


def test_resolve_navigation_target_returns_none_when_there_is_no_current_milestone():
    assert resolve_navigation_target(None, 40) is None


def test_resolve_navigation_target_returns_none_without_verified_tile_coordinates():
    assert resolve_navigation_target(_milestone(), 40) is None


def test_resolve_navigation_target_returns_the_milestones_tile_coordinates():
    milestone = _milestone(target_x=4, target_y=5)

    target = resolve_navigation_target(milestone, current_map=40)

    assert target == NavigationTarget(map_id=40, x=4, y=5)


def test_resolve_navigation_target_resolves_a_target_across_maps_with_a_known_route():
    """`current_map` differing from the milestone's own map isn't itself a
    reason to return `None` (#102): Pallet Town -> Oak's Lab is a known
    single-hop route in `travel_graph.py`'s milestone graph."""
    milestone = _milestone(target_x=4, target_y=5)

    target = resolve_navigation_target(milestone, current_map=_MAP_PALLET_TOWN)

    assert target == NavigationTarget(map_id=40, x=4, y=5)


def test_resolve_navigation_target_returns_none_without_a_known_route_yet():
    """A milestone whose target map has no known travel-graph route yet
    (ADR-0002's incremental build) degrades to `None`, not a crash - the
    same "can't resolve a destination" signal as unverified coordinates."""
    milestone = Milestone(
        milestone_id="test_milestone_not_yet_routed",
        description="a milestone used only by this test",
        target=MilestoneTarget(
            map_id=_MAP_NOT_YET_ROUTED,
            map_name="Cerulean Gym",
            target_x=4,
            target_y=2,
        ),
    )

    assert resolve_navigation_target(milestone, current_map=_MAP_PALLET_TOWN) is None


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
