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
    EXP is maxed so any EXP gained from winning doesn't get used to
    recompute real (tiny) stats from the species' own growth curve -
    confirmed to happen otherwise: a single win silently dropped this
    from level 100 back to level ~2, stats and all, regardless of what
    MON_EXP was set to going in. `_resolve_any_battle` re-applies this
    after every battle for the same reason - the recalculation happens
    the instant EXP is gained, not something set once up front survives.
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
    mon[14:17] = (0xFFFFFF).to_bytes(3, "big")  # EXP: maxed
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


def _advance_past_any_encounter(pyboy: PyBoy) -> None:
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
    """
    if not (_dialog_text_visible(pyboy) or _in_battle(pyboy)):
        return
    for _ in range(20):
        if _in_battle(pyboy):
            break
        execute_button(pyboy, "a")
        pyboy.tick(20, True)
        if not (_dialog_text_visible(pyboy) or _in_battle(pyboy)):
            # #113: stop here. Pressing "a" any further re-talks to the
            # stationary NPC this player is left standing adjacent to and
            # facing, re-opening the very dialog just closed - which from
            # outside looks like an unresolvable stuck loop. A trainer's
            # "already beaten" canned line takes exactly this branch (no
            # battle to break into), so the old unconditional mash kept
            # pressing 14+ more times after the dialog was already gone.
            # Settled and re-checked once, because a battle's own start-up
            # shows a clear frame between its last dialog page and
            # `wIsInBattle` going nonzero.
            pyboy.tick(30, True)
            if not (_dialog_text_visible(pyboy) or _in_battle(pyboy)):
                return
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
        # cleared).
        if not _dialog_text_visible(pyboy):
            break
        execute_button(pyboy, "a")
        pyboy.tick(20, True)


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
    _disable_wild_encounters(pyboy)


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
    _disable_wild_encounters(pyboy)


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
    _disable_wild_encounters(pyboy)


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


# #100: per-milestone boot verification, batch 2 (parent #96, sibling of
# #99's batch 1 above). `_walk_to_milestone_target` (unlike #99's
# `_walk_toward`) lets the target sit on a different map than wherever the
# player currently is - #102's cross-map routing does the actual crossing,
# one travel-graph hop per `execute_navigation_macro` call, as long as
# every map on the route is in `travel_graph.MILESTONE_MAP_IDS` (extended
# per milestone below).
#
# earth_badge (Viridian Gym, map 45) is NOT covered below despite being a
# direct warp off Viridian City (already in scope alongside Pallet Town/
# Route 1) - confirmed via an exhaustive real-emulator BFS (save-state-
# forked, every reachable tile from the Route 1 entrance explored, 530
# distinct tiles, queue emptied naturally rather than hitting a depth cap)
# that the gym's own door tile, and the two tiles south of it, are
# genuinely unreachable by ordinary walking from Viridian City's only
# in-scope entrance - with or without the other 7 badges set. This isn't
# the known "door tiles misread as collision-blocked" limitation
# `_nudge_across_hop` already works around (that's a local, few-tile
# nudge; this is a real, town-wide unreachable region) - it needs either a
# corrected ROM-parsed door coordinate or real visual investigation to
# find the actual approach, neither of which this pass had budget for.
#
# cascade_badge/got_ss_ticket/thunder_badge/rainbow_badge (#99's original
# remaining 4, all reachable only via Route 3 -> Route 4 -> Cerulean City
# per `travel_graph.py`'s own scope) are also NOT covered below, for the
# same class of reason as earth_badge above - see
# `_load_pewter_to_route3_fixture`'s own docstring (#113) for the second,
# separate blocker its own investigation found past Pewter City's escort:
# a real, exhaustively-confirmed dead end partway across Route 3 itself.


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
