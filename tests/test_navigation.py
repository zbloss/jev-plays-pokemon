import io
from pathlib import Path

import pytest
from pyboy import PyBoy

from jev_plays_pokemon.game_state import extract_game_state
from jev_plays_pokemon.milestones import Milestone, MilestoneTarget
from jev_plays_pokemon.navigation import (
    RAW_BUTTONS,
    NavigationTarget,
    execute_button,
    execute_navigation_macro,
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


def _boot_past_intro(pyboy: PyBoy) -> None:
    """Mash through Pokemon Red's un-skippable boot sequence - title screen,
    "NEW GAME", and the player/rival naming screens - up to the point the
    player has full control in their bedroom.

    Gen 1's intro has no shortcut, and no bundled save state exists for this
    ROM (it's gitignored), so any test that needs a real, controllable game
    state has to replay a fixed input sequence to get there first. This one
    is fully deterministic: PyBoy has no real-time-clock dependency this
    early, since the Pokemon Red cartridge itself has no RTC chip, and this
    sequence was confirmed to land on the same in-game position across
    repeated runs against the real ROM while building this ticket.

    The three phases below mirror the three screens Gen 1's intro forces a
    new save through, reverse-engineered by booting the ROM headless and
    inspecting rendered frames at each step (see this ticket - nothing else
    in this repo's RAM-map research needed real gameplay input before it):

    1. mash through the title screen, "NEW GAME", and the player-naming
       keyboard, up to the rival-naming keyboard;
    2. mash through the rival-naming keyboard itself - its cursor starts on
       the "A" key, so this both fills and submits the name "AAAAAA";
    3. mash through the name confirmation and Oak's closing monologue, which
       ends with the player standing, controllable, in their bedroom.

    Each phase's button hold has to be a single `PyBoy.tick(n)` call, not a
    loop of `n` individual `PyBoy.tick(1)` calls: `PyBoy.button`'s `delay`
    counts *calls* to `tick`, not frames, so spreading the wait across many
    calls releases the button far earlier than intended (see
    `navigation.py`'s "Button timing" docstring for the same distinction).
    """
    for frame in range(1, 45 * 60 + 1):
        pyboy.tick(1, False)
        if frame % 20 == 0:
            pyboy.button("a", 2)
        if frame % 41 == 0:
            pyboy.button("start", 2)

    for _ in range(20):
        pyboy.button("a", 2)
        pyboy.tick(20, False)

    for _ in range(21):
        pyboy.button("a", 3)
        pyboy.tick(40, False)


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
    _boot_past_intro(pyboy)
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
