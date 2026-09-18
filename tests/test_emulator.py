import numpy as np
import pytest
from pyboy import PyBoy

from jev_plays_pokemon.emulator import (
    DEFAULT_ROM_PATH,
    boot_to_controllable_state,
    create_pyboy,
    render_current_frame,
)
from jev_plays_pokemon.navigation import execute_button

# Same gitignored-ROM gate as the other emulator-backed suites.
pytestmark = pytest.mark.skipif(
    not DEFAULT_ROM_PATH.exists(), reason=f"{DEFAULT_ROM_PATH} not present locally"
)


def _position(pyboy: PyBoy) -> tuple[int, int]:
    memory = pyboy.memory
    return (memory[0xD362], memory[0xD361])


@pytest.fixture
def booted_pyboy():
    pyboy = boot_to_controllable_state(DEFAULT_ROM_PATH)
    yield pyboy
    pyboy.stop(save=False)


def test_create_pyboy_returns_a_usable_headless_instance():
    # create_pyboy only constructs + configures the instance: no intro is run,
    # so the player is not yet in a game and the position reads as unset.
    pyboy = create_pyboy(DEFAULT_ROM_PATH)
    try:
        pyboy.tick(1, False)
        assert isinstance(pyboy, PyBoy)
        assert _position(pyboy) == (0, 0)
    finally:
        pyboy.stop(save=False)


def test_boot_to_controllable_state_leaves_the_player_able_to_walk(
    booted_pyboy: PyBoy,
):
    # "Controllable" is the whole point of the boot: from a cold-booted save,
    # a directional input actually moves the player's RAM position. One settle
    # tick first, matching how the navigation fixtures drive a loaded state.
    boot_done_position = _position(booted_pyboy)
    booted_pyboy.tick(1, False)

    execute_button(booted_pyboy, "down")

    after = _position(booted_pyboy)
    assert after != boot_done_position
    assert after[1] == boot_done_position[1] + 1


def test_boot_to_controllable_state_is_deterministic():
    # The loop and its tests rely on the fixed intro sequence landing on the
    # same in-game spot every cold boot (see emulator.py's docstring) - the
    # cartridge has no RTC, so there is no clock dependency to break this.
    first = boot_to_controllable_state(DEFAULT_ROM_PATH)
    first_position = _position(first)
    first.stop(save=False)

    second = boot_to_controllable_state(DEFAULT_ROM_PATH)
    second_position = _position(second)
    second.stop(save=False)

    assert first_position == second_position


def test_boot_past_intro_moves_off_the_title_screen(booted_pyboy: PyBoy):
    # A controllable state is inside a real map, not the title screen: the
    # player map id is a live, nonzero value by the time the intro is done.
    assert booted_pyboy.memory[0xD35E] != 0


def test_render_current_frame_refreshes_the_stale_screen_buffer():
    # Guards the blank-frame bug: the hot path ticks render=False, which
    # leaves pyboy.screen.ndarray uniform (so the dialog decoder would see a
    # white frame). render_current_frame's render=True tick is what puts real
    # pixels in the buffer - the precondition for dialog_vision.capture_screen.
    pyboy = create_pyboy(DEFAULT_ROM_PATH)
    try:
        for _ in range(30):
            pyboy.tick(1, False)
        assert float(np.std(pyboy.screen.ndarray)) == 0.0

        # Scan rendered frames rather than a specific one: Gen 1's boot has
        # blank-white stretches, but a rendered frame within the first few
        # seconds rasterizes real pixels, which render=False never does.
        rasterized = any(
            (render_current_frame(pyboy), float(np.std(pyboy.screen.ndarray)))[1] > 0.0
            for _ in range(240)
        )
        assert rasterized
    finally:
        pyboy.stop(save=False)
