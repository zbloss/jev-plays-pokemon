"""Boot a Pokemon Red ROM under PyBoy into a live, controllable game state.

This is the emulator-startup half of the end-to-end tactical loop (#23):
`main.py` boots the ROM through here so its first turn already has a real,
player-controllable `GameState` rather than a title screen the loop would
have to button-mash its way out of. The ROM-gated integration test
(`tests/test_main.py`) drives the exact same boot, so the loop is tested
against the same live game state it runs against in production.

Getting a fresh Pokemon Red save to a controllable state has no shortcut and
no bundled save state (the ROM is gitignored, so no `.state` ships either -
see `tests/test_game_state.py`), so the intro has to be replayed as a fixed
input sequence every cold boot. The sequence below is lifted verbatim from
the one `tests/test_navigation.py` worked out by booting this repo's own
`pokemon_red.gb` headless and inspecting rendered frames at each step; it
lives here now so the live loop and the navigation tests share one
definition rather than each carrying their own copy.

Deterministic: the Pokemon Red cartridge has no RTC chip, so nothing in this
sequence depends on wall-clock time, and the same inputs land on the same
in-game position across repeated runs (confirmed while building #20).
"""

from __future__ import annotations

from pathlib import Path

from pyboy import PyBoy

# The repo's own ROM at the working-tree root, gitignored (a copyrighted ROM
# isn't redistributable - see tests/test_game_state.py). Callers that keep
# their ROM elsewhere pass an explicit `rom_path`.
DEFAULT_ROM_PATH = Path(__file__).resolve().parents[2] / "pokemon_red.gb"

_WINDOW_BACKEND = "null"


def create_pyboy(rom_path: str | Path = DEFAULT_ROM_PATH) -> PyBoy:
    """Create an unattended PyBoy instance for `rom_path`.

    `window="null"` runs headless (no SDL window), and
    `set_emulation_speed(0)` unthrottles the emulator to run as fast as the
    host allows - both required for a headless loop and for tests. The
    caller owns the instance and must `pyboy.stop(save=False)` it.
    """
    pyboy = PyBoy(str(rom_path), window=_WINDOW_BACKEND)
    pyboy.set_emulation_speed(0)
    return pyboy


def render_current_frame(pyboy: PyBoy) -> None:
    """Advance one emulator frame with rendering enabled, so screen buffers refresh.

    PyBoy 2.2.0 only writes `pyboy.screen.ndarray` on a tick with `render=True`;
    the entire no-vision hot path (this module's intro-mash, `navigation.
    execute_button`, the tactical loop) ticks with `render=False` for speed,
    which leaves the screen buffer stale - verified against the real ROM, after
    a run of `render=False` ticks `screen.ndarray` is a uniform 255 (std 0.0),
    and only a `render=True` tick makes it real pixels. So the vision-fallback
    dialog decode (#19), which reads that buffer via `dialog_vision.
    capture_screen`, must be preceded by exactly this call, or the decoder is
    handed a blank white frame instead of the dialog. One extra rendered frame
    (1/60 s) is negligible and only paid on a turn where a dialog is open.
    """
    pyboy.tick(1, True)


def boot_past_intro(pyboy: PyBoy) -> None:
    """Mash through Pokemon Red's un-skippable boot sequence - title screen,
    "NEW GAME", and the player/rival naming screens - up to the point the
    player has full control in their bedroom.

    Gen 1's intro forces every new save through three screens, reverse-
    engineered by booting the ROM headless and inspecting rendered frames at
    each step (nothing else in this repo's RAM-map research needed real
    gameplay input before it):

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


def boot_to_controllable_state(
    rom_path: str | Path = DEFAULT_ROM_PATH,
) -> PyBoy:
    """Boot `rom_path` fresh and mash through the intro into a controllable state.

    Returns a live PyBoy instance the caller drives (and stops). This is what
    the tactical loop boots against; the player is controllable in their
    bedroom on return.
    """
    pyboy = create_pyboy(rom_path)
    boot_past_intro(pyboy)
    return pyboy
