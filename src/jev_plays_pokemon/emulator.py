"""Boot a Pokemon Red ROM under PyBoy into a live, controllable game state.

This is the emulator-startup half of the end-to-end tactical loop (#23):
`main.py` boots the ROM through here so its first turn already has a real,
player-controllable `GameState` rather than a title screen the loop would
have to button-mash its way out of. The ROM-gated integration test
(`tests/test_main.py`) drives the exact same boot, so the loop is tested
against the same live game state it runs against in production.

Getting a fresh Pokemon Red save to a controllable state has no shortcut and
no bundled save state ships in the repo (the ROM is gitignored, so no
`.state` does either - see `tests/test_game_state.py`), so the intro has to
be replayed as a fixed input sequence every cold boot. The sequence below is
lifted verbatim from the one `tests/test_navigation.py` worked out by
booting this repo's own `pokemon_red.gb` headless and inspecting rendered
frames at each step; it lives here now so the live loop and the navigation
tests share one definition rather than each carrying their own copy.

`save_snapshot`/`load_snapshot`/`boot_or_resume` (#55) are this project's
minimal persistence: a live run's `pyboy.save_state()` overwrites the single
latest snapshot on disk (no history), and `boot_or_resume` loads it back on
the next startup instead of replaying the intro-mash - the intro sequence
below only ever runs again once no snapshot exists yet. A cold boot also
*saves* the state it just mashed its way to, so a snapshot exists from the
first second of every run rather than only after the first thing worth
saving has happened (see `boot_or_resume` for why that matters to #57's
recovery ladder).

Deterministic: the Pokemon Red cartridge has no RTC chip, so nothing in this
sequence depends on wall-clock time, and the same inputs land on the same
in-game position across repeated runs (confirmed while building #20).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from pyboy import PyBoy

# The repo's own ROM at the working-tree root, gitignored (a copyrighted ROM
# isn't redistributable - see tests/test_game_state.py). Callers that keep
# their ROM elsewhere pass an explicit `rom_path`.
DEFAULT_ROM_PATH = Path(__file__).resolve().parents[2] / "pokemon_red.gb"

# The single latest snapshot (#55), alongside the ROM at the working-tree
# root. Also gitignored: a save file is derived, regenerable game state, not
# something to check in.
DEFAULT_SNAPSHOT_PATH = Path(__file__).resolve().parents[2] / "pokemon_red.state"

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
    this module's one-time intro-mash (`boot_past_intro`) ticks with
    `render=False` for speed, which leaves the screen buffer stale right after
    boot - verified against the real ROM, after a run of `render=False` ticks
    `screen.ndarray` is a uniform 255 (std 0.0), and only a `render=True` tick
    makes it real pixels. `navigation.execute_button` (the live per-turn loop)
    ticks with `render=True` itself (#47), so the buffer is continuously fresh
    once play starts; this function remains the vision-fallback dialog decode's
    (#19) explicit guarantee that a frame has been rendered before
    `dialog_vision.capture_screen` reads it, regardless of what came before. One
    extra rendered frame (1/60 s) is negligible and only paid on a turn where a
    dialog is open.
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


def save_snapshot(pyboy: PyBoy, path: str | Path = DEFAULT_SNAPSHOT_PATH) -> None:
    """Save `pyboy`'s complete emulator state to `path`.

    Overwrites whatever was already at `path` - only the single latest
    snapshot is ever kept, no history/retention logic (#55) - so this is
    safe to call as often as a caller likes.
    """
    with open(path, "wb") as snapshot_file:
        pyboy.save_state(snapshot_file)


def load_snapshot(pyboy: PyBoy, path: str | Path = DEFAULT_SNAPSHOT_PATH) -> None:
    """Restore `pyboy`'s complete emulator state from `path`."""
    with open(path, "rb") as snapshot_file:
        pyboy.load_state(snapshot_file)


def make_pyboy_snapshot_saver(
    pyboy: PyBoy, path: str | Path = DEFAULT_SNAPSHOT_PATH
) -> Callable[[], None]:
    """Build a zero-argument save-snapshot seam bound to `pyboy` (#55).

    Mirrors `decision.make_pyboy_action_executor`'s injection shape:
    production callers (`main.main`) just pass `pyboy`, while `main.run_loop`
    only ever sees the returned zero-arg callable, so its tests inject their
    own save spy instead of a real `PyBoy`.
    """

    def save() -> None:
        save_snapshot(pyboy, path)

    return save


def boot_or_resume(
    rom_path: str | Path = DEFAULT_ROM_PATH,
    snapshot_path: str | Path = DEFAULT_SNAPSHOT_PATH,
) -> PyBoy:
    """Boot `rom_path`, resuming from `snapshot_path` if a snapshot exists (#55).

    A snapshot already carries a fully in-game state, so it's loaded onto a
    freshly created (not intro-mashed) instance. With no snapshot on disk -
    every cold start until the first save - this falls back to
    `boot_to_controllable_state`'s fixed intro-mash sequence, unchanged, and
    then immediately saves the controllable state it reached.

    That boot-time save is what makes `stuck_detection.py`'s top recovery
    rung (reload the latest snapshot) actually reachable. Until now a
    snapshot only ever appeared on a milestone completion or
    `main._SAVE_INTERVAL_SECONDS` (600s) - and a run that gets stuck early
    produces neither, so the escalation could only log "no snapshot exists
    yet" and carry on in the same stuck state, re-detecting every ~100 turns
    forever. That is how #59's run billed 570 decisions in ~102s without
    ever breaking out. Reloading the boot state is a slow recovery (back to
    the bedroom, empty party), but it is a real one: the position changes, so
    the stuck streak breaks.
    """
    snapshot_path = Path(snapshot_path)
    if snapshot_path.exists():
        pyboy = create_pyboy(rom_path)
        load_snapshot(pyboy, snapshot_path)
        return pyboy
    pyboy = boot_to_controllable_state(rom_path)
    save_snapshot(pyboy, snapshot_path)
    return pyboy
