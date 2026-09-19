"""Snapshot persistence (#55): save/load seam + startup resume.

Unit tests here drive `emulator.py`'s save/load/resume functions against a
fake PyBoy-like spy - no real PyBoy, ROM, or ticking - mirroring
`decision.py`'s `make_pyboy_action_executor` test style (`test_decision.py`).
A secondary, ROM-gated test at the bottom confirms an actual save/load
round-trip against the real `pokemon_red.gb`, mirroring
`test_emulator.py`/`test_navigation.py`'s real-ROM secondary-test pattern.
"""

from typing import cast

import pytest
from pyboy import PyBoy

from jev_plays_pokemon.emulator import (
    DEFAULT_ROM_PATH,
    boot_or_resume,
    boot_to_controllable_state,
    load_snapshot,
    make_pyboy_snapshot_saver,
    save_snapshot,
)
from jev_plays_pokemon.navigation import execute_button


class _FakePyBoy:
    """Records `save_state`/`load_state` calls; no real emulator underneath."""

    def __init__(self) -> None:
        self.saved_to: list[object] = []
        self.loaded_from: list[object] = []

    def save_state(self, file_like_object) -> None:
        self.saved_to.append(file_like_object)
        file_like_object.write(b"fake-state-bytes")

    def load_state(self, file_like_object) -> None:
        self.loaded_from.append(file_like_object)
        file_like_object.read()


def test_save_snapshot_writes_pyboys_state_to_the_given_path(tmp_path):
    fake = _FakePyBoy()
    path = tmp_path / "snapshot.state"

    save_snapshot(cast(PyBoy, fake), path)

    assert len(fake.saved_to) == 1
    assert path.read_bytes() == b"fake-state-bytes"


def test_save_snapshot_overwrites_rather_than_keeping_history(tmp_path):
    # Only the single latest snapshot is ever kept - a second save replaces
    # the first, no `.1`/`.2` retention files appear alongside it.
    fake = _FakePyBoy()
    path = tmp_path / "snapshot.state"

    save_snapshot(cast(PyBoy, fake), path)
    save_snapshot(cast(PyBoy, fake), path)

    assert list(tmp_path.iterdir()) == [path]


def test_load_snapshot_reads_the_given_path_into_pyboy(tmp_path):
    fake = _FakePyBoy()
    path = tmp_path / "snapshot.state"
    path.write_bytes(b"fake-state-bytes")

    load_snapshot(cast(PyBoy, fake), path)

    assert len(fake.loaded_from) == 1


def test_make_pyboy_snapshot_saver_returns_a_zero_arg_seam_bound_to_pyboy(tmp_path):
    # Mirrors make_pyboy_action_executor's injection shape: production wires
    # this to a real PyBoy; run_loop only ever sees the zero-arg callable.
    fake = _FakePyBoy()
    path = tmp_path / "snapshot.state"

    save = make_pyboy_snapshot_saver(cast(PyBoy, fake), path)
    save()

    assert len(fake.saved_to) == 1
    assert path.exists()


def test_boot_or_resume_boots_fresh_when_no_snapshot_exists(monkeypatch, tmp_path):
    calls: list[str] = []
    monkeypatch.setattr(
        "jev_plays_pokemon.emulator.boot_to_controllable_state",
        lambda rom_path: calls.append("boot_fresh") or "fresh-pyboy",
    )
    monkeypatch.setattr(
        "jev_plays_pokemon.emulator.load_snapshot",
        lambda pyboy, path: pytest.fail("should not load - no snapshot exists"),
    )

    result = boot_or_resume("rom.gb", tmp_path / "missing.state")

    assert result == "fresh-pyboy"
    assert calls == ["boot_fresh"]


def test_boot_or_resume_loads_the_snapshot_when_one_exists(monkeypatch, tmp_path):
    snapshot_path = tmp_path / "snapshot.state"
    snapshot_path.write_bytes(b"fake-state-bytes")
    calls: list[tuple] = []
    monkeypatch.setattr(
        "jev_plays_pokemon.emulator.create_pyboy",
        lambda rom_path: calls.append(("create", rom_path)) or "resumed-pyboy",
    )
    monkeypatch.setattr(
        "jev_plays_pokemon.emulator.load_snapshot",
        lambda pyboy, path: calls.append(("load", pyboy, path)),
    )
    monkeypatch.setattr(
        "jev_plays_pokemon.emulator.boot_to_controllable_state",
        lambda rom_path: pytest.fail("should not intro-mash - a snapshot exists"),
    )

    result = boot_or_resume("rom.gb", snapshot_path)

    assert result == "resumed-pyboy"
    assert calls == [("create", "rom.gb"), ("load", "resumed-pyboy", snapshot_path)]


# -- secondary real-ROM test: an actual save/load round-trip ------------------

pytestmark_e2e = pytest.mark.skipif(
    not DEFAULT_ROM_PATH.exists(), reason=f"{DEFAULT_ROM_PATH} not present locally"
)


@pytestmark_e2e
def test_snapshot_round_trips_real_game_state_against_the_real_rom(tmp_path):
    snapshot_path = tmp_path / "snapshot.state"

    pyboy = boot_to_controllable_state(DEFAULT_ROM_PATH)
    try:
        pyboy.tick(1, False)
        saved_position = (pyboy.memory[0xD362], pyboy.memory[0xD361])
        save_snapshot(pyboy, snapshot_path)

        # Diverge from the saved state via the one directional move already
        # proven safe from this exact boot position
        # (test_emulator.py's test_boot_to_controllable_state_leaves_the_
        # player_able_to_walk), so loading back is the only way the position
        # below could match `saved_position` again.
        execute_button(pyboy, "down")
        assert (pyboy.memory[0xD362], pyboy.memory[0xD361]) != saved_position

        load_snapshot(pyboy, snapshot_path)

        assert (pyboy.memory[0xD362], pyboy.memory[0xD361]) == saved_position
    finally:
        pyboy.stop(save=False)


@pytestmark_e2e
def test_boot_or_resume_resumes_from_a_real_saved_snapshot(tmp_path):
    snapshot_path = tmp_path / "snapshot.state"

    first = boot_to_controllable_state(DEFAULT_ROM_PATH)
    try:
        first.tick(1, False)
        execute_button(first, "down")
        saved_position = (first.memory[0xD362], first.memory[0xD361])
        save_snapshot(first, snapshot_path)
    finally:
        first.stop(save=False)

    resumed = boot_or_resume(DEFAULT_ROM_PATH, snapshot_path)
    try:
        # Resuming loads the saved mid-game state directly - no intro-mash -
        # so the position picks up exactly where the snapshot left off.
        assert (resumed.memory[0xD362], resumed.memory[0xD361]) == saved_position
    finally:
        resumed.stop(save=False)
