import threading
import time
from typing import cast

from pyboy import PyBoy

from jev_plays_pokemon.emulator_clock import start_emulator_clock


class _FakePyBoy:
    """Call-counting fake standing in for a real `pyboy.PyBoy` - records
    every `tick(...)` call's arguments, no real emulator anywhere."""

    def __init__(self) -> None:
        self.tick_calls: list[tuple[int, bool]] = []

    def tick(self, frames: int, render: bool) -> None:
        self.tick_calls.append((frames, render))


class _RaisingPyBoy:
    def tick(self, frames: int, render: bool) -> None:
        raise RuntimeError("boom")


def test_start_emulator_clock_ticks_pyboy_with_render_true_while_running():
    fake = _FakePyBoy()
    lock = threading.Lock()

    clock = start_emulator_clock(cast(PyBoy, fake), lock, interval_seconds=0.01)
    time.sleep(0.1)
    clock.stop()

    assert len(fake.tick_calls) >= 2
    assert all(call == (1, True) for call in fake.tick_calls)


def test_start_emulator_clock_returns_a_handle_that_stops_cleanly():
    lock = threading.Lock()

    clock = start_emulator_clock(cast(PyBoy, _FakePyBoy()), lock, interval_seconds=0.01)
    clock.stop()


def test_start_emulator_clock_acquires_the_shared_lock_for_each_tick():
    fake = _FakePyBoy()
    lock = threading.Lock()

    # Hold the lock ourselves before starting the clock: if the clock
    # ticked without acquiring it, tick_calls would grow immediately.
    lock.acquire()
    try:
        clock = start_emulator_clock(cast(PyBoy, fake), lock, interval_seconds=0.01)
        time.sleep(0.05)
        assert fake.tick_calls == []
    finally:
        lock.release()

    time.sleep(0.05)
    clock.stop()
    assert len(fake.tick_calls) >= 1


def test_start_emulator_clock_survives_a_tick_that_raises():
    # A tick failure is logged and the clock keeps running onto the next
    # tick rather than silently dying - a dead clock would freeze the feed
    # with no visible error.
    clock = start_emulator_clock(
        cast(PyBoy, _RaisingPyBoy()), threading.Lock(), interval_seconds=0.01
    )
    time.sleep(0.05)
    clock.stop()
