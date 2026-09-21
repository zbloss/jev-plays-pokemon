"""Free-running emulator clock: keeps PyBoy advancing independent of Jev.

Part of #109 - the tactical loop only ever ticked `pyboy` from inside a turn
(`decision.make_pyboy_action_executor`'s dispatch, `main.build_dialog_text_
source`'s render step), so a turn that stalls waiting on Jev - whether real
network latency or a deliberately slow `--max-calls-per-second` (#59's debug
mode, `rate_limit.py`) - froze the entire emulator, not just the decision
loop. `frame_capture.py`'s background capture timer samples whatever's
currently rendered at `--display-fps`, but a frozen `pyboy` just gives it the
same frame back every tick, so the `/viewer` feed looked stalled too.

`start_emulator_clock` runs its own background daemon thread ticking `pyboy`
once per `interval_seconds` (`main.main` derives this from `--display-fps`,
the same interval `frame_capture.py`'s timer uses, so the two stay in step),
so the game world keeps moving at the display's own pace no matter how the
Jev call rate is configured.

This supersedes #81's "only one thread ever ticks pyboy" constraint, which
existed because nothing synchronized concurrent access. `lock` (shared with
every other `pyboy` touchpoint - see `main.main`'s wiring) is what makes
cross-thread ticking safe now: this clock and the tactical loop's own ticks
are serialized through it, so they interleave rather than race. A tick this
clock takes is always a single, uninterrupted `pyboy.tick(1, True)` call
under the lock - short enough that it never meaningfully delays a turn
waiting on the same lock.
"""

from __future__ import annotations

import logging
import threading

from pyboy import PyBoy

logger = logging.getLogger(__name__)


class EmulatorClockHandle:
    """Handle to the background daemon thread driving the idle tick.

    Mirrors `frame_capture.FrameCaptureTimer`'s shape: a small handle
    exposing `stop()`, so callers never need to know the timer's innards to
    control it.
    """

    def __init__(self, thread: threading.Thread, stop_event: threading.Event) -> None:
        self._thread = thread
        self._stop_event = stop_event

    def stop(self) -> None:
        """Stop the clock and block until the background thread has exited."""
        self._stop_event.set()
        self._thread.join()


def start_emulator_clock(
    pyboy: PyBoy, lock: threading.Lock, interval_seconds: float
) -> EmulatorClockHandle:
    """Run `pyboy.tick(1, render=True)` on its own background daemon thread,
    once per `interval_seconds`, forever - independent of, and never
    blocking on, the tactical loop's own turn cadence.

    `lock` is acquired for just the single tick call, then released before
    the next `interval_seconds` wait - so a turn in progress on another
    thread (holding the same lock for its own, possibly multi-tick,
    dispatch) is never starved, and this clock never ticks concurrently
    with it.

    A tick that raises is logged and the loop continues onto the next tick
    rather than killing the thread - a dead clock would otherwise silently
    freeze the feed with no error visible to a viewer, exactly the symptom
    #109 fixes.
    """
    stop_event = threading.Event()

    def _run() -> None:
        while not stop_event.is_set():
            try:
                with lock:
                    pyboy.tick(1, True)
            except Exception:
                logger.exception("emulator clock tick failed")
            stop_event.wait(interval_seconds)

    thread = threading.Thread(target=_run, name="emulator-clock", daemon=True)
    thread.start()
    return EmulatorClockHandle(thread, stop_event)
