"""Debug-mode API call rate limiting.

The overnight verification run for #59 booted successfully, then made 570
consecutive `system_one()` calls in ~102 seconds - every one the same action,
zero progress - before its stuck-recovery escalation (#57) turned out to be a
no-op on a fresh boot (nothing to snapshot-reload to yet). Diagnosing *why*
the loop was stuck needs a human watching each decision go by; at several
calls per second that's impossible, and every call is a real, billed
TypeSafe request.

Two things about that run itself have since been fixed: a cold boot now saves
a snapshot (`emulator.boot_or_resume`), so the reload tier has something to
reload; and the ladder gives up after `stuck_detection.DEFAULT_MAX_RELOADS`
reload that didn't take, instead of re-detecting every ~100 turns forever.
Slowing the loop down is still the only way to actually *watch* a stuck run go
by, which is what this module is for.

`RateLimitedJevClient` is `JevClient`-shaped (same `system_one(state,
questions)` signature as `resilience.ResilientJevClient`, which it composes
with the same way: wherever a real or fake `JevClient` goes), and caps the
rate of calls *through* it so a slow human - or a slow log tail - can keep
up. It is opt-in (`main.py` only wraps with it when a debug rate is
explicitly given) and orthogonal to retry/backoff: composing the two just
means retries are also spaced no closer than the cap, which is harmless.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from typesafe_sdk import Choice


class _JevClientLike(Protocol):
    """Structural shape this module needs - see `resilience._JevClientLike`
    for why this isn't `decision.JevClient` directly (avoids a needless
    import; a real `JevClient` is assignable here with no structural
    mismatch)."""

    def system_one(self, state: Any, questions: Mapping[str, Choice]) -> Any: ...


class RateLimitedJevClient:
    """`JevClient`-shaped wrapper that caps calls to `max_calls_per_second`.

    Enforces a minimum spacing between call *dispatches* (not call
    completions): if the previous call was dispatched less than
    `1 / max_calls_per_second` ago, sleeps off the remainder first. A slow
    underlying call that already exceeds the interval on its own never waits
    on top of that.
    """

    def __init__(
        self,
        jev_client: _JevClientLike,
        *,
        max_calls_per_second: float,
        sleep: Callable[[float], None] = time.sleep,
        time_source: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_calls_per_second <= 0:
            raise ValueError(
                f"max_calls_per_second must be positive, got {max_calls_per_second!r}"
            )
        self._jev_client = jev_client
        self._min_interval_seconds = 1.0 / max_calls_per_second
        self._sleep = sleep
        self._time_source = time_source
        self._last_dispatch_time: float | None = None

    def system_one(self, state: Any, questions: Mapping[str, Choice]) -> Any:
        now = self._time_source()
        if self._last_dispatch_time is not None:
            wait_seconds = self._min_interval_seconds - (now - self._last_dispatch_time)
            if wait_seconds > 0:
                self._sleep(wait_seconds)
                now = self._time_source()
        self._last_dispatch_time = now
        return self._jev_client.system_one(state, questions)
