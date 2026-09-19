"""Crash/hang watchdog with heartbeat (#58).

Standalone and dependency-free: this module imports nothing from the rest
of the package (only the standard library), and runs the live loop as a
*subprocess* (``python -m jev_plays_pokemon.cli run ...``, ADR 0001) rather
than importing and calling it in-process - so the watchdog's own reliability
never couples to whichever project dependency (PyBoy, the TypeSafe SDK, the
vision backend, ...) might be the thing that's broken. ``cli.py``'s
``watchdog`` subcommand (the only caller of this module's functions from
inside the CLI) keeps that same isolation one level up: it imports this
module eagerly (safe - stdlib only) but defers importing ``main``/
``benchmark`` (which pull in PyBoy et al.) to their own subcommands.

Two-tier handling composes with the rest of #45's reliability work without
this module depending on either directly:

- A **crash** (main() raises) is caught in-process by main()'s own
  try/except around ``run_loop`` (see ``main.py``): it logs the exception
  and exits nonzero. This module's watchdog loop restarts the subprocess
  whenever it sees that nonzero exit.
- A **hang** (the process is alive but wedged - nothing raised, so nothing
  in-process could catch it) is caught here instead, by polling the
  heartbeat file's mtime: ``main.py`` touches it (``touch_heartbeat``) on
  every retry/backoff tick (via ``resilience.py``'s ``on_attempt`` hook,
  #56) and on every full turn completion, so a healthy run - even one
  currently backing off through an API outage - keeps it fresh.
  ``DEFAULT_HEARTBEAT_TIMEOUT_SECONDS`` (120s) is set comfortably above
  ``resilience.DEFAULT_MAX_INTER_TURN_BACKOFF_SECONDS`` (60s) so a
  legitimate long-outage backoff is never mistaken for a hang.

Restarting resumes from the latest persisted snapshot (#55) rather than
rebooting fresh, since that's simply what ``main()`` itself already does on
every start (``emulator.boot_or_resume``) - nothing extra for this module
to wire.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

logger = logging.getLogger(__name__)

DEFAULT_HEARTBEAT_TIMEOUT_SECONDS = 120.0
DEFAULT_POLL_INTERVAL_SECONDS = 5.0
# Alongside the ROM/snapshot at the working-tree root, gitignored: derived,
# regenerable run state, not something to check in.
DEFAULT_HEARTBEAT_PATH = Path(__file__).resolve().parents[2] / "heartbeat.txt"


def touch_heartbeat(path: str | Path = DEFAULT_HEARTBEAT_PATH) -> None:
    """Update `path`'s mtime to now, creating it if it doesn't exist yet.

    The one thing `main.py`/`resilience.py` call into here - wiring the
    heartbeat into the live loop never needs to import the rest of this
    (dependency-free) watchdog module's subprocess/polling machinery.
    """
    Path(path).touch()


def _heartbeat_age_seconds(path: Path, *, now: float, since: float) -> float:
    """Seconds since `path` was last touched, floored at `since` (the
    current subprocess's spawn time).

    Without the floor, a heartbeat file that's missing entirely, or still
    holds a stale mtime left over from a *previous* run, would look
    infinitely/arbitrarily stale the instant a fresh subprocess starts -
    before it has had any chance to write to it. Flooring at `since` gives
    every fresh spawn a full `heartbeat_timeout_seconds` grace period
    before its (lack of) heartbeat can trigger a restart.
    """
    try:
        mtime = path.stat().st_mtime
    except FileNotFoundError:
        mtime = since
    return now - max(mtime, since)


class ProcessLike(Protocol):
    """The slice of `subprocess.Popen`'s API this module needs - lets tests
    inject a fake process instead of a real subprocess."""

    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


def run_watchdog(
    spawn: Callable[[], ProcessLike],
    *,
    heartbeat_path: Path,
    heartbeat_timeout_seconds: float = DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    time_source: Callable[[], float] = time.time,
    sleep: Callable[[float], None] = time.sleep,
    max_restarts: int | None = None,
    max_polls: int | None = None,
) -> None:
    """Run `spawn()`'s process, restarting it on a nonzero exit or a stale
    heartbeat - forever, by default (production); `max_restarts`/
    `max_polls` bound the loop for tests rather than needing a real hang or
    a real subprocess.

    `spawn` is a zero-argument seam returning a fresh process handle each
    call - production passes a real `subprocess.Popen`; tests inject a fake
    exposing just `poll`/`terminate`/`kill`/`wait`. This function never
    builds a command line or knows what it's running.
    """
    restarts = 0
    polls = 0
    while True:
        logger.info("watchdog: starting subprocess (restart #%d)", restarts)
        spawn_time = time_source()
        process = spawn()
        while True:
            exit_code = process.poll()
            if exit_code is not None:
                logger.warning(
                    "watchdog: subprocess exited with code %s; restarting", exit_code
                )
                break

            age = _heartbeat_age_seconds(
                heartbeat_path, now=time_source(), since=spawn_time
            )
            if age > heartbeat_timeout_seconds:
                logger.warning(
                    "watchdog: heartbeat stale (%.1fs > %.1fs); "
                    "terminating and restarting",
                    age,
                    heartbeat_timeout_seconds,
                )
                process.terminate()
                try:
                    process.wait(timeout=poll_interval_seconds)
                except subprocess.TimeoutExpired:
                    process.kill()
                break

            polls += 1
            if max_polls is not None and polls >= max_polls:
                logger.info("watchdog: reached max_polls=%d; stopping", max_polls)
                return
            sleep(poll_interval_seconds)

        restarts += 1
        if max_restarts is not None and restarts >= max_restarts:
            logger.info("watchdog: reached max_restarts=%d; stopping", max_restarts)
            return


def spawn_run_subprocess(
    *,
    rom_path: str | None,
    stream_port: int,
    max_calls_per_second: float | None,
    openai_api_key: str | None = None,
    openai_base_url: str | None = None,
    openai_vision_model: str | None = None,
    dialog_decode_backend: str | None = None,
    typesafe_api_key: str | None = None,
) -> subprocess.Popen:
    """Spawn ``jev-plays-pokemon run`` as a subprocess, forwarding every
    setting the watchdog itself resolved (ADR 0001) - not just `rom_path` -
    so a value passed as a CLI flag to the watchdog invocation still reaches
    the child it spawns. A value that only came from an environment
    variable/`.env` doesn't need forwarding: the child inherits the same
    process environment and working directory, and resolves it again on its
    own.

    Global settings are re-emitted before the ``run`` subcommand name (see
    `cli.py`), only for the ones that are set - an unset one is simply
    omitted rather than forwarded as an explicit empty value.
    """
    args = [sys.executable, "-m", "jev_plays_pokemon.cli"]
    global_overrides = {
        "--openai-api-key": openai_api_key,
        "--openai-base-url": openai_base_url,
        "--openai-vision-model": openai_vision_model,
        "--dialog-decode-backend": dialog_decode_backend,
        "--typesafe-api-key": typesafe_api_key,
    }
    for flag, value in global_overrides.items():
        if value is not None:
            args += [flag, value]

    args.append("run")
    if rom_path is not None:
        args += ["--rom-path", rom_path]
    args += ["--stream-port", str(stream_port)]
    if max_calls_per_second is not None:
        args += ["--max-calls-per-second", str(max_calls_per_second)]
    return subprocess.Popen(args)
