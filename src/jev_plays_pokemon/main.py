"""The live tactical action-selection loop: this project's entry point (#23).

Wires every prior module into the real per-turn cycle that a placeholder
``main`` stood in for until now: boot Pokemon Red on PyBoy from the repo's own
``pokemon_red.gb`` into a controllable state, then each turn read the game
state (RAM extraction + the vision-fallback dialog text when a dialog is
open), ask Jev its single Choice, execute the chosen action, log the decision
unconditionally, and update the stream-facing surface - one full
state-extraction (#17 + #19) -> Jev Choice (#21) -> action execution (#20) ->
unconditional logging (#21) -> stream-surface update (#22) cycle per turn.

The composition lives in three pieces so each is testable at the seam the
parent spec calls for, without any real TypeSafe/vision call in the unit
tests:

- ``build_dialog_text_source`` turns the configured dialog decoder (#19) into
  the loop's ``dialog_text_source`` seam: it decodes only when a dialog is
  open, and never raises, so a missing or failing vision backend degrades to
  "no dialog text" instead of stalling the live loop.
- ``run_loop`` is the pure per-turn cycle over injected seams; the real
  end-to-end test (``tests/test_main.py``) drives it against a booted ROM with
  a scripted fake Jev client.
- ``main`` is the production wiring: it boots the ROM (resuming from the
  latest snapshot if one exists - #55), constructs the real Jev and
  dialog-decode clients, starts the read-only stream surface and its
  frame-capture timer (#51, reading the same live screen the vision fallback
  captures), wraps its state/action seams for stuck detection (#57), and
  runs ``run_loop`` until interrupted - catching (and logging) a crash
  in-process and exiting nonzero, for ``watchdog.py`` (#58) to restart.

Like the rest of the package, nothing here reaches TypeSafe for vision or the
reverse: Jev's Choice goes through the ``typesafe_sdk`` client (#21); dialog
text is decoded through the separately-configured OpenAI-compatible backend
(#19), never through Jev.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Callable

from pyboy import PyBoy

from jev_plays_pokemon import emulator
from jev_plays_pokemon.decision import (
    Decision,
    DecisionLogger,
    DialogTextSource,
    JevClient,
    build_jev_client,
    make_pyboy_action_executor,
    run_turn,
)
from jev_plays_pokemon.dialog_decode import DialogDecoder, load_dialog_decoder_or_none
from jev_plays_pokemon.dialog_vision import capture_screen
from jev_plays_pokemon.frame_capture import FrameCapture, start_frame_capture
from jev_plays_pokemon.game_state import GameState, extract_game_state
from jev_plays_pokemon.rate_limit import RateLimitedJevClient
from jev_plays_pokemon.resilience import ResilientJevClient
from jev_plays_pokemon.stream_surface import (
    StreamSurface,
    start_stream_surface_server,
    stream_logger,
)
from jev_plays_pokemon.stuck_detection import wrap_for_stuck_detection
from jev_plays_pokemon.watchdog import touch_heartbeat

logger = logging.getLogger(__name__)

# The 10-minute time-based save safety net (#55), independent of milestone
# progress - a run that stalls between milestones for a long stretch still
# gets a recent snapshot.
_SAVE_INTERVAL_SECONDS = 600.0

# Distinguishes "no decision has completed yet" from a legitimate observed
# `MilestoneProgress.current` of `None` (every milestone done) - only the
# latter is a real transition worth an extra save.
_MILESTONE_UNOBSERVED = object()


def build_dialog_text_source(
    pyboy: PyBoy,
    decoder: DialogDecoder | None,
    *,
    render: Callable[[PyBoy], None] = emulator.render_current_frame,
) -> DialogTextSource:
    """Build the loop's ``dialog_text_source`` seam from a dialog decoder.

    The decoder is triggered exactly when ``GameState.dialog_open`` is true -
    the automatic, build-time trigger #19 fixes (never a per-turn Jev call) -
    and is skipped entirely when no decoder is configured (``decoder is
    None``), so the loop runs unchanged whether or not the vision backend is
    wired up.

    ``render`` is called before capturing the screen as an explicit guarantee
    that a frame has been rendered - PyBoy only refreshes its screen buffer on
    a ``render=True`` tick - regardless of what ticks came before (see
    ``emulator.render_current_frame``). It is a seam so the ordering can be
    asserted without a real PyBoy.

    Never raises: a decode failure returns ``None`` (with a warning) rather
    than propagating into ``run_turn``, so one bad frame from a flaky or
    misconfigured endpoint can't stall the live tactical loop - the chosen
    action still executes and the decision still logs.
    """

    def dialog_text_source(state: GameState) -> str | None:
        if not state.dialog_open or decoder is None:
            return None
        try:
            render(pyboy)
            return decoder(capture_screen(pyboy))
        except Exception:
            logger.warning(
                "dialog decode failed; continuing with no dialog text", exc_info=True
            )
            return None

    return dialog_text_source


def run_loop(
    state_source: Callable[[], GameState],
    jev_client: JevClient,
    execute_action: Callable[[str], None],
    *,
    on_decision: DecisionLogger,
    dialog_text_source: DialogTextSource | None = None,
    max_turns: int | None = None,
    save_snapshot: Callable[[], None] | None = None,
    save_interval_seconds: float = _SAVE_INTERVAL_SECONDS,
    time_source: Callable[[], float] = time.monotonic,
) -> None:
    """Run the per-turn tactical cycle, once per turn, indefinitely or up to ``max_turns``.

    Purely a repeated :func:`decision.run_turn` over the same injected seams:
    the read -> ask -> execute -> log -> surface-update ordering and the
    "act regardless of confidence" behaviour both live in ``run_turn``, so a
    low-confidence decision never blocks a subsequent turn (the loop has no
    retry/escalation branch to stall on, per #3's MVP decision). ``max_turns``
    bounds the loop for the end-to-end test; production passes ``None`` and
    runs until interrupted.

    ``save_snapshot`` (#55), when given, is called after a turn whose
    ``Decision.milestone_progress.current`` differs from the previous turn's
    (a milestone-completion transition - the very first turn never counts,
    having no previous turn to transition from) and independently on a fixed
    ``save_interval_seconds`` time-based safety net; either trigger alone is
    enough. ``None`` (the default) disables saving entirely - the seam is
    inert unless a caller opts in. ``time_source`` is a seam over
    ``time.monotonic`` purely so the safety net is testable without a real
    wait.

    ``run_turn`` returns ``None`` for a turn Jev's retries exhausted (#56,
    ``resilience.TurnSkipped``) - no action was taken and there's no
    ``Decision`` to check for a milestone transition, but the time-based
    safety net is still evaluated for that turn, independent of Jev.
    """
    turns = 0
    last_milestone_id: object = _MILESTONE_UNOBSERVED
    last_save_time = time_source()
    while max_turns is None or turns < max_turns:
        decision = run_turn(
            state_source,
            jev_client,
            execute_action,
            on_decision=on_decision,
            dialog_text_source=dialog_text_source,
        )
        turns += 1

        if save_snapshot is not None:
            milestone_completed = False
            if decision is not None:
                current = decision.milestone_progress.current
                current_id = current.milestone_id if current else None
                milestone_completed = (
                    last_milestone_id is not _MILESTONE_UNOBSERVED
                    and current_id != last_milestone_id
                )
                last_milestone_id = current_id

            now = time_source()
            safety_net_due = now - last_save_time >= save_interval_seconds
            if milestone_completed or safety_net_due:
                save_snapshot()
                last_save_time = now


def main(
    rom_path: str | None = None,
    *,
    stream_port: int = 0,
    jev_client: JevClient | None = None,
    max_calls_per_second: float | None = None,
) -> None:
    """Boot the ROM and run the live tactical loop until interrupted.

    ``jev_client``/``rom_path``/``stream_port`` are overridable for a manual or
    scripted live run; with no arguments this reads ``TYPESAFE_API_KEY`` from
    the environment (via :func:`decision.build_jev_client`), boots the repo's
    own ROM through :mod:`emulator`, and binds the read-only stream surface to
    an ephemeral loopback port (see
    :func:`stream_surface.start_stream_surface_server` for reading it back).

    ``max_calls_per_second``, when given, wraps the resolved client in
    :class:`rate_limit.RateLimitedJevClient` - debug mode: slow enough for a
    human (or a log tail) to watch each decision, without changing anything
    about production behaviour when left unset. Pass a low rate (e.g. ``0.5``
    - one call every two seconds) to investigate a stuck-loop symptom like
    #59's without burning hundreds of real API calls doing it.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if jev_client is None:
        jev_client = build_jev_client()
    if max_calls_per_second is not None:
        jev_client = RateLimitedJevClient(
            jev_client, max_calls_per_second=max_calls_per_second
        )
    # Wraps whichever client was resolved above (real or caller-supplied) in
    # retry/backoff handling (#56) - transparent to run_loop/run_turn, which
    # only ever see the JevClient seam. `on_attempt` touches the heartbeat
    # file (#58) on every attempt/backoff tick, so a run currently backing
    # off through an API outage still looks alive to the watchdog.
    jev_client = ResilientJevClient(
        jev_client, on_attempt=lambda event: touch_heartbeat()
    )

    pyboy = emulator.boot_or_resume(rom_path or emulator.DEFAULT_ROM_PATH)
    surface = StreamSurface()
    # `capture_screen` (not `pyboy.screen.image`, which is `None` under this
    # project's `window="null"` backend - see its own docstring) is the same
    # PIL-Image-from-`ndarray` read `dialog_vision.py` already uses, so
    # `/video.mjpg` JPEG-encodes the same pixels the vision fallback would
    # decode. `navigation.execute_button`'s per-turn `render=True` tick (#47)
    # keeps the buffer continuously fresh once play starts, independent of
    # this capture timer's own ~10fps pull cadence (#48).
    frame_capture = FrameCapture(lambda: capture_screen(pyboy))
    frame_capture_timer = start_frame_capture(frame_capture)
    server = start_stream_surface_server(
        surface, port=stream_port, frame_capture=frame_capture
    )
    logger.info(
        "stream surface listening on http://127.0.0.1:%d/", server.server_address[1]
    )

    decoder = load_dialog_decoder_or_none()
    dialog_text_source = build_dialog_text_source(pyboy, decoder)
    execute_action = make_pyboy_action_executor(pyboy)
    save_snapshot = emulator.make_pyboy_snapshot_saver(pyboy)

    def load_snapshot_if_present() -> None:
        # A stuck escalation before any snapshot has ever been saved has
        # nothing to reload yet - log and continue rather than crash.
        if emulator.DEFAULT_SNAPSHOT_PATH.exists():
            emulator.load_snapshot(pyboy)
        else:
            logger.warning(
                "stuck escalated to a snapshot reload, but no snapshot exists "
                "yet; continuing without reloading"
            )

    # A thin wrapper around run_loop's own seams (#57), not a change inside
    # run_turn: pairs each turn's freshly-read position with the action
    # run_turn goes on to execute, and injects a nudge/reload as a side
    # effect when the rolling history says the run is stuck.
    state_source, execute_action = wrap_for_stuck_detection(
        lambda: extract_game_state(pyboy),
        execute_action,
        load_snapshot=load_snapshot_if_present,
    )

    log_and_update_surface = stream_logger(surface)

    def on_decision(decision: Decision) -> None:
        # Full-turn completion is the heartbeat's other write point (#58),
        # alongside resilience.py's per-attempt one above - a healthy run
        # with no retries at all still keeps the file fresh.
        touch_heartbeat()
        log_and_update_surface(decision)

    crashed = False
    try:
        run_loop(
            state_source,
            jev_client,
            execute_action,
            on_decision=on_decision,
            dialog_text_source=dialog_text_source,
            save_snapshot=save_snapshot,
        )
    except KeyboardInterrupt:
        logger.info("interrupted; shutting down")
    except Exception:
        # Caught in-process (#58) rather than left to propagate as a bare
        # traceback: logged here, then a deliberate nonzero exit below (once
        # cleanup finishes) is what tells watchdog.py's subprocess-restart
        # loop this run crashed.
        logger.exception("main() crashed")
        crashed = True
    finally:
        server.shutdown()
        server.server_close()
        frame_capture_timer.stop()
        pyboy.stop(save=False)

    if crashed:
        sys.exit(1)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "rom_path",
        nargs="?",
        default=None,
        help="passed through to emulator.boot_or_resume",
    )
    parser.add_argument(
        "--max-calls-per-second",
        type=float,
        default=None,
        help=(
            "Debug mode: cap TypeSafe API calls/sec so a human can watch "
            "decisions without burning many real calls (e.g. 0.5 = one call "
            "every 2s). Unset = no limit (production)."
        ),
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    _args = _parse_args()
    main(_args.rom_path, max_calls_per_second=_args.max_calls_per_second)
