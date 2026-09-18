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
- ``main`` is the production wiring: it boots the ROM, constructs the real Jev
  and dialog-decode clients, starts the read-only stream surface, and runs
  ``run_loop`` until interrupted.

Like the rest of the package, nothing here reaches TypeSafe for vision or the
reverse: Jev's Choice goes through the ``typesafe_sdk`` client (#21); dialog
text is decoded through the separately-configured OpenAI-compatible backend
(#19), never through Jev.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from pyboy import PyBoy

from jev_plays_pokemon import emulator
from jev_plays_pokemon.decision import (
    DecisionLogger,
    DialogTextSource,
    JevClient,
    build_jev_client,
    make_pyboy_action_executor,
    run_turn,
)
from jev_plays_pokemon.dialog_decode import DialogDecoder, load_dialog_decoder_or_none
from jev_plays_pokemon.dialog_vision import capture_screen
from jev_plays_pokemon.game_state import GameState, extract_game_state
from jev_plays_pokemon.stream_surface import (
    StreamSurface,
    start_stream_surface_server,
    stream_logger,
)

logger = logging.getLogger(__name__)


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

    ``render`` is called before capturing the screen because PyBoy only
    refreshes its screen buffer on a ``render=True`` tick, while the hot loop
    ticks with ``render=False`` - without it ``capture_screen`` would hand the
    decoder a stale/blank frame (see ``emulator.render_current_frame``). It is
    a seam so the ordering can be asserted without a real PyBoy.

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
) -> None:
    """Run the per-turn tactical cycle, once per turn, indefinitely or up to ``max_turns``.

    Purely a repeated :func:`decision.run_turn` over the same injected seams:
    the read -> ask -> execute -> log -> surface-update ordering and the
    "act regardless of confidence" behaviour both live in ``run_turn``, so a
    low-confidence decision never blocks a subsequent turn (the loop has no
    retry/escalation branch to stall on, per #3's MVP decision). ``max_turns``
    bounds the loop for the end-to-end test; production passes ``None`` and
    runs until interrupted.
    """
    turns = 0
    while max_turns is None or turns < max_turns:
        run_turn(
            state_source,
            jev_client,
            execute_action,
            on_decision=on_decision,
            dialog_text_source=dialog_text_source,
        )
        turns += 1


def main(
    rom_path: str | None = None,
    *,
    stream_port: int = 0,
    jev_client: JevClient | None = None,
) -> None:
    """Boot the ROM and run the live tactical loop until interrupted.

    ``jev_client``/``rom_path``/``stream_port`` are overridable for a manual or
    scripted live run; with no arguments this reads ``TYPESAFE_API_KEY`` from
    the environment (via :func:`decision.build_jev_client`), boots the repo's
    own ROM through :mod:`emulator`, and binds the read-only stream surface to
    an ephemeral loopback port (see
    :func:`stream_surface.start_stream_surface_server` for reading it back).
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if jev_client is None:
        jev_client = build_jev_client()

    pyboy = emulator.boot_to_controllable_state(rom_path or emulator.DEFAULT_ROM_PATH)
    surface = StreamSurface()
    server = start_stream_surface_server(surface, port=stream_port)
    logger.info(
        "stream surface listening on http://127.0.0.1:%d/", server.server_address[1]
    )

    decoder = load_dialog_decoder_or_none()
    dialog_text_source = build_dialog_text_source(pyboy, decoder)
    execute_action = make_pyboy_action_executor(pyboy)

    try:
        run_loop(
            lambda: extract_game_state(pyboy),
            jev_client,
            execute_action,
            on_decision=stream_logger(surface),
            dialog_text_source=dialog_text_source,
        )
    except KeyboardInterrupt:
        logger.info("interrupted; shutting down")
    finally:
        server.shutdown()
        server.server_close()
        pyboy.stop(save=False)


if __name__ == "__main__":
    main()
