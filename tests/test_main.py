import dataclasses
import logging
from typing import cast

import pytest
from pyboy import PyBoy

from jev_plays_pokemon.decision import make_pyboy_action_executor
from jev_plays_pokemon.emulator import DEFAULT_ROM_PATH, boot_to_controllable_state
from jev_plays_pokemon.game_state import BattleState, GameState, extract_game_state
from jev_plays_pokemon.main import build_dialog_text_source, main, run_loop
from jev_plays_pokemon.resilience import TurnSkipped
from jev_plays_pokemon.settings import Settings
from jev_plays_pokemon.stream_surface import StreamSurface, stream_logger
from jev_plays_pokemon.stuck_detection import StuckRunAborted
from jev_plays_pokemon.watchdog import EXIT_CODE_STUCK_ABORT


def _game_state(**overrides) -> GameState:
    base = GameState(
        party=(),
        money=0,
        inventory=(),
        badges=(),
        event_flags=frozenset(),
        battle=BattleState(
            in_battle=False,
            battle_type="none",
            opponent_species=None,
            opponent_level=None,
        ),
        dialog_open=False,
        map_id=40,
        map_name="Oaks Lab",
        player_x=4,
        player_y=5,
    )
    return dataclasses.replace(base, **overrides)


class _FakePyBoy:
    """Stands in for `pyboy.PyBoy` where `build_dialog_text_source`'s
    capture_screen and render seams are both overridden by the test."""


class _FakeChoiceAnswer:
    def __init__(self, choice: str, confidence: float) -> None:
        self.choice = choice
        self.confidence = confidence


class _FakeSystemOneResult:
    def __init__(self, choices: dict) -> None:
        self.choices = choices


class _ScriptedJevClient:
    """Scripted fake `typesafe_sdk.TypeSafeClient`: hands out the next
    (choice, confidence) per turn, no real TypeSafe API call."""

    def __init__(self, scripted: list[tuple[str, float]]) -> None:
        self._scripted = scripted
        self.calls = 0
        self.state_payloads: list[dict] = []

    def system_one(self, state, questions):
        self.state_payloads.append(state)
        choice, confidence = self._scripted[self.calls]
        self.calls += 1
        return _FakeSystemOneResult({"action": _FakeChoiceAnswer(choice, confidence)})


# -- build_dialog_text_source: the loop's #19 vision-fallback seam ------------


def test_dialog_text_source_returns_none_and_does_not_decode_when_no_dialog_is_open(
    monkeypatch,
):
    monkeypatch.setattr(
        "jev_plays_pokemon.main.capture_screen", lambda pyboy: pytest.fail("captured")
    )
    source = build_dialog_text_source(cast(PyBoy, _FakePyBoy()), lambda screen: "text")

    assert source(_game_state(dialog_open=False)) is None


def test_dialog_text_source_returns_none_when_no_decoder_is_configured(monkeypatch):
    # Vision not wired up is a normal live-run state: no decode, no crash.
    monkeypatch.setattr(
        "jev_plays_pokemon.main.capture_screen", lambda pyboy: pytest.fail("captured")
    )
    source = build_dialog_text_source(cast(PyBoy, _FakePyBoy()), None)

    assert source(_game_state(dialog_open=True)) is None


def test_dialog_text_source_forces_a_rendered_frame_before_capturing(monkeypatch):
    # The whole loop ticks render=False, so the screen buffer is stale until a
    # render=True tick happens. Decoding must force that tick first, or the
    # decoder gets a blank frame (see emulator.render_current_frame).
    order: list[str] = []

    monkeypatch.setattr(
        "jev_plays_pokemon.main.capture_screen",
        lambda pyboy: (order.append("capture"), "screen")[1],
    )
    source = build_dialog_text_source(
        cast(PyBoy, _FakePyBoy()),
        lambda screen: "text",
        render=lambda pyboy: order.append("render"),
    )

    assert source(_game_state(dialog_open=True)) == "text"
    assert order == ["render", "capture"]


def test_dialog_text_source_decodes_the_screen_when_a_dialog_is_open(monkeypatch):
    def fake_decoder(screen):
        assert screen == "screen-image"
        return "A WILD RATTATA APPEARED!"

    monkeypatch.setattr(
        "jev_plays_pokemon.main.capture_screen", lambda pyboy: "screen-image"
    )
    source = build_dialog_text_source(
        cast(PyBoy, _FakePyBoy()), fake_decoder, render=lambda pyboy: None
    )

    assert source(_game_state(dialog_open=True)) == "A WILD RATTATA APPEARED!"


def test_dialog_text_source_swallows_a_decode_failure_and_continues(
    monkeypatch, caplog
):
    # One bad frame from a flaky/misconfigured endpoint must not stall the
    # live loop: it degrades to "no dialog text" rather than raising.
    def exploding_decoder(screen):
        raise RuntimeError("endpoint down")

    monkeypatch.setattr("jev_plays_pokemon.main.capture_screen", lambda pyboy: "img")
    source = build_dialog_text_source(
        cast(PyBoy, _FakePyBoy()), exploding_decoder, render=lambda pyboy: None
    )

    with caplog.at_level(logging.WARNING, logger="jev_plays_pokemon.main"):
        assert source(_game_state(dialog_open=True)) is None

    assert any("dialog decode failed" in r.getMessage() for r in caplog.records)


class _RecordingLock:
    """Stands in for a real `threading.Lock` - records whether it was held
    (#109) while `render`/`capture_screen` ran, and whether it was still
    held during the decoder call."""

    def __init__(self) -> None:
        self.held = False
        self.held_during_render = False
        self.held_during_decode = False

    def __enter__(self):
        self.held = True
        return self

    def __exit__(self, *exc_info):
        self.held = False
        return False


def test_dialog_text_source_holds_the_lock_around_render_and_capture(monkeypatch):
    lock = _RecordingLock()

    def fake_render(pyboy):
        lock.held_during_render = lock.held

    monkeypatch.setattr("jev_plays_pokemon.main.capture_screen", lambda pyboy: "img")
    source = build_dialog_text_source(
        cast(PyBoy, _FakePyBoy()),
        lambda screen: "text",
        render=fake_render,
        lock=lock,
    )

    assert source(_game_state(dialog_open=True)) == "text"
    assert lock.held_during_render is True
    assert lock.held is False  # released again once the callable returns


def test_dialog_text_source_releases_the_lock_before_calling_the_decoder(monkeypatch):
    # The decoder is a real network round trip to a vision endpoint (#109) -
    # it must never run while the shared pyboy lock is held, or it would
    # stall the free-running emulator clock for the whole call.
    lock = _RecordingLock()

    def fake_decoder(screen):
        lock.held_during_decode = lock.held
        return "text"

    monkeypatch.setattr("jev_plays_pokemon.main.capture_screen", lambda pyboy: "img")
    source = build_dialog_text_source(
        cast(PyBoy, _FakePyBoy()),
        fake_decoder,
        render=lambda pyboy: None,
        lock=lock,
    )

    assert source(_game_state(dialog_open=True)) == "text"
    assert lock.held_during_decode is False


def test_dialog_text_source_works_with_no_lock_given(monkeypatch):
    # Pre-#109 behaviour: no lock argument at all still works, unlocked.
    monkeypatch.setattr("jev_plays_pokemon.main.capture_screen", lambda pyboy: "img")
    source = build_dialog_text_source(
        cast(PyBoy, _FakePyBoy()),
        lambda screen: "text",
        render=lambda pyboy: None,
    )

    assert source(_game_state(dialog_open=True)) == "text"


# -- run_loop: the pure, repeated per-turn cycle ------------------------------


def test_run_loop_runs_exactly_max_turns_turns():
    client = _ScriptedJevClient([("a", 0.9)] * 3)
    executed: list[str] = []
    state_reads = {"count": 0}

    def state_source():
        state_reads["count"] += 1
        return _game_state()

    run_loop(
        state_source,
        client,
        executed.append,
        on_decision=lambda decision: None,
        max_turns=3,
    )

    assert client.calls == 3
    assert executed == ["a", "a", "a"]
    # One fresh state read per turn - no turn reuses a stale snapshot.
    assert state_reads["count"] == 3


def test_run_loop_keeps_going_through_low_confidence_decisions_without_stalling():
    # AC: multiple consecutive turns, including low confidence, with no
    # retry/escalation branch blocking execution - every turn acts once.
    client = _ScriptedJevClient([("down", 0.02), ("right", 0.01), ("up", 0.99)])
    executed: list[str] = []

    run_loop(
        lambda: _game_state(),
        client,
        executed.append,
        on_decision=lambda decision: None,
        max_turns=3,
    )

    assert executed == ["down", "right", "up"]


def test_run_loop_feeds_the_dialog_text_source_into_jev_each_turn():
    client = _ScriptedJevClient([("a", 0.9), ("a", 0.9)])
    calls = {"count": 0}

    def dialog_text_source(state):
        calls["count"] += 1
        return f"text-{calls['count']}"

    run_loop(
        lambda: _game_state(dialog_open=True),
        client,
        lambda action: None,
        on_decision=lambda decision: None,
        dialog_text_source=dialog_text_source,
        max_turns=2,
    )

    assert [p["dialog_text"] for p in client.state_payloads] == ["text-1", "text-2"]


# -- run_loop: skipped turns (#56) ---------------------------------------------


def test_run_loop_continues_past_a_turn_the_jev_client_skips():
    # A JevClient (e.g. a ResilientJevClient with retries exhausted) can
    # raise TurnSkipped instead of answering - run_turn returns None for
    # that turn, and run_loop must carry on to the next turn rather than
    # crashing or stalling.
    class _FlakyThenFineClient:
        def __init__(self) -> None:
            self.calls = 0

        def system_one(self, state, questions):
            self.calls += 1
            if self.calls == 1:
                raise TurnSkipped("retries exhausted")
            return _FakeSystemOneResult({"action": _FakeChoiceAnswer("a", 0.9)})

    client = _FlakyThenFineClient()
    executed: list[str] = []

    run_loop(
        lambda: _game_state(),
        client,
        executed.append,
        on_decision=lambda decision: None,
        max_turns=2,
    )

    assert client.calls == 2
    # Turn 1 was skipped (no action executed); turn 2 executed normally.
    assert executed == ["a"]


def test_run_loop_still_evaluates_the_save_safety_net_on_a_skipped_turn():
    class _AlwaysSkipsClient:
        def system_one(self, state, questions):
            raise TurnSkipped("retries exhausted")

    saves: list[None] = []
    clock = {"now": 0.0}

    def time_source() -> float:
        clock["now"] += 6.0  # "6s elapses" every time run_loop checks the clock
        return clock["now"]

    run_loop(
        lambda: _game_state(),
        _AlwaysSkipsClient(),
        lambda action: None,
        on_decision=lambda decision: None,
        max_turns=1,
        save_snapshot=lambda: saves.append(None),
        save_interval_seconds=5.0,
        time_source=time_source,
    )

    # The turn was skipped, but 6s "elapsed" past the 5s safety-net
    # threshold, so a save still happens independent of Jev.
    assert len(saves) == 1


# -- run_loop: snapshot persistence (#55) -------------------------------------


def test_run_loop_does_not_save_on_the_very_first_turn():
    # The first turn has no previous turn to "transition" from, so it never
    # counts as a milestone completion on its own.
    client = _ScriptedJevClient([("a", 0.9)])
    saves: list[None] = []

    run_loop(
        lambda: _game_state(),
        client,
        lambda action: None,
        on_decision=lambda decision: None,
        max_turns=1,
        save_snapshot=lambda: saves.append(None),
        save_interval_seconds=10_000.0,
        time_source=lambda: 0.0,
    )

    assert saves == []


def test_run_loop_does_not_save_again_while_the_current_milestone_is_unchanged():
    client = _ScriptedJevClient([("a", 0.9), ("a", 0.9)])
    saves: list[None] = []

    run_loop(
        lambda: _game_state(),  # same milestone ("got_starter") every turn
        client,
        lambda action: None,
        on_decision=lambda decision: None,
        max_turns=2,
        save_snapshot=lambda: saves.append(None),
        save_interval_seconds=10_000.0,
        time_source=lambda: 0.0,
    )

    assert saves == []


def test_run_loop_saves_when_the_current_milestone_transitions():
    client = _ScriptedJevClient([("a", 0.9), ("a", 0.9)])
    saves: list[None] = []
    # got_starter (turn 1) -> got_oaks_parcel (turn 2, event flag 34 set).
    states = [_game_state(), _game_state(event_flags=frozenset({34}))]
    calls = {"count": 0}

    def state_source():
        state = states[calls["count"]]
        calls["count"] += 1
        return state

    run_loop(
        state_source,
        client,
        lambda action: None,
        on_decision=lambda decision: None,
        max_turns=2,
        save_snapshot=lambda: saves.append(None),
        save_interval_seconds=10_000.0,
        time_source=lambda: 0.0,
    )

    assert len(saves) == 1


def test_run_loop_saves_on_the_time_based_safety_net_without_a_milestone_change():
    client = _ScriptedJevClient([("a", 0.9)] * 3)
    saves: list[None] = []
    clock = {"now": 0.0}

    def time_source():
        return clock["now"]

    def state_source():
        clock["now"] += 2.0  # 2s "elapses" per turn
        return _game_state()  # same milestone every turn

    run_loop(
        state_source,
        client,
        lambda action: None,
        on_decision=lambda decision: None,
        max_turns=3,
        save_snapshot=lambda: saves.append(None),
        save_interval_seconds=5.0,  # due once elapsed >= 5s (turn 3, at 6s)
        time_source=time_source,
    )

    assert len(saves) == 1


def test_run_loop_never_saves_when_no_save_snapshot_seam_is_given():
    # save_snapshot=None (the default) disables saving entirely - the same
    # shape every pre-#55 run_loop call already used.
    client = _ScriptedJevClient([("a", 0.9), ("a", 0.9)])
    states = [_game_state(), _game_state(event_flags=frozenset({34}))]
    calls = {"count": 0}

    def state_source():
        state = states[calls["count"]]
        calls["count"] += 1
        return state

    # Would raise if run_loop tried to call None() - passing simply confirms
    # no attempt is made when a milestone transition happens with no seam.
    run_loop(
        state_source,
        client,
        lambda action: None,
        on_decision=lambda decision: None,
        max_turns=2,
    )


# -- main(): the exit codes watchdog.py (#58) reads ---------------------------


class _FakePyBoyWithStop:
    """Just enough PyBoy for `main()`'s cleanup path: it only ever gets
    `stop(save=False)` called on it here, since `run_loop` never runs."""

    def __init__(self) -> None:
        self.stop_saved: bool | None = None

    def stop(self, save: bool = True) -> None:
        self.stop_saved = save


class _FakeServer:
    def __init__(self) -> None:
        self.server_address = ("127.0.0.1", 54321)
        self.shutdowns = 0
        self.closes = 0

    def shutdown(self) -> None:
        self.shutdowns += 1

    def server_close(self) -> None:
        self.closes += 1


class _FakeTimer:
    def __init__(self) -> None:
        self.stops = 0

    def stop(self) -> None:
        self.stops += 1


def _main_with_a_failing_loop(monkeypatch, boom: BaseException):
    """Run `main()` with every external seam faked and `run_loop` raising
    `boom`, so what's left under test is `main()`'s own error handling: the
    distinct exit code it maps that to, and that cleanup still happens.
    """
    pyboy = _FakePyBoyWithStop()
    server = _FakeServer()
    timers = [_FakeTimer(), _FakeTimer()]
    monkeypatch.setattr(
        "jev_plays_pokemon.emulator.boot_or_resume", lambda rom_path: pyboy
    )
    monkeypatch.setattr("jev_plays_pokemon.main.capture_screen", lambda pyboy: "img")
    monkeypatch.setattr(
        "jev_plays_pokemon.main.start_frame_capture", lambda *a, **k: timers[0]
    )
    monkeypatch.setattr(
        "jev_plays_pokemon.main.start_emulator_clock", lambda *a, **k: timers[1]
    )
    monkeypatch.setattr(
        "jev_plays_pokemon.main.start_stream_surface_server", lambda *a, **k: server
    )
    monkeypatch.setattr(
        "jev_plays_pokemon.main.load_dialog_decoder_or_none", lambda **k: None
    )
    # Not what these tests are about, and it writes at the working-tree root.
    monkeypatch.setattr("jev_plays_pokemon.main.touch_heartbeat", lambda: None)

    def failing_loop(*args, **kwargs):
        raise boom

    monkeypatch.setattr("jev_plays_pokemon.main.run_loop", failing_loop)

    with pytest.raises(SystemExit) as exit_info:
        main(jev_client=_ScriptedJevClient([]), settings=Settings(), stream_port=0)

    assert pyboy.stop_saved is False, "the run must not leave a save behind"
    assert server.shutdowns == 1 and server.closes == 1
    assert [timer.stops for timer in timers] == [1, 1]
    return exit_info.value.code


def test_main_exits_with_the_stuck_abort_code_when_the_ladder_gives_up(monkeypatch):
    # watchdog.py deliberately does not restart on this code, so the run
    # giving up on an unrecoverable stuck state has to be distinguishable from
    # a crash - which is what keeps #57's give-up from becoming a restart loop
    # back into the same stuck state.
    code = _main_with_a_failing_loop(
        monkeypatch, StuckRunAborted("stuck persisted through 3 snapshot reload(s)")
    )

    assert code == EXIT_CODE_STUCK_ABORT


def test_main_still_exits_one_on_an_ordinary_crash(monkeypatch):
    # The pre-existing crash path (#58's restart signal) must be unchanged by
    # the give-up branch added above it.
    code = _main_with_a_failing_loop(monkeypatch, RuntimeError("PyBoy died"))

    assert code == 1


# -- end-to-end: the real per-turn cycle against a booted ROM ------------------

pytestmark_e2e = pytest.mark.skipif(
    not DEFAULT_ROM_PATH.exists(), reason=f"{DEFAULT_ROM_PATH} not present locally"
)


@pytest.fixture
def live_pyboy():
    pyboy = boot_to_controllable_state(DEFAULT_ROM_PATH)
    yield pyboy
    pyboy.stop(save=False)


@pytestmark_e2e
def test_end_to_end_loop_runs_consecutive_turns_against_the_real_rom(
    live_pyboy, caplog
):
    """This ticket's headline acceptance criterion: state extraction (#17) ->
    Jev Choice (#21) -> action execution (#20) -> unconditional logging (#21)
    -> stream-surface update (#22), driven over several consecutive turns
    against a live PyBoy Pokemon Red instance, with a scripted fake Jev so no
    real TypeSafe call happens (the ROM/PyBoy half is entirely real)."""
    pyboy = live_pyboy

    def position(state):
        return (state.map_id, state.player_x, state.player_y)

    before = extract_game_state(pyboy)

    # Low confidence on two of four turns: none of them may block the loop.
    client = _ScriptedJevClient(
        [("down", 0.02), ("right", 0.4), ("down", 0.03), ("right", 0.8)]
    )
    surface = StreamSurface()
    actions = ("down", "right", "down", "right")

    with caplog.at_level(logging.INFO, logger="jev_plays_pokemon.decision"):
        run_loop(
            lambda: extract_game_state(pyboy),
            client,
            make_pyboy_action_executor(pyboy),
            on_decision=stream_logger(surface),
            max_turns=len(actions),
        )

    # The loop completed every turn (no stall), the ROM actually executed the
    # actions (the player moved), and the surface + log reflect each decision.
    assert client.calls == len(actions)
    snapshot = surface.snapshot()
    assert snapshot["decision_count"] == len(actions)
    assert snapshot["action"] == actions[-1]
    assert snapshot["confidence"] == pytest.approx(0.8)

    after = extract_game_state(pyboy)
    assert position(after) != position(before)

    decision_logs = [
        r.getMessage()
        for r in caplog.records
        if r.name == "jev_plays_pokemon.decision" and "action=" in r.getMessage()
    ]
    assert len(decision_logs) == len(actions)
    # The low-confidence turns were logged unconditionally, same as the rest.
    assert any("confidence=0.020" in line for line in decision_logs)


@pytestmark_e2e
def test_end_to_end_loop_survives_an_unconfigured_dialog_backend(live_pyboy):
    """A run with no vision backend configured (#19) still runs consecutive
    turns: the dialog seam degrades to no text rather than stalling."""
    pyboy = live_pyboy

    client = _ScriptedJevClient([("a", 0.5), ("a", 0.5)])
    surface = StreamSurface()
    # No decoder: build_dialog_text_source(pyboy, None) is the live-run state
    # when the vision backend is unconfigured.
    source = build_dialog_text_source(pyboy, None)

    run_loop(
        lambda: extract_game_state(pyboy),
        client,
        make_pyboy_action_executor(pyboy),
        on_decision=stream_logger(surface),
        dialog_text_source=source,
        max_turns=2,
    )

    assert client.calls == 2
    assert surface.snapshot()["decision_count"] == 2
    assert all(p["dialog_text"] is None for p in client.state_payloads)
