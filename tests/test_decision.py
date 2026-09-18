import logging
from typing import cast

import pytest
from pyboy import PyBoy
from typesafe_sdk import TypeSafeClient, TypeSafeError

from jev_plays_pokemon.decision import (
    ACTION_SPACE,
    NAVIGATION_MACRO_ACTION,
    Decision,
    build_jev_client,
    decide_action,
    log_decision,
    make_pyboy_action_executor,
    run_turn,
)
from jev_plays_pokemon.game_state import BattleState, GameState
from jev_plays_pokemon.milestones import track_milestones
from jev_plays_pokemon.navigation import NavigationTarget


def _game_state(**overrides) -> GameState:
    defaults = {
        "party": (),
        "money": 0,
        "inventory": (),
        "badges": (),
        "event_flags": frozenset(),
        "battle": BattleState(
            in_battle=False,
            battle_type="none",
            opponent_species=None,
            opponent_level=None,
        ),
        "dialog_open": False,
        "map_id": 40,
        "map_name": "Oaks Lab",
        "player_x": 4,
        "player_y": 5,
    }
    defaults.update(overrides)
    return GameState(**defaults)


class _FakeChoiceAnswer:
    def __init__(self, choice: str, confidence: float) -> None:
        self.choice = choice
        self.confidence = confidence


class _FakeSystemOneResult:
    def __init__(self, choices: dict) -> None:
        self.choices = choices


class _FakeJevClient:
    """Scripted fake standing in for `typesafe_sdk.TypeSafeClient` - no real
    TypeSafe API call, per this ticket."""

    def __init__(self, choice: str, confidence: float) -> None:
        self._choice = choice
        self._confidence = confidence
        self.calls: list[tuple[dict, dict]] = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        return _FakeSystemOneResult(
            {"action": _FakeChoiceAnswer(self._choice, self._confidence)}
        )


def test_decide_action_returns_the_jev_clients_chosen_action_and_confidence():
    client = _FakeJevClient("a", 0.87)
    state = _game_state()
    progress = track_milestones(state.event_flags, state.badges)

    decision = decide_action(client, state, progress)

    assert decision.action == "a"
    assert decision.confidence == 0.87
    assert decision.milestone is not None
    assert decision.milestone.milestone_id == "got_starter"


def test_decide_action_issues_exactly_one_system_one_call():
    client = _FakeJevClient("a", 0.87)
    state = _game_state()
    progress = track_milestones(state.event_flags, state.badges)

    decide_action(client, state, progress)

    assert len(client.calls) == 1


def test_decide_action_presents_a_single_choice_over_the_full_action_space():
    client = _FakeJevClient("a", 0.87)
    state = _game_state()
    progress = track_milestones(state.event_flags, state.badges)

    decide_action(client, state, progress)

    ((_, questions),) = client.calls
    assert set(questions.keys()) == {"action"}
    assert set(questions["action"].criteria.keys()) == set(ACTION_SPACE)
    assert NAVIGATION_MACRO_ACTION in questions["action"].criteria


def test_decide_action_low_confidence_is_still_returned_with_no_special_handling():
    # Per this ticket's MVP scope: no retry/escalation branch exists for low
    # confidence - decide_action just reports whatever Jev returned.
    client = _FakeJevClient("b", 0.12)
    state = _game_state()
    progress = track_milestones(state.event_flags, state.badges)

    decision = decide_action(client, state, progress)

    assert decision.action == "b"
    assert decision.confidence == 0.12


def test_decide_action_state_payload_reflects_the_current_objective():
    client = _FakeJevClient("a", 0.9)
    state = _game_state(event_flags=frozenset({34}))  # got_starter complete
    progress = track_milestones(state.event_flags, state.badges)
    assert progress.current is not None
    assert progress.current.milestone_id == "got_oaks_parcel"

    decide_action(client, state, progress)

    ((state_payload, _),) = client.calls
    assert state_payload["current_objective"] == progress.current.description


def test_decide_action_milestone_is_none_once_every_milestone_is_complete():
    all_badges = (
        "BOULDERBADGE",
        "CASCADEBADGE",
        "THUNDERBADGE",
        "RAINBOWBADGE",
        "SOULBADGE",
        "MARSHBADGE",
        "VOLCANOBADGE",
        "EARTHBADGE",
    )
    all_flags = frozenset({34, 57, 37, 296, 1372, 2305})
    client = _FakeJevClient("a", 0.9)
    state = _game_state(event_flags=all_flags, badges=all_badges)
    progress = track_milestones(state.event_flags, state.badges)

    decision = decide_action(client, state, progress)

    assert decision.milestone is None


def test_run_turn_reads_state_asks_jev_executes_and_logs_in_order():
    events: list[str] = []
    state = _game_state()

    def state_source() -> GameState:
        events.append("state")
        return state

    class _OrderedFakeJevClient(_FakeJevClient):
        def system_one(self, state, questions):
            events.append("ask")
            return super().system_one(state, questions)

    client = _OrderedFakeJevClient("a", 0.87)
    executed: list[str] = []

    def execute_action(action: str) -> None:
        events.append("execute")
        executed.append(action)

    logged: list[Decision] = []

    def fake_log(decision: Decision) -> None:
        events.append("log")
        logged.append(decision)

    decision = run_turn(state_source, client, execute_action, on_decision=fake_log)

    assert events == ["state", "ask", "execute", "log"]
    assert executed == ["a"]
    assert logged == [decision]
    assert decision.action == "a"


def test_run_turn_executes_the_chosen_action_immediately_even_at_low_confidence():
    # No confidence-based retry/escalation branch exists in MVP - the chosen
    # action always runs exactly once regardless of confidence.
    state = _game_state()
    client = _FakeJevClient("down", 0.05)
    executed: list[str] = []

    run_turn(lambda: state, client, executed.append, on_decision=lambda decision: None)

    assert executed == ["down"]


def test_run_turn_logs_unconditionally_via_the_default_logger(caplog):
    state = _game_state()
    client = _FakeJevClient("down", 0.05)

    with caplog.at_level(logging.INFO, logger="jev_plays_pokemon.decision"):
        run_turn(lambda: state, client, lambda action: None)

    assert len(caplog.records) == 1
    message = caplog.records[0].getMessage()
    assert "action=down" in message
    assert "confidence=0.050" in message
    assert "milestone_id=got_starter" in message


def test_log_decision_logs_none_milestone_when_progress_is_complete(caplog):
    decision = Decision(action="a", confidence=0.5, milestone=None)

    with caplog.at_level(logging.INFO, logger="jev_plays_pokemon.decision"):
        log_decision(decision)

    message = caplog.records[0].getMessage()
    assert "milestone_id=None" in message


class _FakePyBoy:
    """Stands in for `pyboy.PyBoy` - never touched by these tests since
    `make_pyboy_action_executor`'s collaborators are injected directly as
    fakes; no real PyBoy or ROM involved."""


def test_pyboy_action_executor_dispatches_a_raw_button_to_press_button():
    calls: list[tuple[object, str]] = []
    pyboy = cast(PyBoy, _FakePyBoy())
    execute_action = make_pyboy_action_executor(
        pyboy, press_button=lambda pyboy, button: calls.append((pyboy, button))
    )

    execute_action("a")

    assert calls == [(pyboy, "a")]


def test_pyboy_action_executor_dispatches_the_macro_using_the_current_milestones_target():
    state = _game_state(event_flags=frozenset(), badges=())
    target = NavigationTarget(map_id=40, x=4, y=5)
    macro_calls: list[tuple[object, object]] = []

    def run_macro(pyboy: object, t: NavigationTarget) -> bool:
        macro_calls.append((pyboy, t))
        return True

    pyboy = cast(PyBoy, _FakePyBoy())
    execute_action = make_pyboy_action_executor(
        pyboy,
        extract_state=lambda pyboy: state,
        resolve_target=lambda milestone: target,
        run_macro=run_macro,
    )

    execute_action(NAVIGATION_MACRO_ACTION)

    assert macro_calls == [(pyboy, target)]


def test_pyboy_action_executor_macro_is_a_noop_without_a_resolvable_target():
    state = _game_state()
    macro_calls: list[tuple] = []

    def run_macro(pyboy: object, t: NavigationTarget) -> bool:
        macro_calls.append((pyboy, t))
        return True

    execute_action = make_pyboy_action_executor(
        cast(PyBoy, _FakePyBoy()),
        extract_state=lambda pyboy: state,
        resolve_target=lambda milestone: None,
        run_macro=run_macro,
    )

    execute_action(NAVIGATION_MACRO_ACTION)

    assert macro_calls == []


def test_build_jev_client_reads_the_api_key_from_the_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")

    client = build_jev_client()

    assert isinstance(client, TypeSafeClient)


def test_build_jev_client_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    with pytest.raises(TypeSafeError):
        build_jev_client()
