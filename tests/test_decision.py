import dataclasses
import logging
from typing import cast

import pytest
from pyboy import PyBoy
from typesafe_sdk import TypeSafeClient, TypeSafeError

from jev_plays_pokemon.decision import (
    ACTION_SPACE,
    NAVIGATION_MACRO_ACTION,
    RUN_ACTION,
    Decision,
    build_jev_client,
    decide_action,
    log_decision,
    make_pyboy_action_executor,
    run_turn,
)
from jev_plays_pokemon.game_state import (
    BattleState,
    GameState,
    InventoryItem,
    PartyPokemon,
)
from jev_plays_pokemon.milestones import track_milestones
from jev_plays_pokemon.navigation import NavigationTarget
from jev_plays_pokemon.resilience import TurnSkipped


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


# The game-state data that completes every milestone in milestones.py: all
# 8 badges plus the 6 event flags it tracks.
_ALL_BADGES = (
    "BOULDERBADGE",
    "CASCADEBADGE",
    "THUNDERBADGE",
    "RAINBOWBADGE",
    "SOULBADGE",
    "MARSHBADGE",
    "VOLCANOBADGE",
    "EARTHBADGE",
)
_ALL_EVENT_FLAGS = frozenset({34, 57, 37, 296, 1372, 2305})


def test_decide_action_returns_the_jev_clients_chosen_action_and_confidence():
    client = _FakeJevClient("a", 0.87)
    state = _game_state()
    progress = track_milestones(state.event_flags, state.badges)

    decision = decide_action(client, state, progress)

    assert decision.action == "a"
    assert decision.confidence == 0.87
    assert decision.milestone_progress.current is not None
    assert decision.milestone_progress.current.milestone_id == "got_starter"


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


def test_decide_action_state_payload_is_the_documented_jev_input_shape():
    # The full payload keys are Jev's input contract: this pins the whole
    # shape, including what must NOT leak (event flags, map_id) and what
    # #53 added (party moves/pp, money, inventory).
    state = _game_state(
        party=(
            PartyPokemon(
                species="PIKACHU",
                level=5,
                hp=21,
                max_hp=21,
                status="OK",
                moves=(169, 45, 163, 0),
                pp=(30, 30, 15, 0),
            ),
        ),
        money=3000,
        inventory=(
            InventoryItem(item="POTION", quantity=3),
            InventoryItem(item="POKé BALL", quantity=5),
        ),
        badges=("BOULDERBADGE",),
        event_flags=frozenset({34, 57, 37}),
        battle=BattleState(
            in_battle=True,
            battle_type="wild",
            opponent_species="RATTATA",
            opponent_level=3,
        ),
        dialog_open=True,
        map_id=45,
        map_name="Viridian Gym",
        player_x=4,
        player_y=6,
    )
    client = _FakeJevClient("a", 0.9)
    progress = track_milestones(state.event_flags, state.badges)

    decide_action(client, state, progress)

    ((state_payload, _),) = client.calls
    assert state_payload == {
        "map_name": "Viridian Gym",
        "player_x": 4,
        "player_y": 6,
        "party": [
            {
                "species": "PIKACHU",
                "level": 5,
                "hp": 21,
                "max_hp": 21,
                "status": "OK",
                "moves": [
                    {"name": "Unknown Move (169)", "pp": 30},
                    {"name": "GROWL", "pp": 30},
                    {"name": "SLASH", "pp": 15},
                    {"name": "Unknown Move (0)", "pp": 0},
                ],
            }
        ],
        "money": 3000,
        "inventory": [
            {"item": "POTION", "quantity": 3},
            {"item": "POKé BALL", "quantity": 5},
        ],
        "badges": ["BOULDERBADGE"],
        "battle": {
            "in_battle": True,
            "battle_type": "wild",
            "opponent_species": "RATTATA",
            "opponent_level": 3,
        },
        "dialog_open": True,
        "dialog_text": None,
        "current_objective": "Defeat Misty for the Cascade Badge",
    }


def test_decide_action_state_payload_omits_no_max_pp_or_opponent_hp_field():
    # This ticket's explicit exclusions: no max-PP/PP-Up-bonus field, and no
    # opponent-HP field - JevBattleState stays species+level only.
    state = _game_state(
        party=(
            PartyPokemon(
                species="PIKACHU",
                level=5,
                hp=21,
                max_hp=21,
                status="OK",
                moves=(84, 0, 0, 0),
                pp=(20, 0, 0, 0),
            ),
        ),
        battle=BattleState(
            in_battle=True,
            battle_type="wild",
            opponent_species="RATTATA",
            opponent_level=3,
        ),
    )
    client = _FakeJevClient("a", 0.9)
    progress = track_milestones(state.event_flags, state.badges)

    decide_action(client, state, progress)

    ((state_payload, _),) = client.calls
    assert set(state_payload["party"][0]["moves"][0].keys()) == {"name", "pp"}
    assert set(state_payload["battle"].keys()) == {
        "in_battle",
        "battle_type",
        "opponent_species",
        "opponent_level",
    }


def _mon(**overrides) -> PartyPokemon:
    defaults = {
        "species": "PIKACHU",
        "level": 10,
        "hp": 30,
        "max_hp": 30,
        "status": "OK",
        "moves": (84, 45, 0, 0),  # THUNDERSHOCK, GROWL, empty, empty
        "pp": (15, 30, 0, 0),
    }
    defaults.update(overrides)
    return PartyPokemon(**defaults)


def _battle_criteria(state: GameState) -> dict:
    """This ticket's dynamic in-battle Choice, read the same way `decide_action`
    builds it - via a real `system_one()` call against a fake client, not by
    reaching into `decision.py`'s private helper directly."""
    client = _FakeJevClient("a", 0.9)
    progress = track_milestones(state.event_flags, state.badges)
    decide_action(client, state, progress)
    ((_, questions),) = client.calls
    return questions["action"].criteria


def test_battle_choice_offers_only_moves_with_a_move_and_remaining_pp():
    state = _game_state(
        party=(_mon(moves=(84, 45, 33, 0), pp=(15, 0, 10, 0)),),
        battle=BattleState(
            in_battle=True,
            battle_type="wild",
            opponent_species="RATTATA",
            opponent_level=3,
        ),
    )

    criteria = _battle_criteria(state)

    # Slot 1 (15 PP) and slot 3 (10 PP) are legal; slot 2 (0 PP) and slot 4
    # (no move, id 0) are excluded.
    assert "USE_MOVE_1" in criteria
    assert "USE_MOVE_3" in criteria
    assert "USE_MOVE_2" not in criteria
    assert "USE_MOVE_4" not in criteria


def test_battle_choice_offers_one_option_per_distinct_held_item():
    state = _game_state(
        party=(_mon(),),
        inventory=(
            InventoryItem(item="POTION", quantity=2),
            InventoryItem(item="OAK's PARCEL", quantity=1),
        ),
        battle=BattleState(
            in_battle=True,
            battle_type="trainer",
            opponent_species="RATTATA",
            opponent_level=3,
        ),
    )

    criteria = _battle_criteria(state)

    assert "USE_ITEM_POTION" in criteria
    assert "USE_ITEM_OAKS_PARCEL" in criteria


def test_battle_choice_switch_options_exclude_the_leader_and_fainted_members():
    state = _game_state(
        party=(
            _mon(species="PIKACHU"),
            _mon(species="CHARMANDER", hp=0),  # fainted - excluded
            _mon(species="SQUIRTLE", hp=12),  # living, non-active - offered
        ),
        battle=BattleState(
            in_battle=True,
            battle_type="wild",
            opponent_species="RATTATA",
            opponent_level=3,
        ),
    )

    criteria = _battle_criteria(state)

    assert "SWITCH_TO_1" not in criteria  # the active leader itself
    assert "SWITCH_TO_2" not in criteria  # fainted
    assert "SWITCH_TO_3" in criteria


def test_battle_choice_offers_run_in_wild_battles_only():
    wild = _game_state(
        party=(_mon(),),
        battle=BattleState(
            in_battle=True,
            battle_type="wild",
            opponent_species="RATTATA",
            opponent_level=3,
        ),
    )
    trainer = _game_state(
        party=(_mon(),),
        battle=BattleState(
            in_battle=True,
            battle_type="trainer",
            opponent_species="RATTATA",
            opponent_level=3,
        ),
    )

    assert RUN_ACTION in _battle_criteria(wild)
    assert RUN_ACTION not in _battle_criteria(trainer)


def test_battle_choice_forced_switch_on_faint_offers_only_living_switch_targets():
    # The active leader has fainted (hp == 0) - Gen 1 forces a switch with no
    # other menu options, handled by this same battle macro rather than
    # stuck detection.
    state = _game_state(
        party=(
            _mon(species="PIKACHU", hp=0),
            _mon(species="SQUIRTLE", hp=12),
        ),
        inventory=(InventoryItem(item="POTION", quantity=2),),
        battle=BattleState(
            in_battle=True,
            battle_type="wild",
            opponent_species="RATTATA",
            opponent_level=3,
        ),
    )

    criteria = _battle_criteria(state)

    assert criteria == {"SWITCH_TO_2": "Switch in SQUIRTLE (Lv.10)"}


def test_battle_choice_replaces_rather_than_extends_the_static_action_space():
    state = _game_state(
        party=(_mon(),),
        battle=BattleState(
            in_battle=True,
            battle_type="wild",
            opponent_species="RATTATA",
            opponent_level=3,
        ),
    )

    criteria = _battle_criteria(state)

    assert "up" not in criteria
    assert NAVIGATION_MACRO_ACTION not in criteria


def test_out_of_battle_choice_is_unchanged_by_battle_action_space_support():
    state = _game_state()  # battle.in_battle is False by default

    criteria = _battle_criteria(state)

    assert set(criteria.keys()) == set(ACTION_SPACE)


def test_decide_action_hands_the_dialog_text_to_jev_when_given():
    # #23 wiring: the loop's vision-decoded dialog text (#19) reaches Jev in
    # the state payload - the one vision-sourced field, kept out of
    # GameState - so it can react to story text RAM can't decode.
    client = _FakeJevClient("a", 0.9)
    state = _game_state(dialog_open=True)
    progress = track_milestones(state.event_flags, state.badges)

    decide_action(client, state, progress, dialog_text="A WILD RATTATA APPEARED!")

    ((state_payload, _),) = client.calls
    assert state_payload["dialog_open"] is True
    assert state_payload["dialog_text"] == "A WILD RATTATA APPEARED!"


def test_run_turn_feeds_dialog_text_source_into_the_jev_payload():
    # run_turn's dialog_text_source seam is handed the freshly-read state and
    # its return value flows into Jev's payload - the whole #19 -> #21 wiring
    # exercised without PyBoy, the ROM, or a vision call.
    state = _game_state(dialog_open=True)
    client = _FakeJevClient("a", 0.9)
    seen_states: list[GameState] = []

    def dialog_text_source(read_state: GameState) -> str:
        seen_states.append(read_state)
        return "PROF. OAK: WILL YOU CATCH THIS ONE?"

    run_turn(
        lambda: state,
        client,
        lambda action: None,
        on_decision=lambda decision: None,
        dialog_text_source=dialog_text_source,
    )

    assert seen_states == [state]
    ((state_payload, _),) = client.calls
    assert state_payload["dialog_text"] == "PROF. OAK: WILL YOU CATCH THIS ONE?"


def test_run_turn_without_a_dialog_text_source_leaves_dialog_text_none():
    client = _FakeJevClient("a", 0.9)
    state = _game_state(dialog_open=False)

    run_turn(
        lambda: state,
        client,
        lambda action: None,
        on_decision=lambda decision: None,
    )

    ((state_payload, _),) = client.calls
    assert state_payload["dialog_text"] is None


def test_decide_action_current_objective_is_none_once_every_milestone_is_complete():
    client = _FakeJevClient("a", 0.9)
    state = _game_state(event_flags=_ALL_EVENT_FLAGS, badges=_ALL_BADGES)
    progress = track_milestones(state.event_flags, state.badges)

    decision = decide_action(client, state, progress)

    assert decision.milestone_progress.current is None


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
    assert decision is not None
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


def test_run_turn_returns_none_and_skips_the_action_when_jev_client_raises_turn_skipped(
    caplog,
):
    # #56: a ResilientJevClient (or any JevClient) signals an exhausted-
    # retries turn via TurnSkipped - run_turn must not execute an action or
    # call on_decision for that turn, and must not propagate the exception.
    state = _game_state()
    executed: list[str] = []
    logged: list[Decision] = []

    class _ExhaustedJevClient:
        def system_one(self, state, questions):
            raise TurnSkipped("retries exhausted")

    with caplog.at_level(logging.WARNING, logger="jev_plays_pokemon.decision"):
        decision = run_turn(
            lambda: state,
            _ExhaustedJevClient(),
            executed.append,
            on_decision=logged.append,
        )

    assert decision is None
    assert executed == []
    assert logged == []
    assert any("turn skipped" in r.getMessage().lower() for r in caplog.records)


def test_log_decision_logs_none_milestone_when_progress_is_complete(caplog):
    progress = track_milestones(_ALL_EVENT_FLAGS, _ALL_BADGES)
    decision = Decision(action="a", confidence=0.5, milestone_progress=progress)

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


def test_build_jev_client_prefers_an_explicit_api_key_over_the_environment(
    monkeypatch,
):
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-env")

    client = build_jev_client(api_key="sk-explicit")

    assert isinstance(client, TypeSafeClient)
