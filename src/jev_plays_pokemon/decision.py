"""Tactical Choice decision core: Jev integration + unconditional logging.

Implements #21, part of #14's MVP tactical action-selection loop: the
decision core that asks Jev (TypeSafe's System One model) a single Choice
per turn over the flat action space - raw Game Boy button presses
(`navigation.RAW_BUTTONS`) plus the navigation macro (`navigation.py`, #20)
- and acts on the result immediately, regardless of confidence.

Per the TypeSafe SDK research (`docs/research/typesafe-sdk-integration.md`,
#6) and #3's MVP decision: one `system_one()` call per turn, the action pick
modeled as plain Choice (not SDK function-calling - see that doc's #3) over
the whole action space, executed immediately with no confidence-based
retry/escalation branch. `result.choices[id].confidence` (the SDK's uniform
per-`Answer` field, per that doc's #4) is logged unconditionally alongside
the chosen action and the current-objective milestone split (#18) it was
made against.

`run_turn` is the module's single entry point, composed from three injected
seams so it's testable without PyBoy, the ROM, or a real TypeSafe API call
(per this ticket's acceptance criteria):

- `state_source`: reads one turn's `GameState` (real: `game_state.
  extract_game_state` bound to a running `PyBoy` instance).
- `jev_client`: answers the Choice (real: `build_jev_client()`, which reads
  `TYPESAFE_API_KEY` from the environment on its own - see `JevClient`
  below for the structural shape this module actually needs).
- `execute_action`: carries out the chosen action (real:
  `make_pyboy_action_executor`, which dispatches to `navigation.
  execute_button`/`execute_navigation_macro`).

`decide_action` is kept separate from `run_turn` so a caller that already
has a `GameState` for the turn (e.g. sharing one snapshot across several
systems) can ask Jev without paying for another state read.
"""

import logging
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from pyboy import PyBoy
from pydantic import BaseModel
from typesafe_sdk import Choice, JSONContent, TypeSafeClient

from jev_plays_pokemon.game_state import GameState, extract_game_state
from jev_plays_pokemon.milestones import Milestone, MilestoneProgress, track_milestones
from jev_plays_pokemon.navigation import (
    RAW_BUTTONS,
    NavigationTarget,
    execute_button,
    execute_navigation_macro,
    resolve_navigation_target,
)

logger = logging.getLogger(__name__)

# A flat Choice option, not a function-call: it takes no destination
# argument (`resolve_navigation_target` reads it from the current milestone
# instead) - see `CONTEXT.md`'s "Navigation macro" entry.
NAVIGATION_MACRO_ACTION = "NAVIGATE_TO_OBJECTIVE"

ACTION_SPACE: tuple[str, ...] = (*RAW_BUTTONS, NAVIGATION_MACRO_ACTION)

_ACTION_CRITERIA: dict[str, str] = {
    "up": "Move/face up",
    "down": "Move/face down",
    "left": "Move/face left",
    "right": "Move/face right",
    "a": "Confirm/interact/select the currently highlighted option",
    "b": "Cancel/back out of the current menu",
    "start": "Open the pause/start menu",
    "select": "Press the select button",
    NAVIGATION_MACRO_ACTION: (
        "Automatically walk toward the current story objective instead of "
        "choosing a single directional button yourself"
    ),
}

_ACTION_QUESTION_ID = "action"


class _ChoiceAnswerLike(Protocol):
    """Structural shape `decide_action` needs from a Choice answer.

    Matches `typesafe_sdk.ChoiceAnswer` (see this ticket's research doc's
    confidence section) without depending on that concrete class - purely a
    building block for `JevClient` below, the actual injection point.
    """

    @property
    def choice(self) -> str: ...

    @property
    def confidence(self) -> float: ...


class _SystemOneResultLike(Protocol):
    """Structural shape `decide_action` needs from a `system_one()` result."""

    @property
    def choices(self) -> Mapping[str, _ChoiceAnswerLike]: ...


class JevClient(Protocol):
    """Structural shape of `typesafe_sdk.TypeSafeClient` this module needs.

    Lets tests drive `decide_action`/`run_turn` with a scripted fake client
    returning deterministic Choice responses with fixed confidence (per this
    ticket's test-coverage requirement), rather than a real TypeSafe API
    call.
    """

    def system_one(
        self, state: JSONContent, questions: Mapping[str, Choice]
    ) -> _SystemOneResultLike: ...


def build_jev_client() -> TypeSafeClient:
    """Construct the real Jev client for production use.

    `TypeSafeClient()` reads `TYPESAFE_API_KEY` from the environment on its
    own and raises `typesafe_sdk.TypeSafeError` if it's unset (per the SDK's
    own `Config.resolve` - see this ticket's research doc's §1), so no
    extra env-reading or validation belongs here.
    """
    return TypeSafeClient()


@dataclass(frozen=True)
class Decision:
    action: str
    confidence: float
    # The full current-objective split (#18) this decision was made against,
    # so downstream consumers (e.g. #22's stream-facing surface) can report
    # the state the pick was made on - `None` current once every milestone
    # is complete (see `milestones.py`).
    milestone_progress: MilestoneProgress


ActionExecutor = Callable[[str], None]
DecisionLogger = Callable[[Decision], None]


def log_decision(decision: Decision) -> None:
    """Log one decision unconditionally - no confidence threshold gates this.

    Per this ticket's MVP scope: every decision is logged, regardless of how
    confident Jev was, alongside the current-objective milestone split it
    was made against.
    """
    milestone = decision.milestone_progress.current
    logger.info(
        "action=%s confidence=%.3f milestone_id=%s milestone_description=%s",
        decision.action,
        decision.confidence,
        milestone.milestone_id if milestone else None,
        milestone.description if milestone else None,
    )


class JevPartyMon(BaseModel):
    species: str
    level: int
    hp: int
    max_hp: int
    status: str


class JevBattleState(BaseModel):
    in_battle: bool
    battle_type: str
    opponent_species: str | None
    opponent_level: int | None


class JevStatePayload(BaseModel):
    """The `state` argument of `system_one()` - Jev's input contract.

    Deliberately a subset of `GameState` plus the current objective: money,
    inventory, event flags, `map_id` and party moves/pp are excluded, so
    this model - not a caller's dict literal - is what says what Jev sees.
    """

    map_name: str
    player_x: int
    player_y: int
    party: list[JevPartyMon]
    badges: list[str]
    battle: JevBattleState
    dialog_open: bool
    current_objective: str | None


def _serialize_state(state: GameState, milestone: Milestone | None) -> dict:
    """Build the JSON-able `state` payload handed to `system_one()`."""
    return JevStatePayload(
        map_name=state.map_name,
        player_x=state.player_x,
        player_y=state.player_y,
        party=[
            JevPartyMon(
                species=mon.species,
                level=mon.level,
                hp=mon.hp,
                max_hp=mon.max_hp,
                status=mon.status,
            )
            for mon in state.party
        ],
        badges=list(state.badges),
        battle=JevBattleState(
            in_battle=state.battle.in_battle,
            battle_type=state.battle.battle_type,
            opponent_species=state.battle.opponent_species,
            opponent_level=state.battle.opponent_level,
        ),
        dialog_open=state.dialog_open,
        current_objective=milestone.description if milestone else None,
    ).model_dump(mode="json")


def _build_action_question() -> Choice:
    return Choice(
        instructions="Which single action should be taken next?",
        criteria=_ACTION_CRITERIA,
    )


def decide_action(
    jev_client: JevClient, state: GameState, milestone_progress: MilestoneProgress
) -> Decision:
    """Ask Jev a single Choice over the full action space for `state`.

    Issues exactly one `system_one()` call, per this ticket's MVP scope: the
    action pick is the only judgment this ticket asks for, so there's
    nothing else to batch alongside it yet (see module docstring on
    batching independent judgments into the same call).
    """
    milestone = milestone_progress.current
    state_payload = _serialize_state(state, milestone)
    result = jev_client.system_one(
        state_payload, {_ACTION_QUESTION_ID: _build_action_question()}
    )
    answer = result.choices[_ACTION_QUESTION_ID]
    return Decision(
        action=answer.choice,
        confidence=answer.confidence,
        milestone_progress=milestone_progress,
    )


def run_turn(
    state_source: Callable[[], GameState],
    jev_client: JevClient,
    execute_action: ActionExecutor,
    *,
    on_decision: DecisionLogger = log_decision,
) -> Decision:
    """Run one full turn: read state, ask Jev, act, log - in that order.

    The chosen action is executed immediately with no confidence-based
    retry/escalation path, and every decision is logged unconditionally via
    `on_decision` (both per this ticket's MVP scope).
    """
    state = state_source()
    milestone_progress = track_milestones(state.event_flags, state.badges)
    decision = decide_action(jev_client, state, milestone_progress)
    execute_action(decision.action)
    on_decision(decision)
    return decision


def make_pyboy_action_executor(
    pyboy: PyBoy,
    *,
    extract_state: Callable[[PyBoy], GameState] = extract_game_state,
    resolve_target: Callable[
        [Milestone | None], NavigationTarget | None
    ] = resolve_navigation_target,
    run_macro: Callable[[PyBoy, NavigationTarget], bool] = execute_navigation_macro,
    press_button: Callable[[PyBoy, str], None] = execute_button,
) -> ActionExecutor:
    """Build the real `execute_action` callable for `run_turn`, against `pyboy`.

    The `extract_state`/`resolve_target`/`run_macro`/`press_button` seams
    default to the real `game_state`/`navigation` functions and only need
    overriding in tests (per this ticket's no-real-PyBoy test-coverage
    requirement) - production callers just pass `pyboy`.

    Dispatches a raw button straight to `press_button`. The navigation macro
    action re-reads the current milestone itself (rather than reusing the
    triggering `Decision`'s own snapshot) so it always targets whatever's
    current at execution time, and is a no-op when that milestone has no
    verified tile target yet (see `navigation.py`'s docstring -
    `resolve_navigation_target` returns `None` in that case).
    """

    def execute_action(action: str) -> None:
        if action == NAVIGATION_MACRO_ACTION:
            state = extract_state(pyboy)
            milestone_progress = track_milestones(state.event_flags, state.badges)
            target = resolve_target(milestone_progress.current)
            if target is not None:
                run_macro(pyboy, target)
            return
        press_button(pyboy, action)

    return execute_action
