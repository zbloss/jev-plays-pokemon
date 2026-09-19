"""Tactical Choice decision core: Jev integration + unconditional logging.

Implements #21, part of #14's MVP tactical action-selection loop: the
decision core that asks Jev (TypeSafe's System One model) a single Choice
per turn over the flat action space - raw Game Boy button presses
(`navigation.RAW_BUTTONS`) plus the navigation macro (`navigation.py`, #20)
outside battle - and acts on the result immediately, regardless of
confidence. The macro is only offered when `navigation.
resolve_navigation_target` can resolve a destination for the current
milestone (#83); otherwise the Choice is raw buttons only, so Jev can't pick
an action that presses nothing (see `out_of_battle_action_space`). While
`GameState.battle.in_battle` is true, the Choice swaps to the dynamic
battle action space instead (#54, `_battle_action_criteria`):
`USE_MOVE_<slot>` / `USE_ITEM_<item>` / `SWITCH_TO_<slot>` / `RUN`, built
fresh each turn and filtered to legal options only.

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

Plus one optional seam: `dialog_text_source` (#23's wiring of #19) hands the
loop's vision-decoded dialog text into Jev's state payload when a dialog is
open, so `run_turn` stays vision-agnostic and its seam set stays the whole
per-turn cycle.

`decide_action` is kept separate from `run_turn` so a caller that already
has a `GameState` for the turn (e.g. sharing one snapshot across several
systems) can ask Jev without paying for another state read.
"""

import logging
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from pyboy import PyBoy
from pydantic import BaseModel
from typesafe_sdk import Choice, JSONContent, TypeSafeClient

from jev_plays_pokemon import emulator, navigation
from jev_plays_pokemon.game_state import GameState, extract_game_state
from jev_plays_pokemon.lookup.moves import move_name
from jev_plays_pokemon.milestones import Milestone, MilestoneProgress, track_milestones
from jev_plays_pokemon.navigation import (
    RAW_BUTTONS,
    NavigationTarget,
    execute_button,
    execute_navigation_macro,
    main_battle_menu_delta_buttons,
    menu_list_delta_buttons,
    resolve_navigation_target,
)
from jev_plays_pokemon.resilience import TurnSkipped

logger = logging.getLogger(__name__)

# A flat Choice option, not a function-call: it takes no destination
# argument (`resolve_navigation_target` reads it from the current milestone
# instead) - see `CONTEXT.md`'s "Navigation macro" entry.
NAVIGATION_MACRO_ACTION = "NAVIGATE_TO_OBJECTIVE"

ACTION_SPACE: tuple[str, ...] = (*RAW_BUTTONS, NAVIGATION_MACRO_ACTION)


def out_of_battle_action_space(milestone: Milestone | None) -> tuple[str, ...]:
    """The out-of-battle actions actually legal to offer/nudge with this
    turn, for `milestone`: `ACTION_SPACE` is the flat, milestone-independent
    full space (every raw button plus the macro); this is the turn-specific
    subset of it (#83) - the macro drops out whenever it would be a no-op.

    Consults `resolve_navigation_target` - the same resolver `make_pyboy_
    action_executor`'s macro branch calls at execution time - so the offer
    predicate and the execution path can't drift apart into two different
    "does this milestone have a target yet?" checks.
    """
    if resolve_navigation_target(milestone) is not None:
        return ACTION_SPACE
    return RAW_BUTTONS


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

# In-battle Choice option prefixes/keys (#54): built fresh each turn from
# `GameState`, filtered to legal options only - replacing `_ACTION_CRITERIA`
# for the duration of the battle rather than extending it, since raw
# movement buttons and the navigation macro aren't legal battle actions.
_USE_MOVE_PREFIX = "USE_MOVE_"
_USE_ITEM_PREFIX = "USE_ITEM_"
_SWITCH_TO_PREFIX = "SWITCH_TO_"
RUN_ACTION = "RUN"

# This project's active-battle-slot convention: the party leader
# (`state.party[0]`) is always treated as the Pokemon currently in battle.
# Gen 1's real active-party-index isn't separately tracked by `game_state.py`
# (see its RAM-map research doc), so this is what "party-leader move slot"
# and "the active party slot's hp == 0" (this ticket's forced-switch check)
# both mean in practice.
_ACTIVE_PARTY_SLOT = 1

# The main battle menu's 2x2 grid layout (#76) - standard across Gen 1's
# wild/trainer battles, cross-checked against `InsaneJSK/DeepRed`'s
# `battle_controller.py`:
#   FIGHT(0) | PKMN(1)
#   ---------+--------
#   ITEM(2)  | RUN(3)
_MAIN_MENU_FIGHT = 0
_MAIN_MENU_PKMN = 1
_MAIN_MENU_ITEM = 2
_MAIN_MENU_RUN = 3

# Picking a party member from the PKMN list opens a SWITCH/STATS/CANCEL
# confirmation - SWITCH is always its top (id-0) option, per
# `InsaneJSK/DeepRed`'s `battle_controller.py`.
_CONFIRM_SWITCH_INDEX = 0


def _sanitize_action_token(text: str) -> str:
    """Turn a free-form display name into an ASCII, underscore-joined token.

    Apostrophes and periods are dropped rather than replaced (`"OAK's
    PARCEL"` -> `"OAKS_PARCEL"`, not `"OAK_S_PARCEL"`); every other
    non-alphanumeric run (spaces, non-ASCII glyphs like the `é` in `"POKé
    BALL"`, ...) becomes a single underscore.
    """
    stripped = text.replace("'", "").replace(".", "")
    return re.sub(r"[^A-Za-z0-9]+", "_", stripped).strip("_").upper()


def _battle_action_criteria(state: GameState) -> dict[str, str]:
    """Build this turn's in-battle Choice options, filtered to legal ones only.

    - `USE_MOVE_<slot>`: one per party-leader move slot (1-indexed) with a
      non-empty move and current PP > 0.
    - `USE_ITEM_<item>`: one per distinct held item (Red's bag has no
      duplicate stacks, so the sanitized item name alone is a unique key).
    - `SWITCH_TO_<slot>`: one per living (`hp > 0`), non-active party member,
      keyed by its 1-indexed position in `state.party` (not species, to
      avoid collisions between same-species party members).
    - `RUN`: wild battles only, never trainer battles.

    The Gen 1 forced-switch-on-faint state (the active slot's `hp == 0`) is
    handled here, not routed through stuck detection: with the leader
    fainted, using a move/item or running isn't a legal menu option in the
    real game, so only `SWITCH_TO_<slot>` options are offered.
    """
    leader = state.party[0] if state.party else None
    forced_switch = leader is not None and leader.hp <= 0

    criteria: dict[str, str] = {}

    if not forced_switch:
        if leader is not None:
            for slot, (move_id, pp) in enumerate(
                zip(leader.moves, leader.pp, strict=True), start=1
            ):
                if move_id == 0 or pp <= 0:
                    continue
                criteria[f"{_USE_MOVE_PREFIX}{slot}"] = (
                    f"Use {move_name(move_id)} ({pp} PP left)"
                )

        for item in state.inventory:
            key = f"{_USE_ITEM_PREFIX}{_sanitize_action_token(item.item)}"
            criteria[key] = f"Use {item.item} (have {item.quantity})"

    for slot, mon in enumerate(state.party, start=1):
        if slot == _ACTIVE_PARTY_SLOT or mon.hp <= 0:
            continue
        criteria[f"{_SWITCH_TO_PREFIX}{slot}"] = (
            f"Switch in {mon.species} (Lv.{mon.level})"
        )

    if not forced_switch and state.battle.battle_type == "wild":
        criteria[RUN_ACTION] = "Run from this wild battle"

    return criteria


def _is_battle_menu_action(action: str) -> bool:
    return action == RUN_ACTION or action.startswith(
        (_USE_MOVE_PREFIX, _USE_ITEM_PREFIX, _SWITCH_TO_PREFIX)
    )


def _navigate_menu(
    pyboy: PyBoy,
    delta_buttons: Callable[[int], tuple[str, ...]],
    *,
    press_button: Callable[[PyBoy, str], None],
    read_menu_cursor: Callable[[PyBoy], int],
) -> None:
    """Read one battle-menu screen's live cursor, press only what
    `delta_buttons` computes for it, then confirm with A.

    The one navigation primitive every step of `_execute_battle_action`
    (#76) is built from: never a fixed-length sequence, since Gen 1's menu
    cursor persists across turns (see `navigation.menu_list_delta_buttons`/
    `main_battle_menu_delta_buttons`'s docstrings). Reading fresh on every
    call - rather than letting a caller reuse an earlier read - matters
    because a screen's cursor means something different the moment a new
    menu opens on top of it.
    """
    cursor = read_menu_cursor(pyboy)
    for button in delta_buttons(cursor):
        press_button(pyboy, button)
    press_button(pyboy, "a")


def _execute_battle_action(
    pyboy: PyBoy,
    action: str,
    state: GameState,
    *,
    press_button: Callable[[PyBoy, str], None],
    read_menu_cursor: Callable[[PyBoy], int],
) -> bool:
    """Translate one in-battle `Decision.action` (#54) into the Game Boy
    input Gen 1's battle menus require (#76), via `_navigate_menu`.

    Move/switch/item targets are the real, unfiltered slot/bag position
    (`_battle_action_criteria` already keys them that way) - a fainted party
    member or an unusable move still occupies its own real row in the menu,
    so no compaction/re-indexing happens here either (per this ticket's
    acceptance criteria).

    Parses `USE_MOVE_<slot>`/`SWITCH_TO_<slot>`'s numeric suffix defensively:
    an unparseable slot (Jev picking something outside the criteria it was
    given) is a no-op rather than a crash, matching this ticket's whole
    point - a battle turn should never raise over the chosen action. Same for
    `USE_ITEM_<item>` when the token doesn't match anything in `state.
    inventory`.

    Returns whether it actually pressed anything - `False` on any of the
    no-op cases above (`make_pyboy_action_executor` uses this to know when a
    turn needs #81's explicit extra render, since a no-op here never ticks
    `pyboy` itself).
    """

    def navigate(delta_buttons: Callable[[int], tuple[str, ...]]) -> None:
        _navigate_menu(
            pyboy,
            delta_buttons,
            press_button=press_button,
            read_menu_cursor=read_menu_cursor,
        )

    if action == RUN_ACTION:
        navigate(lambda cursor: main_battle_menu_delta_buttons(cursor, _MAIN_MENU_RUN))
        return True

    if action.startswith(_USE_MOVE_PREFIX):
        try:
            slot = int(action[len(_USE_MOVE_PREFIX) :])
        except ValueError:
            return False
        navigate(
            lambda cursor: main_battle_menu_delta_buttons(cursor, _MAIN_MENU_FIGHT)
        )
        # The move list is 1-indexed by cursor value (cursor == slot) -
        # cross-checked against `2389-research/jev-plays-pokemon`'s
        # `battle.py`, validated there against a real battle save state.
        navigate(lambda cursor: menu_list_delta_buttons(cursor, slot))
        return True

    if action.startswith(_SWITCH_TO_PREFIX):
        try:
            slot = int(action[len(_SWITCH_TO_PREFIX) :])
        except ValueError:
            return False
        navigate(lambda cursor: main_battle_menu_delta_buttons(cursor, _MAIN_MENU_PKMN))
        navigate(lambda cursor: menu_list_delta_buttons(cursor, slot - 1))
        # Picking a party member opens a SWITCH/STATS/CANCEL confirmation
        # (SWITCH is always its top, id-0, option) - cross-checked against
        # `InsaneJSK/DeepRed`'s `switch()`. Read live rather than assumed,
        # same as every other menu here (#76): a blind fixed press would be
        # exactly the bug this ticket exists to close if that assumption
        # ever turns out wrong.
        navigate(lambda cursor: menu_list_delta_buttons(cursor, _CONFIRM_SWITCH_INDEX))
        return True

    if action.startswith(_USE_ITEM_PREFIX):
        item_token = action[len(_USE_ITEM_PREFIX) :]
        item_index = next(
            (
                index
                for index, entry in enumerate(state.inventory)
                if _sanitize_action_token(entry.item) == item_token
            ),
            None,
        )
        if item_index is None:
            return False
        navigate(lambda cursor: main_battle_menu_delta_buttons(cursor, _MAIN_MENU_ITEM))
        navigate(lambda cursor: menu_list_delta_buttons(cursor, item_index))
        return True

    return False


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


def build_jev_client(*, api_key: str | None = None) -> TypeSafeClient:
    """Construct the real Jev client for production use.

    `api_key`, when given, overrides `TYPESAFE_API_KEY` (see
    `settings.Settings`, ADR 0001). Left as `None`, `TypeSafeClient` reads
    `TYPESAFE_API_KEY` from the environment on its own and raises
    `typesafe_sdk.TypeSafeError` if it's unset (per the SDK's own
    `Config.resolve` - see this ticket's research doc's §1), so no extra
    env-reading or validation belongs here either way.
    """
    return TypeSafeClient(api_key=api_key)


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
# The loop's vision-fallback seam (#19): reads a turn's `GameState` and returns
# the decoded dialog text to offer Jev, or `None`.
DialogTextSource = Callable[[GameState], "str | None"]


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


class JevMove(BaseModel):
    """A party Pokemon's move slot - name and current PP only.

    Max PP and the PP-Up bonus formula are deliberately excluded: current PP
    alone is sufficient for legal-move filtering elsewhere (#53's scope).
    """

    name: str
    pp: int


class JevPartyMon(BaseModel):
    species: str
    level: int
    hp: int
    max_hp: int
    status: str
    moves: list[JevMove]


class JevBattleState(BaseModel):
    """Opponent state stays species-and-level-only, deliberately: no
    opponent-HP field, and no new vision-fallback seam for it (#53's scope).
    """

    in_battle: bool
    battle_type: str
    opponent_species: str | None
    opponent_level: int | None


class JevInventoryItem(BaseModel):
    item: str
    quantity: int


class JevStatePayload(BaseModel):
    """The `state` argument of `system_one()` - Jev's input contract.

    Deliberately a subset of `GameState` plus the current objective plus the
    vision-decoded dialog text: event flags and `map_id` are excluded, so
    this model - not a caller's dict literal - is what says what Jev sees.

    `dialog_text` is the one vision-sourced field (#19): `GameState` itself
    stays RAM-only (`dialog_open` only says *that* a dialog is up), so the
    loop hands the decoded text in alongside the RAM snapshot rather than
    folding it into `GameState`. `None` whenever no dialog is open, or the
    dialog is open but the vision fallback isn't configured/failed.
    """

    map_name: str
    player_x: int
    player_y: int
    party: list[JevPartyMon]
    money: int
    inventory: list[JevInventoryItem]
    badges: list[str]
    battle: JevBattleState
    dialog_open: bool
    dialog_text: str | None
    current_objective: str | None


def _serialize_state(
    state: GameState, milestone: Milestone | None, dialog_text: str | None
) -> dict:
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
                moves=[
                    JevMove(name=move_name(move_id), pp=pp)
                    for move_id, pp in zip(mon.moves, mon.pp, strict=True)
                ],
            )
            for mon in state.party
        ],
        money=state.money,
        inventory=[
            JevInventoryItem(item=entry.item, quantity=entry.quantity)
            for entry in state.inventory
        ],
        badges=list(state.badges),
        battle=JevBattleState(
            in_battle=state.battle.in_battle,
            battle_type=state.battle.battle_type,
            opponent_species=state.battle.opponent_species,
            opponent_level=state.battle.opponent_level,
        ),
        dialog_open=state.dialog_open,
        dialog_text=dialog_text,
        current_objective=milestone.description if milestone else None,
    ).model_dump(mode="json")


def _build_action_question(state: GameState, milestone: Milestone | None) -> Choice:
    """Build this turn's Choice: the dynamic battle action space (#54) while
    `state.battle.in_battle`, otherwise the raw buttons plus the navigation
    macro - but only when `out_of_battle_action_space` says the macro can
    actually resolve a destination for `milestone` right now (#83).
    """
    if state.battle.in_battle:
        return Choice(
            instructions="Which single battle action should be taken next?",
            criteria=_battle_action_criteria(state),
        )
    action_space = out_of_battle_action_space(milestone)
    criteria = {action: _ACTION_CRITERIA[action] for action in action_space}
    return Choice(
        instructions="Which single action should be taken next?",
        criteria=criteria,
    )


def decide_action(
    jev_client: JevClient,
    state: GameState,
    milestone_progress: MilestoneProgress,
    *,
    dialog_text: str | None = None,
) -> Decision:
    """Ask Jev a single Choice over the full action space for `state`.

    Issues exactly one `system_one()` call, per this ticket's MVP scope: the
    action pick is the only judgment this ticket asks for, so there's
    nothing else to batch alongside it yet (see module docstring on
    batching independent judgments into the same call).

    `dialog_text` is the vision-decoded dialog/NPC text (#19) for this turn,
    handed to Jev in the state payload so it can react to story text RAM
    can't decode; `None` (the default) when there's none. It's a keyword
    argument sourced by the caller (`run_turn`'s `dialog_text_source` seam),
    never read from `state`, keeping `GameState` RAM-only.
    """
    milestone = milestone_progress.current
    state_payload = _serialize_state(state, milestone, dialog_text)
    result = jev_client.system_one(
        state_payload, {_ACTION_QUESTION_ID: _build_action_question(state, milestone)}
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
    dialog_text_source: DialogTextSource | None = None,
) -> Decision | None:
    """Run one full turn: read state, ask Jev, act, log - in that order.

    The chosen action is executed immediately with no confidence-based
    retry/escalation path, and every decision is logged unconditionally via
    `on_decision` (both per this ticket's MVP scope).

    `dialog_text_source`, when given, is the loop's vision-fallback seam
    (#19): it's handed the freshly-read `GameState` and returns the decoded
    dialog text to offer Jev this turn (or `None`). Deciding whether to
    decode is the source's own trigger - it returns `None` when no dialog is
    open or the fallback isn't configured - so `run_turn` stays agnostic to
    how/whether vision is wired, and its own ordering stays the single
    canonical per-turn cycle (see module docstring).

    Returns `None`, executing no action and logging nothing, if `jev_client`
    raises `resilience.TurnSkipped` (#56: a `ResilientJevClient` wrapping the
    real client that exhausted its retries for this turn) - the one
    exception to "act regardless of confidence" above, since there's no
    Decision to act on at all.
    """
    state = state_source()
    milestone_progress = track_milestones(state.event_flags, state.badges)
    dialog_text = dialog_text_source(state) if dialog_text_source is not None else None
    try:
        decision = decide_action(
            jev_client, state, milestone_progress, dialog_text=dialog_text
        )
    except TurnSkipped:
        logger.warning("turn skipped: Jev retries exhausted, no action taken")
        return None
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
    read_menu_cursor: Callable[[PyBoy], int] = navigation.read_menu_cursor,
    render_frame: Callable[[PyBoy], None] = emulator.render_current_frame,
) -> ActionExecutor:
    """Build the real `execute_action` callable for `run_turn`, against `pyboy`.

    The `extract_state`/`resolve_target`/`run_macro`/`press_button`/
    `read_menu_cursor`/`render_frame` seams default to the real
    `game_state`/`navigation`/`emulator` functions and only need overriding
    in tests (per this ticket's no-real-PyBoy test-coverage requirement) -
    production callers just pass `pyboy`.

    Dispatches a raw button straight to `press_button`. The navigation macro
    action re-reads the current milestone itself (rather than reusing the
    triggering `Decision`'s own snapshot) so it always targets whatever's
    current at execution time, and is a no-op when that milestone has no
    verified tile target yet (see `navigation.py`'s docstring -
    `resolve_navigation_target` returns `None` in that case). A dynamic
    battle action (#54's `USE_MOVE_<slot>`/`USE_ITEM_<item>`/
    `SWITCH_TO_<slot>`/`RUN`) is translated into the matching button sequence
    by `_execute_battle_action` (#76) instead, since none of those are legal
    `press_button` arguments on their own.

    `render_frame` (#81) only runs when the turn's dispatch was itself a
    no-op: `NAVIGATION_MACRO_ACTION` whenever `resolve_target` can't resolve
    a target (true for every milestone today - see above), or a malformed
    battle action (an unparseable slot, an item not in `state.inventory`,
    both reported back via `_execute_battle_action`'s return value). Those
    are the only two paths that leave `pyboy` completely untouched - every
    other dispatch already ticks with `render=True` inside `press_button`
    (directly, or via `_execute_battle_action`/`run_macro`), so calling
    `render_frame` there too would just pay a redundant extra tick forever.
    A perpetually-untouched `pyboy` starves `frame_capture.py`'s background
    capture of any rendered frame to read - its `capture_screen` read has no
    ticking of its own (see `main.main`'s wiring) - which is exactly #81's
    bug: a run where Jev keeps picking the (today, always no-op) navigation
    macro never renders a single frame, so `/video.mjpg` stays solid white
    forever.

    This is a deliberate narrower fix than #81's other two "worth checking"
    angles: it doesn't add any cross-thread ticking of `pyboy` from
    `frame_capture.py`'s own background capture thread (the tactical loop's
    thread is the only thread that ever calls `pyboy.tick()`, and it stays
    that way - ticking `pyboy` from two threads at once is unsupported and
    would risk corrupting `navigation.execute_button`'s own precisely-counted
    tick loops), and it doesn't address a possible read/tick data race
    between that capture thread and the tactical loop (pre-existing, not
    introduced or worsened here - #81's own symptom, a clean uniform-white
    frame with no corruption/errors, matches a screen that was simply never
    rendered, not a torn or raced one).
    """

    def execute_action(action: str) -> None:
        if action == NAVIGATION_MACRO_ACTION:
            state = extract_state(pyboy)
            milestone_progress = track_milestones(state.event_flags, state.badges)
            target = resolve_target(milestone_progress.current)
            if target is not None:
                run_macro(pyboy, target)
            else:
                render_frame(pyboy)
        elif _is_battle_menu_action(action):
            state = extract_state(pyboy)
            pressed = _execute_battle_action(
                pyboy,
                action,
                state,
                press_button=press_button,
                read_menu_cursor=read_menu_cursor,
            )
            if not pressed:
                render_frame(pyboy)
        else:
            press_button(pyboy, action)

    return execute_action
