"""Stuck detection with nudge-to-snapshot-reload escalation (#57).

A pure function over recent turn history (player position + chosen-action
sequence) - not milestone progress, which is too coarse a signal: the
~12-beat milestone list legitimately spans many turns per beat, so a run
can sit "between milestones" for a long, entirely healthy stretch.
Milestone completion is never treated as, or conflated with, the stuck
signal here.

Declares stuck when position has been stationary *and* the chosen-action
sequence has cycled, over a tunable trailing window (`DEFAULT_STUCK_THRESHOLD`,
100 turns - ~30s at the tactical loop's ~3.3 turns/s, per
`docs/turn-rate-budget.md`).

Recovery is two-tier, tracked by `StuckRecoveryPolicy`:

1. Nudge: the first stuck detection injects one random legal action to
   break the input pattern - no reload.
2. Reload: if stuck re-triggers on the second consecutive detection after
   a nudge (three consecutive detections total, counting the one that
   triggered the nudge), escalate to reloading the latest persisted
   snapshot (`emulator.py`, #55). A detection streak that breaks (position
   moves again) resets the policy back to tier 1 for next time.

`wrap_for_stuck_detection` is the production integration point: a thin
wrapper seam around `main.run_loop`'s own `state_source`/`execute_action`
seams, not a change woven into `decision.run_turn` itself. It pairs each
turn's freshly-read position with the action `run_turn` goes on to execute,
appends it to a rolling history, and applies whatever recovery tier that
turn's `is_stuck` check calls for.
"""

from __future__ import annotations

import logging
import random
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum

from jev_plays_pokemon.decision import out_of_battle_action_space
from jev_plays_pokemon.game_state import GameState
from jev_plays_pokemon.milestones import Milestone, track_milestones

logger = logging.getLogger(__name__)

DEFAULT_STUCK_THRESHOLD = 100
# Consecutive stuck detections - counting the one that triggers the nudge -
# before escalating to a snapshot reload.
DEFAULT_ESCALATION_THRESHOLD = 3

Position = tuple[int, int, int]  # (map_id, player_x, player_y)


@dataclass(frozen=True)
class TurnRecord:
    position: Position
    action: str


def _smallest_period(actions: Sequence[str]) -> int:
    """The smallest `p` such that `actions[i] == actions[i % p]` for every
    `i` - standard "smallest repeating period" of a sequence. `p == len
    (actions)` means no smaller period fits (the sequence never repeats).
    """
    n = len(actions)
    for p in range(1, n + 1):
        if all(actions[i] == actions[i % p] for i in range(n)):
            return p
    return n  # unreachable: p == n always satisfies the check above


def is_stuck(
    history: Sequence[TurnRecord], threshold: int = DEFAULT_STUCK_THRESHOLD
) -> bool:
    """True iff the most recent `threshold` turns are all at the same
    position *and* their action sequence cycles with a period short enough
    to have repeated at least twice within that window.

    Pure and independent of PyBoy/the ROM/milestone state: `history` is
    just data. Fewer than `threshold` turns of history is never stuck -
    there isn't enough signal yet.
    """
    if len(history) < threshold:
        return False
    # `list(...)` first: a `deque` (production's rolling-history container)
    # doesn't support slicing directly.
    window = list(history)[-threshold:]
    positions = {record.position for record in window}
    if len(positions) != 1:
        return False
    actions = [record.action for record in window]
    period = _smallest_period(actions)
    return period <= len(actions) // 2


class RecoveryTier(Enum):
    NONE = "none"
    NUDGE = "nudge"
    RELOAD = "reload"


class StuckRecoveryPolicy:
    """Tracks the nudge/reload escalation state machine across turns.

    `on_turn(stuck)` is the whole interface: feed it this turn's `is_stuck`
    verdict, get back the recovery tier to apply. A `False` at any point
    resets the streak - a run that's moving again starts back at tier 1
    (nudge) the next time it gets stuck.
    """

    def __init__(
        self, escalation_threshold: int = DEFAULT_ESCALATION_THRESHOLD
    ) -> None:
        self._escalation_threshold = escalation_threshold
        self._consecutive_stuck = 0

    def on_turn(self, stuck: bool) -> RecoveryTier:
        if not stuck:
            self._consecutive_stuck = 0
            return RecoveryTier.NONE
        self._consecutive_stuck += 1
        if self._consecutive_stuck >= self._escalation_threshold:
            self._consecutive_stuck = 0
            return RecoveryTier.RELOAD
        if self._consecutive_stuck == 1:
            return RecoveryTier.NUDGE
        return RecoveryTier.NONE


def wrap_for_stuck_detection(
    state_source: Callable[[], GameState],
    execute_action: Callable[[str], None],
    *,
    load_snapshot: Callable[[], None],
    threshold: int = DEFAULT_STUCK_THRESHOLD,
    escalation_threshold: int = DEFAULT_ESCALATION_THRESHOLD,
    legal_actions: Sequence[str] | None = None,
    random_choice: Callable[[Sequence[str]], str] = random.choice,
) -> tuple[Callable[[], GameState], Callable[[str], None]]:
    """Build instrumented `(state_source, execute_action)` seams that add
    stuck detection/recovery (#57) as a side effect, for handing to the
    unmodified `main.run_loop` - a wrapper around its seams, not a change
    inside `run_turn`.

    `state_source` is called once per turn, before `execute_action` (the
    order `run_turn` already uses), so the wrapped pair correctly pairs
    each turn's position (read at the wrapped `state_source` call) with the
    action `run_turn` goes on to execute that same turn.

    `legal_actions`, when left at its default of `None`, is recomputed every
    nudge from that turn's own current milestone via `decision.
    out_of_battle_action_space` - the same "can the macro actually resolve a
    destination right now?" predicate the per-turn Choice itself is built
    from (#83), so an injected legal action is never a no-op. Pass an
    explicit sequence to pin the nudge pool instead (tests only).

    The nudge tier calls the *original*, unwrapped `execute_action` for its
    injected button, not the wrapped one - the nudge is an out-of-band
    recovery action, not another turn, so it isn't itself recorded into the
    history `is_stuck` evaluates.
    """
    history: deque[TurnRecord] = deque(maxlen=threshold)
    policy = StuckRecoveryPolicy(escalation_threshold=escalation_threshold)
    last_position: list[Position | None] = [None]
    last_milestone: list[Milestone | None] = [None]

    def wrapped_state_source() -> GameState:
        state = state_source()
        last_position[0] = (state.map_id, state.player_x, state.player_y)
        last_milestone[0] = track_milestones(state.event_flags, state.badges).current
        return state

    def wrapped_execute_action(action: str) -> None:
        execute_action(action)

        position = last_position[0]
        if position is None:
            # state_source hasn't been called yet this run - nothing to
            # pair this action with; skip recording rather than guess.
            return
        history.append(TurnRecord(position=position, action=action))

        stuck = is_stuck(history, threshold=threshold)
        tier = policy.on_turn(stuck)
        if tier is RecoveryTier.NUDGE:
            pool = (
                legal_actions
                if legal_actions is not None
                else out_of_battle_action_space(last_milestone[0], position[0])
            )
            nudge_action = random_choice(pool)
            logger.warning("stuck detected; nudging with action=%s", nudge_action)
            execute_action(nudge_action)
        elif tier is RecoveryTier.RELOAD:
            logger.warning("stuck persisted after nudge; reloading the latest snapshot")
            load_snapshot()
            history.clear()

    return wrapped_state_source, wrapped_execute_action
