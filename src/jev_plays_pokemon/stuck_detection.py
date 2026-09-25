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

Recovery is three-tier, tracked by `StuckRecoveryPolicy`:

1. Nudge: the first stuck detection injects one random legal action to
   break the input pattern - no reload.
2. Reload: if stuck re-triggers on the second consecutive detection after
   a nudge (three consecutive detections total, counting the one that
   triggered the nudge), escalate to reloading the latest persisted
   snapshot (`emulator.py`, #55). A detection streak that breaks (position
   moves again) resets the policy back to tier 1 for next time.
3. Give up: after `DEFAULT_MAX_RELOADS` reloads that never took - nothing
   moved the player from where the reload left them - `wrap_for_stuck_detection`
   raises `StuckRunAborted` instead of reloading again. Without this the
   ladder had no floor: #59's run re-detected every ~100 turns, reloaded
   something that either didn't exist yet or was itself saved mid-stuck,
   cleared its history, and billed another ~100 decisions - 570 of them in
   ~102 seconds, unbounded. Aborting costs a bounded number of calls and
   exits with `watchdog.EXIT_CODE_STUCK_ABORT`, which the watchdog
   deliberately does not restart on: restarting would resume the same stuck
   state.

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
# Reloads that never took - each followed by no non-stuck turn at all - before
# the run gives up entirely. Each futile reload costs roughly
# `DEFAULT_STUCK_THRESHOLD` more billed decisions before it is even detected,
# so this is the cap on how much a permanently-stuck run can spend: ~3
# reloads + the detections between them is a few hundred decisions instead of
# an unbounded number until a human notices (#59 billed 570 in ~102s).
DEFAULT_MAX_RELOADS = 3

Position = tuple[int, int, int]  # (map_id, player_x, player_y)


class StuckRunAborted(RuntimeError):
    """Raised out of the wrapped `execute_action` seam when the run has
    exhausted its recovery ladder without ever unsticking.

    Propagates through `main.run_loop` (nothing in `decision.run_turn`
    catches it - only `resilience.TurnSkipped`, and only around the Jev call,
    not around `execute_action`) to `main.main`, which logs it and exits with
    `watchdog.EXIT_CODE_STUCK_ABORT` rather than the generic crash code, so
    the watchdog can tell "this run decided to stop" from "this run broke".
    """


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
    ABORT = "abort"


class StuckRecoveryPolicy:
    """Tracks the nudge/reload/give-up escalation state machine across turns.

    `on_turn(stuck)` is the normal interface: feed it this turn's `is_stuck`
    verdict, get back the recovery tier to apply. A `False` at any point
    resets the streak - a run that's moving again starts back at tier 1
    (nudge) the next time it gets stuck.

    The reload budget deliberately does *not* come back on a `False` verdict,
    because reloading clears the caller's history and the turn right after
    that is never stuck yet - keying the budget to `stuck` would hand it back
    on every single reload and bound nothing. `note_progress()` is the
    caller's way of reporting the signal that actually means something: the
    run has been somewhere since the last reload that the reload itself did
    not put it at.
    """

    def __init__(
        self,
        escalation_threshold: int = DEFAULT_ESCALATION_THRESHOLD,
        max_reloads: int = DEFAULT_MAX_RELOADS,
    ) -> None:
        self._escalation_threshold = escalation_threshold
        self._max_reloads = max_reloads
        self._consecutive_stuck = 0
        self._reloads = 0

    def note_progress(self) -> None:
        """Report that the run has made real progress, restoring the reload
        budget (#59's run never earned this back - it reloaded, re-detected at
        the same spot, and reloaded again)."""
        self._reloads = 0

    def on_turn(self, stuck: bool) -> RecoveryTier:
        if not stuck:
            self._consecutive_stuck = 0
            return RecoveryTier.NONE
        self._consecutive_stuck += 1
        if self._consecutive_stuck >= self._escalation_threshold:
            self._consecutive_stuck = 0
            if self._reloads >= self._max_reloads:
                return RecoveryTier.ABORT
            self._reloads += 1
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
    max_reloads: int = DEFAULT_MAX_RELOADS,
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

    `max_reloads` bounds the reload tier: once that many reloads have gone by
    without the player getting anywhere other than where the reload itself
    left them, the wrapped `execute_action` raises `StuckRunAborted` instead
    of reloading again, so a permanently-stuck run stops itself after a
    bounded spend rather than billing forever (#59's run reached 570
    decisions this way). A run that does walk somewhere new earns its budget
    back. `max_reloads=0` makes the first escalation that reaches the reload
    tier abort outright.
    """
    history: deque[TurnRecord] = deque(maxlen=threshold)
    policy = StuckRecoveryPolicy(
        escalation_threshold=escalation_threshold, max_reloads=max_reloads
    )
    last_position: list[Position | None] = [None]
    last_milestone: list[Milestone | None] = [None]
    # Where the most recent reload left the player, or None once they have
    # been somewhere else since (see the progress check below).
    reloaded_into: list[Position | None] = [None]

    def _position_of(state: GameState) -> Position:
        return (state.map_id, state.player_x, state.player_y)

    def wrapped_state_source() -> GameState:
        state = state_source()
        last_position[0] = _position_of(state)
        last_milestone[0] = track_milestones(state.event_flags, state.badges).current
        return state

    def wrapped_execute_action(action: str) -> None:
        execute_action(action)

        position = last_position[0]
        if position is None:
            # state_source hasn't been called yet this run - nothing to
            # pair this action with; skip recording rather than guess.
            return

        if reloaded_into[0] is not None and position != reloaded_into[0]:
            # Somewhere other than where the last reload dropped the player:
            # the run has moved under its own steam since, so the reload
            # ladder has something to work with again. Compared against the
            # post-reload position, not the position it was stuck at - a
            # reload to the boot state changes the position on its own, and
            # that is not the run making progress.
            policy.note_progress()
            reloaded_into[0] = None

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
            # Read the reloaded state back: `load_snapshot` can put the player
            # anywhere (the boot state's bedroom, most often), and the progress
            # check above only means anything against where it actually landed.
            reloaded_into[0] = _position_of(state_source())
        elif tier is RecoveryTier.ABORT:
            raise StuckRunAborted(
                f"stuck persisted through {max_reloads} snapshot reload(s), each "
                f"detected over a {threshold}-turn window; giving up rather than "
                "billing more decisions against a state recovery cannot fix"
            )

    return wrapped_state_source, wrapped_execute_action
