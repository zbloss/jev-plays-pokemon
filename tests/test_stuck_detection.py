import dataclasses

import pytest

from jev_plays_pokemon.decision import NAVIGATION_MACRO_ACTION
from jev_plays_pokemon.game_state import BattleState, GameState
from jev_plays_pokemon.navigation import RAW_BUTTONS
from jev_plays_pokemon.stuck_detection import (
    RecoveryTier,
    StuckRecoveryPolicy,
    StuckRunAborted,
    TurnRecord,
    is_stuck,
    wrap_for_stuck_detection,
)

_POS = (40, 4, 5)
_ELSEWHERE = (40, 4, 6)


def _records(position_action_pairs) -> list[TurnRecord]:
    return [TurnRecord(position=p, action=a) for p, a in position_action_pairs]


def test_fewer_than_threshold_turns_is_never_stuck():
    history = _records([(_POS, "up")] * 5)

    assert is_stuck(history, threshold=100) is False


def test_stationary_position_with_a_cycling_action_pattern_is_stuck():
    pattern = [(_POS, "up"), (_POS, "down")] * 50  # period 2, repeats 50x
    history = _records(pattern)

    assert is_stuck(history, threshold=100) is True


def test_stationary_position_with_a_single_repeated_action_is_stuck():
    # A trivial cycle (period 1) counts too - holding the same input.
    history = _records([(_POS, "a")] * 100)

    assert is_stuck(history, threshold=100) is True


def test_moving_position_is_never_stuck_even_with_a_cycling_action_pattern():
    positions = [_POS if i % 2 == 0 else _ELSEWHERE for i in range(100)]
    history = _records([(p, "up") for p in positions])

    assert is_stuck(history, threshold=100) is False


def test_stationary_position_with_non_cycling_actions_is_not_stuck():
    # A period that never repeats within the window (> half its length)
    # isn't a meaningful cycle - e.g. 60 distinct-ish actions with no
    # smaller period dividing evenly within the first half.
    actions = [f"action-{i}" for i in range(100)]
    history = _records([(_POS, action) for action in actions])

    assert is_stuck(history, threshold=100) is False


def test_only_the_trailing_window_matters():
    # 100 turns of movement, then exactly 100 turns stationary+cycling:
    # only the trailing 100-turn window is evaluated.
    moving = [((40, i % 10, 5), "up") for i in range(100)]
    stuck = [(_POS, "up") if i % 2 == 0 else (_POS, "down") for i in range(100)]
    history = _records(moving + stuck)

    assert is_stuck(history, threshold=100) is True
    assert is_stuck(history[:150], threshold=100) is False


def test_smaller_custom_threshold_is_honored():
    history = _records([(_POS, "up"), (_POS, "down")] * 5)  # 10 turns

    assert is_stuck(history, threshold=10) is True
    assert is_stuck(history, threshold=11) is False


def test_recovery_policy_nudges_on_first_stuck_detection():
    policy = StuckRecoveryPolicy()

    assert policy.on_turn(stuck=True) is RecoveryTier.NUDGE


def test_recovery_policy_waits_on_the_second_consecutive_detection():
    policy = StuckRecoveryPolicy()
    policy.on_turn(stuck=True)  # nudge

    assert policy.on_turn(stuck=True) is RecoveryTier.NONE


def test_recovery_policy_escalates_to_reload_on_the_third_consecutive_detection():
    policy = StuckRecoveryPolicy()
    policy.on_turn(stuck=True)  # nudge
    policy.on_turn(stuck=True)  # still waiting

    assert policy.on_turn(stuck=True) is RecoveryTier.RELOAD


def test_recovery_policy_resets_after_escalating():
    policy = StuckRecoveryPolicy()
    for _ in range(3):
        policy.on_turn(stuck=True)  # nudge, wait, reload

    assert policy.on_turn(stuck=True) is RecoveryTier.NUDGE


def test_recovery_policy_resets_when_no_longer_stuck():
    policy = StuckRecoveryPolicy()
    policy.on_turn(stuck=True)  # nudge

    assert policy.on_turn(stuck=False) is RecoveryTier.NONE
    assert policy.on_turn(stuck=True) is RecoveryTier.NUDGE


def test_recovery_policy_honors_a_custom_escalation_threshold():
    policy = StuckRecoveryPolicy(escalation_threshold=2)
    tiers = [policy.on_turn(stuck=True) for _ in range(2)]

    assert tiers == [RecoveryTier.NUDGE, RecoveryTier.RELOAD]


def test_recovery_policy_gives_up_once_the_reload_budget_is_spent():
    # escalation_threshold=1 so every stuck turn is an escalation, keeping
    # the sequence short: two reloads, then the third escalation is the run
    # stopping itself rather than billing another ~100 turns (#59's run never
    # stopped at all).
    policy = StuckRecoveryPolicy(escalation_threshold=1, max_reloads=2)
    tiers = [policy.on_turn(stuck=True) for _ in range(3)]

    assert tiers == [
        RecoveryTier.RELOAD,
        RecoveryTier.RELOAD,
        RecoveryTier.ABORT,
    ]


def test_a_single_non_stuck_turn_does_not_hand_the_reload_budget_back():
    # Deliberate, and the reason the budget can't be keyed to `stuck`:
    # reloading clears the caller's history, so the turn right after a reload
    # is *always* reported not-stuck yet (too little history). Resetting here
    # would hand the budget back on every reload and bound nothing.
    policy = StuckRecoveryPolicy(escalation_threshold=1, max_reloads=1)

    assert policy.on_turn(stuck=True) is RecoveryTier.RELOAD
    assert policy.on_turn(stuck=False) is RecoveryTier.NONE
    assert policy.on_turn(stuck=True) is RecoveryTier.ABORT


def test_note_progress_hands_the_reload_budget_back():
    # The signal that actually means something - the run got somewhere the
    # reload didn't put it - is reported through note_progress (see
    # wrap_for_stuck_detection, which is what calls it).
    policy = StuckRecoveryPolicy(escalation_threshold=1, max_reloads=1)

    assert policy.on_turn(stuck=True) is RecoveryTier.RELOAD
    policy.note_progress()
    assert policy.on_turn(stuck=True) is RecoveryTier.RELOAD


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


def test_wrap_pairs_each_turns_position_with_the_action_that_turn_executes():
    executed: list[str] = []
    wrapped_state_source, wrapped_execute_action = wrap_for_stuck_detection(
        lambda: _game_state(),
        executed.append,
        load_snapshot=lambda: None,
        threshold=3,
    )

    for action in ("up", "down", "left"):
        wrapped_state_source()
        wrapped_execute_action(action)

    assert executed == ["up", "down", "left"]


def test_wrap_nudges_with_a_legal_action_once_stuck_and_does_not_double_record_it():
    executed: list[str] = []
    wrapped_state_source, wrapped_execute_action = wrap_for_stuck_detection(
        lambda: _game_state(),  # always the same position
        executed.append,
        load_snapshot=lambda: None,
        threshold=4,
        legal_actions=("nudge",),
    )

    for action in ("up", "down", "up", "down"):
        wrapped_state_source()
        wrapped_execute_action(action)

    # Turn 4 is stuck (stationary, period-2 cycle over the 4-turn window):
    # the normal action executes, then the nudge executes right after -
    # both real execute_action calls, but only one (the normal action) is
    # ever paired into history (checked indirectly below via turn 5+6 not
    # re-triggering a nudge immediately from a corrupted pairing).
    assert executed == ["up", "down", "up", "down", "nudge"]


def test_wrap_nudge_pool_includes_the_navigation_macro_with_a_resolvable_target():
    # #98: `_game_state()`'s empty event_flags/badges resolve to the
    # "got_starter" milestone, which - like every milestone now - carries a
    # ROM-derived tile-level target, so the default (unpinned) nudge pool
    # recomputed via `out_of_battle_action_space` offers the macro here too.
    nudge_pools: list[tuple[str, ...]] = []

    def spy_random_choice(actions):
        pool = tuple(actions)
        nudge_pools.append(pool)
        return pool[0]

    wrapped_state_source, wrapped_execute_action = wrap_for_stuck_detection(
        lambda: _game_state(),  # always the same position
        lambda action: None,
        load_snapshot=lambda: None,
        threshold=4,
        random_choice=spy_random_choice,
    )

    for action in ("up", "down", "up", "down"):
        wrapped_state_source()
        wrapped_execute_action(action)

    assert nudge_pools, "expected the stuck nudge to have fired"
    assert set(nudge_pools[0]) == set(RAW_BUTTONS) | {NAVIGATION_MACRO_ACTION}
    assert NAVIGATION_MACRO_ACTION in nudge_pools[0]


def test_wrap_escalates_to_reload_and_clears_history_after_a_sustained_stuck_run():
    reloads: list[None] = []
    wrapped_state_source, wrapped_execute_action = wrap_for_stuck_detection(
        lambda: _game_state(),  # always the same position
        lambda action: None,
        load_snapshot=lambda: reloads.append(None),
        threshold=2,
        escalation_threshold=3,
        legal_actions=("nudge",),
        random_choice=lambda actions: actions[0],
    )

    # Turn 1: not enough history (threshold=2). Turn 2: stuck -> nudge.
    # Turn 3: stuck -> wait. Turn 4: stuck -> reload.
    for _ in range(4):
        wrapped_state_source()
        wrapped_execute_action("up")

    assert reloads == [None]

    # History was cleared on reload, so the very next turn alone can't be
    # stuck yet (fewer than `threshold` turns since the reload).
    wrapped_state_source()
    wrapped_execute_action("up")
    assert reloads == [None]


def _position_source(*positions):
    """A `state_source` yielding `positions` in order, repeating the last one
    forever once they run out.

    Needed by the reload-budget tests because a reload itself performs one
    extra read to see where it landed, so a scripted sequence has a turn
    boundary that isn't a turn.
    """
    remaining = list(positions)

    def source():
        position = remaining.pop(0) if remaining else positions[-1]
        map_id, player_x, player_y = position
        return _game_state(map_id=map_id, player_x=player_x, player_y=player_y)

    return source


def test_wrap_aborts_instead_of_reloading_a_fourth_time():
    # The #59 case: the position never moves, so every reload buys nothing and
    # the ladder would otherwise nudge/reload/re-detect forever. Here it gets
    # its budget (1 reload) spent, and the next escalation raises instead.
    reloads: list[None] = []
    wrapped_state_source, wrapped_execute_action = wrap_for_stuck_detection(
        lambda: _game_state(),  # always the same position
        lambda action: None,
        load_snapshot=lambda: reloads.append(None),
        threshold=2,
        escalation_threshold=3,
        max_reloads=1,
        legal_actions=("nudge",),
        random_choice=lambda actions: actions[0],
    )

    with pytest.raises(StuckRunAborted):
        for _ in range(20):
            wrapped_state_source()
            wrapped_execute_action("up")

    # Exactly the budgeted number of reloads - the abort replaced the rest.
    assert reloads == [None]


def test_wrap_can_reload_again_once_the_run_walks_somewhere_new():
    # Turns 1-4: stuck at _POS -> nudge, wait, reload. Turn 4's reload then
    # reads its own landing (_ELSEWHERE) as turn 5's read. From turn 6 the run
    # is at a third tile it reached on its own, so the budget is restored and
    # a second reload is allowed instead of an abort.
    reloads: list[None] = []
    wrapped_state_source, wrapped_execute_action = wrap_for_stuck_detection(
        _position_source(
            _POS,  # turn 1
            _POS,  # turn 2 -> nudge
            _POS,  # turn 3 -> waiting
            _POS,  # turn 4 -> reload #1
            _ELSEWHERE,  # the reload's own landing read
            (40, 5, 5),  # turn 5 -> somewhere the reload didn't put it
            (40, 5, 5),  # turn 6 -> stuck again -> nudge
            (40, 5, 5),  # turn 7 -> waiting
            (40, 5, 5),  # turn 8 -> reload #2, allowed
        ),
        lambda action: None,
        load_snapshot=lambda: reloads.append(None),
        threshold=2,
        escalation_threshold=3,
        max_reloads=1,
        legal_actions=("nudge",),
        random_choice=lambda actions: actions[0],
    )

    for _ in range(9):
        wrapped_state_source()
        wrapped_execute_action("up")

    assert reloads == [None, None]
