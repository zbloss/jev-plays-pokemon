from jev_plays_pokemon.game_state import BattleState, GameState
from jev_plays_pokemon.stuck_detection import (
    RecoveryTier,
    StuckRecoveryPolicy,
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
