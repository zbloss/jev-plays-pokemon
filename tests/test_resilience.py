import pytest
from typesafe_sdk import Choice

from jev_plays_pokemon.resilience import (
    AttemptEvent,
    ResilientJevClient,
    TurnSkipped,
)


class _FakeChoiceAnswer:
    def __init__(self, choice: str, confidence: object) -> None:
        self.choice = choice
        self.confidence = confidence


class _FakeSystemOneResult:
    def __init__(self, choices: dict) -> None:
        self.choices = choices


_QUESTIONS = {
    "action": Choice(
        instructions="pick one", criteria={"a": "criterion a", "b": "criterion b"}
    )
}
_VALID_RESULT = _FakeSystemOneResult({"action": _FakeChoiceAnswer("a", 0.9)})


class _ScriptedJevClient:
    """Hands out the next scripted outcome per `system_one` call: an
    exception instance is raised, anything else is returned - no real
    TypeSafe API call."""

    def __init__(self, outcomes: list) -> None:
        self._outcomes = outcomes
        self.calls = 0

    def system_one(self, state, questions):
        outcome = self._outcomes[self.calls]
        self.calls += 1
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _make(client, **kwargs) -> ResilientJevClient:
    kwargs.setdefault("sleep", lambda seconds: None)
    return ResilientJevClient(client, **kwargs)


def test_succeeds_on_first_attempt_with_no_retry_or_sleep():
    client = _ScriptedJevClient([_VALID_RESULT])
    sleeps: list[float] = []
    events: list[AttemptEvent] = []
    resilient = _make(client, sleep=sleeps.append, on_attempt=events.append)

    result = resilient.system_one({}, _QUESTIONS)

    assert result is _VALID_RESULT
    assert client.calls == 1
    assert sleeps == []
    assert [e.kind for e in events] == ["attempt_succeeded"]


def test_retries_on_a_transport_failure_then_succeeds():
    client = _ScriptedJevClient([RuntimeError("timeout"), _VALID_RESULT])
    sleeps: list[float] = []
    events: list[AttemptEvent] = []
    resilient = _make(client, sleep=sleeps.append, on_attempt=events.append)

    result = resilient.system_one({}, _QUESTIONS)

    assert result is _VALID_RESULT
    assert client.calls == 2
    assert sleeps == [1.0]  # base backoff before the one retry
    assert [e.kind for e in events] == [
        "attempt_failed",
        "backoff",
        "attempt_succeeded",
    ]
    assert events[0].error is not None


@pytest.mark.parametrize(
    "invalid_result",
    [
        _FakeSystemOneResult({}),  # no answer at all for "action"
        _FakeSystemOneResult({"action": _FakeChoiceAnswer("not_offered", 0.9)}),
        _FakeSystemOneResult({"action": _FakeChoiceAnswer("a", "not-a-number")}),
    ],
    ids=["missing_answer", "choice_not_offered", "non_numeric_confidence"],
)
def test_an_invalid_or_unparseable_response_is_retried_like_a_transport_failure(
    invalid_result,
):
    client = _ScriptedJevClient([invalid_result, _VALID_RESULT])
    resilient = _make(client)

    result = resilient.system_one({}, _QUESTIONS)

    assert result is _VALID_RESULT
    assert client.calls == 2


def test_exhausts_after_the_initial_attempt_plus_three_retries_and_raises():
    failures = [RuntimeError(f"fail-{i}") for i in range(4)]
    client = _ScriptedJevClient(failures)
    sleeps: list[float] = []
    resilient = _make(client, sleep=sleeps.append)

    with pytest.raises(TurnSkipped):
        resilient.system_one({}, _QUESTIONS)

    assert client.calls == 4  # 1 initial + 3 retries
    assert sleeps == [1.0, 2.0, 4.0]


def test_turn_skipped_chains_the_last_underlying_error():
    last_error = RuntimeError("final failure")
    client = _ScriptedJevClient(
        [RuntimeError("a"), RuntimeError("b"), RuntimeError("c"), last_error]
    )
    resilient = _make(client)

    with pytest.raises(TurnSkipped) as excinfo:
        resilient.system_one({}, _QUESTIONS)

    assert excinfo.value.__cause__ is last_error


def test_inter_turn_backoff_escalates_across_skipped_turns_and_resets_on_success():
    # Turn 1: 4 failures -> skipped. Turn 2: 4 failures -> skipped (backoff
    # doubles). Turn 3: succeeds immediately (backoff resets to 0). Turn 4:
    # no inter-turn backoff, since it was just reset.
    outcomes = [RuntimeError("t1")] * 4 + [RuntimeError("t2")] * 4 + [_VALID_RESULT] * 2
    client = _ScriptedJevClient(outcomes)
    events: list[AttemptEvent] = []
    resilient = _make(client, on_attempt=events.append)

    with pytest.raises(TurnSkipped):
        resilient.system_one({}, _QUESTIONS)  # turn 1
    with pytest.raises(TurnSkipped):
        resilient.system_one({}, _QUESTIONS)  # turn 2
    resilient.system_one({}, _QUESTIONS)  # turn 3
    resilient.system_one({}, _QUESTIONS)  # turn 4

    inter_turn_backoffs = [
        e.backoff_seconds for e in events if e.kind == "inter_turn_backoff"
    ]
    # Turn 1 has none (nothing to escalate from yet); turn 2 waits the base
    # 1s; turn 3 waits the doubled 2s (and resets on its success); turn 4
    # has none, since turn 3 reset it.
    assert inter_turn_backoffs == [1.0, 2.0]


def test_inter_turn_backoff_is_capped():
    failures_per_turn = 4
    turns = 6
    outcomes = [RuntimeError("x")] * (failures_per_turn * turns)
    client = _ScriptedJevClient(outcomes)
    events: list[AttemptEvent] = []
    resilient = _make(
        client,
        max_inter_turn_backoff_seconds=5.0,
        on_attempt=events.append,
    )

    for _ in range(turns):
        with pytest.raises(TurnSkipped):
            resilient.system_one({}, _QUESTIONS)

    inter_turn_backoffs = [
        e.backoff_seconds for e in events if e.kind == "inter_turn_backoff"
    ]
    # 1, 2, 4, capped at 5 thereafter (never 8, 16, ...).
    assert inter_turn_backoffs == [1.0, 2.0, 4.0, 5.0, 5.0]


def test_on_attempt_hook_is_optional():
    # No on_attempt given - must not raise just because nothing observes it.
    client = _ScriptedJevClient([_VALID_RESULT])
    resilient = _make(client)

    resilient.system_one({}, _QUESTIONS)
