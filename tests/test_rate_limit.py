import pytest
from typesafe_sdk import Choice

from jev_plays_pokemon.rate_limit import RateLimitedJevClient

_QUESTIONS = {
    "action": Choice(
        instructions="pick one", criteria={"a": "criterion a", "b": "criterion b"}
    )
}
_RESULT = object()


class _CountingJevClient:
    def __init__(self) -> None:
        self.calls = 0

    def system_one(self, state, questions):
        self.calls += 1
        return _RESULT


class _FakeClock:
    """A controllable `time_source`; `advance` simulates the wrapped call
    itself (or an intervening sleep) taking time."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


def _make(client, max_calls_per_second, clock, sleeps):
    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        clock.advance(seconds)

    return RateLimitedJevClient(
        client,
        max_calls_per_second=max_calls_per_second,
        sleep=sleep,
        time_source=clock,
    )


def test_first_call_never_waits():
    client = _CountingJevClient()
    clock = _FakeClock()
    sleeps: list[float] = []
    limited = _make(client, 1.0, clock, sleeps)

    result = limited.system_one({}, _QUESTIONS)

    assert result is _RESULT
    assert client.calls == 1
    assert sleeps == []


def test_a_second_call_arriving_immediately_waits_out_the_full_interval():
    client = _CountingJevClient()
    clock = _FakeClock()
    sleeps: list[float] = []
    limited = _make(client, 2.0, clock, sleeps)  # 0.5s interval

    limited.system_one({}, _QUESTIONS)
    limited.system_one({}, _QUESTIONS)

    assert client.calls == 2
    assert sleeps == [0.5]


def test_a_call_that_arrives_after_the_interval_has_already_elapsed_does_not_wait():
    client = _CountingJevClient()
    clock = _FakeClock()
    sleeps: list[float] = []
    limited = _make(client, 2.0, clock, sleeps)  # 0.5s interval

    limited.system_one({}, _QUESTIONS)
    clock.advance(0.6)  # the "underlying call" itself took longer than the interval
    limited.system_one({}, _QUESTIONS)

    assert client.calls == 2
    assert sleeps == []


def test_only_waits_out_the_remaining_interval_when_partially_elapsed():
    client = _CountingJevClient()
    clock = _FakeClock()
    sleeps: list[float] = []
    limited = _make(client, 1.0, clock, sleeps)  # 1.0s interval

    limited.system_one({}, _QUESTIONS)
    clock.advance(0.4)
    limited.system_one({}, _QUESTIONS)

    assert sleeps == [0.6]


def test_enforces_spacing_across_many_calls():
    client = _CountingJevClient()
    clock = _FakeClock()
    sleeps: list[float] = []
    limited = _make(client, 4.0, clock, sleeps)  # 0.25s interval

    for _ in range(5):
        limited.system_one({}, _QUESTIONS)

    assert client.calls == 5
    assert sleeps == [0.25, 0.25, 0.25, 0.25]


@pytest.mark.parametrize("bad_rate", [0, -1.0])
def test_rejects_a_non_positive_rate(bad_rate):
    with pytest.raises(ValueError):
        RateLimitedJevClient(_CountingJevClient(), max_calls_per_second=bad_rate)
