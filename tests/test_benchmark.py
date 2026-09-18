import pytest

from jev_plays_pokemon.benchmark import (
    TurnLatencyStats,
    _format_stats,
    _percentile,
    measure_turn_latencies,
)


class _FakeClock:
    """Hands back a fixed sequence of seconds, one per call - so measured
    turn durations are exact, no real time involved."""

    def __init__(self, times: list[float]) -> None:
        self._times = list(times)
        self.calls = 0

    def __call__(self) -> float:
        value = self._times[self.calls]
        self.calls += 1
        return value


def _counting_turn(counter: dict):
    def turn():
        counter["calls"] += 1

    return turn


def test_measure_turn_latencies_reports_percentiles_of_measured_turns():
    # Two clock reads (start, end) per measured turn -> durations of
    # 100/300/200/500ms, i.e. sorted 100/200/300/500.
    clock = _FakeClock([0.0, 0.1, 0.1, 0.4, 0.4, 0.6, 0.6, 1.1])
    counter = {"calls": 0}

    stats = measure_turn_latencies(
        _counting_turn(counter), turns=4, warmup=0, clock=clock
    )

    assert stats.count == 4
    assert stats.min_ms == pytest.approx(100.0)
    assert stats.p50_ms == pytest.approx(250.0)
    assert stats.p95_ms == pytest.approx(500.0)
    assert stats.max_ms == pytest.approx(500.0)
    assert stats.turns_per_second_at_p95 == pytest.approx(2.0)


def test_measure_turn_latencies_runs_but_discards_warmup_turns():
    clock = _FakeClock([5.0, 5.05])  # only the one measured turn is timed
    counter = {"calls": 0}

    stats = measure_turn_latencies(
        _counting_turn(counter), turns=1, warmup=2, clock=clock
    )

    # warmup + measured calls all hit the turn, but only `turns` are timed.
    assert counter["calls"] == 3
    assert clock.calls == 2
    assert stats.count == 1
    assert stats.min_ms == pytest.approx(50.0)


def test_measure_turn_latencies_rejects_a_nonpositive_turn_count():
    with pytest.raises(ValueError):
        measure_turn_latencies(lambda: None, turns=0)


@pytest.mark.parametrize(
    ("fraction", "expected"),
    [(0.01, 10.0), (0.5, 30.0), (0.95, 50.0), (1.0, 50.0)],
)
def test_percentile_uses_nearest_rank(fraction: float, expected: float):
    assert _percentile([10.0, 20.0, 30.0, 40.0, 50.0], fraction) == expected


def test_format_stats_mentions_the_p95_and_turn_rate():
    line = _format_stats(
        TurnLatencyStats(count=20, min_ms=100, p50_ms=250, p95_ms=300, max_ms=500)
    )
    assert "p95=300ms" in line
    assert "turns/s" in line
