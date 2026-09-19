"""Re-benchmark the tactical loop's per-turn latency from where it actually runs.

Issue #23's third acceptance criterion: the turn-rate budget has to be
documented against a latency figure measured from the *production network
path the loop runs on*, not the ~230-300ms warm per-call figure that
`docs/research/typesafe-sdk-integration.md` recorded from a dev machine
(the SDK docs cite ~100ms server-side). That gap is plausibly network/TLS/
geography to the TypeSafe API rather than server inference time, so it has to
be re-measured wherever the loop is actually deployed - this module is the
tool for that.

Run it on the deployment host so the number includes that host's real network
round trip to the TypeSafe API and this machine's real PyBoy throughput:

    uv run jev-plays-pokemon benchmark --turns 20

`measure_turn_latencies` is the pure, testable core: it times an injected
`turn` callable and reports wall-clock percentiles, with a throwaway warmup
so a cold connection/first-call cost doesn't skew the steady-state figure.
`run_benchmark` wires the real per-turn cycle (RAM state extraction + a real
`system_one()` Choice + real action execution against a booted ROM) - it's the
`jev-plays-pokemon benchmark` subcommand's implementation (see `cli.py`, ADR
0001) - and is not unit-tested - consistent with the rest of the repo, no
test here depends on the real TypeSafe API.
"""

from __future__ import annotations

import logging
import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from jev_plays_pokemon.settings import Settings

Clock = Callable[[], float]


@dataclass(frozen=True)
class TurnLatencyStats:
    """Wall-clock per-turn latency, in milliseconds.

    `p95_ms` is the number to size the turn-rate budget around: worst realistic
    turns, not just the median, are what make a live stream feel sluggish.
    """

    count: int
    min_ms: float
    p50_ms: float
    p95_ms: float
    max_ms: float

    @property
    def turns_per_second_at_p95(self) -> float:
        return 1000.0 / self.p95_ms if self.p95_ms else 0.0


def _percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted non-empty sequence."""
    rank = -(-fraction * len(sorted_values) // 1)  # ceil(fraction * n)
    index = min(max(int(rank) - 1, 0), len(sorted_values) - 1)
    return sorted_values[index]


def measure_turn_latencies(
    turn: Callable[[], None],
    *,
    turns: int = 20,
    warmup: int = 3,
    clock: Clock = time.perf_counter,
) -> TurnLatencyStats:
    """Time `turn()` over `turns` measured calls and report latency statistics.

    `warmup` calls run first and are discarded (so a cold TLS connection or a
    first-call cache miss doesn't land in the steady-state numbers). Each
    measured turn is timed with `clock`, which is injected so the percentiles
    can be tested deterministically without real work or real time.
    """
    if turns < 1:
        raise ValueError("turns must be at least 1")

    for _ in range(warmup):
        turn()

    durations_ms: list[float] = []
    for _ in range(turns):
        start = clock()
        turn()
        durations_ms.append((clock() - start) * 1000.0)

    durations_ms.sort()
    return TurnLatencyStats(
        count=len(durations_ms),
        min_ms=durations_ms[0],
        p50_ms=statistics.median(durations_ms),
        p95_ms=_percentile(durations_ms, 0.95),
        max_ms=durations_ms[-1],
    )


def _format_stats(stats: TurnLatencyStats) -> str:
    return (
        f"turn latency over {stats.count} turns: "
        f"min={stats.min_ms:.0f}ms "
        f"p50={stats.p50_ms:.0f}ms "
        f"p95={stats.p95_ms:.0f}ms "
        f"max={stats.max_ms:.0f}ms "
        f"(~{stats.turns_per_second_at_p95:.2f} turns/s at p95)"
    )


def run_benchmark(
    *,
    turns: int = 20,
    warmup: int = 3,
    rom_path: str | None = None,
    no_dialog: bool = False,
    settings: Settings | None = None,
) -> None:
    """Run the real per-turn cycle `turns` times and print latency percentiles.

    Run this from the environment the loop actually ships to; the printed p95
    is the figure to record in `docs/turn-rate-budget.md`. `settings` (see
    `settings.Settings`, ADR 0001) supplies the CLI-/`.env`-resolved
    `TYPESAFE_API_KEY`/`OPENAI_*`/`DIALOG_DECODE_BACKEND` overrides; left as
    `None`, a fresh `Settings()` is resolved from the environment/`.env` alone.
    """
    from jev_plays_pokemon import emulator
    from jev_plays_pokemon.decision import (
        build_jev_client,
        make_pyboy_action_executor,
        run_turn,
    )
    from jev_plays_pokemon.dialog_decode import load_dialog_decoder_or_none
    from jev_plays_pokemon.game_state import extract_game_state
    from jev_plays_pokemon.main import build_dialog_text_source

    logging.basicConfig(level=logging.WARNING)
    settings = settings or Settings()

    pyboy = emulator.boot_to_controllable_state(rom_path or emulator.DEFAULT_ROM_PATH)
    jev_client = build_jev_client(api_key=settings.typesafe_api_key)
    execute_action = make_pyboy_action_executor(pyboy)
    decoder = (
        None
        if no_dialog
        else load_dialog_decoder_or_none(
            backend=settings.dialog_decode_backend,
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            model=settings.openai_vision_model,
        )
    )
    dialog_text_source = build_dialog_text_source(pyboy, decoder)

    def turn() -> None:
        run_turn(
            lambda: extract_game_state(pyboy),
            jev_client,
            execute_action,
            on_decision=lambda decision: None,
            dialog_text_source=dialog_text_source,
        )

    try:
        stats = measure_turn_latencies(turn, turns=turns, warmup=warmup)
        print(_format_stats(stats))
    finally:
        pyboy.stop(save=False)
