# Turn-rate budget for the tactical action-selection loop

Resolves the turn-rate-budget flag on
[issue #23](https://github.com/zbloss/jev-plays-pokemon/issues/23) (child of
the MVP spec [issue #14](https://github.com/zbloss/jev-plays-pokemon/issues/14),
User Story 17): the loop's per-turn budget must be documented against a
latency figure re-benchmarked from the network path the loop actually runs on,
not the ~230–300 ms dev-machine figure `docs/research/typesafe-sdk-integration.md`
left on record (against the SDK's ~100 ms server-side claim).

## How to re-measure

The benchmark lives at `src/jev_plays_pokemon/benchmark.py` and runs the real
per-turn cycle — RAM state extraction + a real `system_one()` Choice + real
action execution against a booted `pokemon_red.gb` — N times and reports
wall-clock percentiles:

```
uv run python -m jev_plays_pokemon.benchmark --turns 25 --warmup 5
```

Add `--no-dialog` to exclude the vision-fallback dialog decode (a separate,
independently-configured OpenAI-compatible endpoint, not a TypeSafe call).
`--warmup` calls are discarded, so a cold TLS connection / first-call cost
does not skew the steady-state figures. **Run it on whatever host actually
runs the loop** — the whole point of this re-benchmark is that the figure is
host/network-specific, not machine-independent.

## Measured on the current run host (2026-09-18)

The figures below were re-benchmarked from this checkout's host (the Windows
dev box this loop was developed and last executed on), which is the host the
loop runs on today. They supersede the dev-machine figures in the research doc.

| Path | n | min | p50 | p95 | max |
| --- | --- | --- | --- | --- | --- |
| Full turn, real `system_one()` + real ROM | 12 | 136 ms | 160 ms | 261 ms | 261 ms |
| Full turn, `--no-dialog` | 25 | 135 ms | 213 ms | 266 ms | 508 ms |
| Local turn, stub Jev client (real PyBoy, **no network**) | 25 | 1.0 ms | 1.9 ms | 2.2 ms | 2.2 ms |

Two reading caveats. The n=12 row's p95 equals its max because the
nearest-rank percentile rounds to the top sample at small `n` — at n=12 there
are only 12 samples, so its "p95" is really its single worst turn. The n=25
row's p95 (266 ms) is the meaningful steady-state figure. And both "Full turn"
rows were measured walking around the bedroom, where no dialog box ever opens
(and the vision backend was unconfigured anyway), so they measured the same
Jev+execute path — the ~50 ms p50 spread between them is network jitter, not a
dialog effect.

**The per-turn cost is almost entirely the TypeSafe API round trip.** State
extraction + milestone tracking + payload serialization + executing one chosen
button press against a live PyBoy instance is ~2 ms (p95) — effectively noise
next to the API call. That means the loop's throughput ceiling is set by
network latency to the TypeSafe endpoint, and PyBoy adds no meaningful budget
pressure.

## What the budget does NOT cover

The measured turn is the Jev decision + action execution (the stream-surface
update is an in-memory dict write with a timestamp — sub-millisecond,
deliberately excluded). The **vision-fallback dialog decode is a separate,
independently-configured cost** (a call to a self-hosted or OpenAI vision/OCR
endpoint, not the TypeSafe API) that fires only on turns where a dialog box is
open. It is not in the table above and can dwarf the 300 ms Jev budget when it
runs; benchmark that endpoint on its own path if dialog-heavy turns need their
own budget. #19 keeps it out of the Jev loop's per-turn latency by design.

## Budget

- **Steady-state budget: ~300 ms per turn.** The n=25 live sample puts p95 at
  ~266 ms; rounding to 300 ms gives headroom over the p95 for normal jitter.
- **Budgeted turn rate: ~3.3 turns/s** (1 turn / 300 ms). Observed p95 supports
  ~3.7–3.8 turns/s, so 3.3 is conservative.
- **Worst-case headroom:** a single turn was observed at ~508 ms. A 300 ms
  steady-state budget with the loop running off a wall-clock tick (rather than
  a fixed high rate) absorbs a turn like that without stalling the stream; the
  loop never blocks on Jev's confidence, so a slow turn delays only that turn.
- This comfortably clears the SDK's ~100 ms server-side figure once network/TLS
  is included, and lands in the ~150–350 ms end-to-end band the research doc's
  §6 recommended budgeting for.

## Caveat — this is the current host, not a separate production host

The benchmark was re-run from the machine the loop currently runs on, which is
more current than the recorded research figure but is still this dev box: no
separate production deployment host was available at re-benchmark time. When
the loop is deployed to its final streaming host, re-run the exact command
above there and update this page — that host's network distance to the TypeSafe
API is the figure the live stream's feel actually depends on. The tooling and
procedure to do so are in place and validated by this run.

## Sources

- Re-benchmark: `uv run python -m jev_plays_pokemon.benchmark`, run on this
  host 2026-09-18 against the live TypeSafe API using this repo's `.env`
  `TYPESAFE_API_KEY` (value never printed or logged; the benchmark prints only
  latency percentiles).
- Prior dev-machine figure and the ~100 ms documented figure:
  `docs/research/typesafe-sdk-integration.md` §5, §6.
