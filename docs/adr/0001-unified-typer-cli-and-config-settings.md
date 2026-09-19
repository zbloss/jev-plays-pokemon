# Unified Typer CLI with pydantic-settings-backed config, including secrets as CLI flags

Status: accepted

Before this decision, the project had three independent argparse-based entry points (`main.py`'s `cli()`, `benchmark.py`, `watchdog.py`) with inconsistently-named overlapping flags (a ROM path spelled as a positional arg, `--rom`, and `--rom-path` in the three respective scripts), and four config values (`OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_VISION_MODEL`, `DIALOG_DECODE_BACKEND`) read ad hoc via `os.environ.get(...)` with no central settings model or `.env` loading.

We unified all three into one Typer app (`jev_plays_pokemon.cli:app`, registered as the `jev-plays-pokemon` script) with `run`/`benchmark`/`watchdog` subcommands — bare `jev-plays-pokemon` still runs the game by default, for backward compatibility. ROM path is now `--rom-path` everywhere. `watchdog` forwards its full resolved settings to the `run` subprocess it spawns, not just the ROM path as before. The `python -m jev_plays_pokemon.benchmark` module invocation is dropped in favor of `jev-plays-pokemon benchmark`.

Config is now backed by a `pydantic-settings` `Settings` model (`env_file=".env"`), covering the four existing env-driven values plus `TYPESAFE_API_KEY` (previously resolved entirely inside `typesafe_sdk`'s `TypeSafeClient`/`Config.resolve`, with no override point in our own code — `build_jev_client()` now accepts an `api_key` override instead of calling `TypeSafeClient()` bare). All five are exposed as top-level global CLI flags shared by all three subcommands, with precedence CLI flag > env var/`.env` > existing hardcoded default or error. A top-level `--env-file PATH` flag overrides which `.env` file is loaded. Env var names are kept exactly as-is, unprefixed — `OPENAI_API_KEY`/`OPENAI_BASE_URL` intentionally match the standard `openai` SDK's own auto-detected env vars.

## Deliberate deviation: secrets as CLI flags

`OPENAI_API_KEY` and `TYPESAFE_API_KEY` are deliberately exposed as plaintext CLI flags (`--openai-api-key`, `--typesafe-api-key`) alongside the non-secret values, despite the usual guidance against putting secrets on a command line (shell history, process listings). This was an explicit choice, made after the design session initially excluded them, then reversed for override consistency across all five config values. Do not "fix" this by quietly removing the flags — if the exposure risk becomes a problem, revisit this ADR rather than silently diverging from it.

## Consequences

- `decision.py::build_jev_client()`'s signature changes (accepts `api_key: str | None = None`).
- `docs/turn-rate-budget.md`'s documented `python -m jev_plays_pokemon.benchmark` invocation needs updating to `jev-plays-pokemon benchmark`.
- `--stream-port`, `--max-calls-per-second`, benchmark's `--turns`/`--warmup`/`--no-dialog`, and watchdog's heartbeat/poll-interval flags remain CLI-only (no `.env` backing) — deliberately out of scope for the settings model.
- `TYPESAFE_BASE_URL`/`TYPESAFE_DEFAULT_MODEL`/`TYPESAFE_LOG_LEVEL` (supported by `typesafe_sdk` but unused anywhere in this repo today) remain out of scope.
