"""Unified `jev-plays-pokemon` CLI (ADR 0001, docs/adr/0001-...).

Three subcommands wrap this project's three existing entry points -
`run` (the live tactical loop, `main.main`; also the default when no
subcommand is given, for backward compatibility with the pre-Typer CLI),
`benchmark` (`benchmark.run_benchmark`), and `watchdog`
(`watchdog.run_watchdog`) - sharing one settings surface instead of each
parsing its own overlapping, inconsistently-named flags.

The five values `settings.Settings` covers (`OPENAI_API_KEY`,
`OPENAI_BASE_URL`, `OPENAI_VISION_MODEL`, `DIALOG_DECODE_BACKEND`,
`TYPESAFE_API_KEY`) are declared once, here, as options on the top-level
`app` callback - given *before* the subcommand name on the command line,
e.g. ``jev-plays-pokemon --openai-api-key=... run`` - rather than
duplicated per subcommand, since `run`/`benchmark`/`watchdog` all resolve
them the same way. Everything else (`--rom-path`, `--stream-port`, ...)
stays a plain per-subcommand option with no `.env` backing, matching the
scope decided in ADR 0001.

Heavy, PyBoy-/TypeSafe-dependent imports (`main`, `benchmark`) are deferred
into their own subcommand functions rather than imported at module level, so
that ``jev-plays-pokemon watchdog`` - which only spawns and monitors a
*subprocess* running the rest of this CLI - never needs them to import
successfully, preserving `watchdog.py`'s own "standalone and
dependency-free" guarantee one level up (see its docstring).
"""

from __future__ import annotations

from pathlib import Path

import typer

from jev_plays_pokemon.settings import Settings, load_settings
from jev_plays_pokemon.watchdog import (
    DEFAULT_HEARTBEAT_PATH,
    DEFAULT_HEARTBEAT_TIMEOUT_SECONDS,
    DEFAULT_POLL_INTERVAL_SECONDS,
    run_watchdog,
    spawn_run_subprocess,
)

app = typer.Typer(
    help="Jev plays Pokemon: boot the ROM and let Jev make the tactical calls.",
)

_DEFAULT_STREAM_PORT = 8080
_MAX_CALLS_PER_SECOND_HELP = (
    "Debug mode: cap TypeSafe API calls/sec so a human can watch decisions "
    "without burning many real calls (e.g. 0.5 = one call every 2s). "
    "Unset = no limit (production)."
)
_STREAM_PORT_HELP = (
    "Port for the read-only stream surface (see "
    "stream_surface.start_stream_surface_server). Default 8080."
)
_ROM_PATH_HELP = "Passed through to emulator.boot_or_resume."


def _run(
    settings: Settings,
    *,
    rom_path: str | None,
    stream_port: int,
    max_calls_per_second: float | None,
) -> None:
    from jev_plays_pokemon.main import main as run_main

    run_main(
        rom_path,
        stream_port=stream_port,
        max_calls_per_second=max_calls_per_second,
        settings=settings,
    )


@app.callback(invoke_without_command=True)
def main_callback(
    ctx: typer.Context,
    openai_api_key: str | None = typer.Option(
        None, "--openai-api-key", help="Overrides OPENAI_API_KEY."
    ),
    openai_base_url: str | None = typer.Option(
        None, "--openai-base-url", help="Overrides OPENAI_BASE_URL."
    ),
    openai_vision_model: str | None = typer.Option(
        None, "--openai-vision-model", help="Overrides OPENAI_VISION_MODEL."
    ),
    dialog_decode_backend: str | None = typer.Option(
        None,
        "--dialog-decode-backend",
        help="Overrides DIALOG_DECODE_BACKEND ('vision-llm' or 'rapidocr').",
    ),
    typesafe_api_key: str | None = typer.Option(
        None, "--typesafe-api-key", help="Overrides TYPESAFE_API_KEY."
    ),
    env_file: Path = typer.Option(
        Path(".env"),
        "--env-file",
        help="`.env` file to load these values from when a flag/env var isn't set.",
    ),
    rom_path: str | None = typer.Option(None, "--rom-path", help=_ROM_PATH_HELP),
    stream_port: int = typer.Option(
        _DEFAULT_STREAM_PORT, "--stream-port", help=_STREAM_PORT_HELP
    ),
    max_calls_per_second: float | None = typer.Option(
        None, "--max-calls-per-second", help=_MAX_CALLS_PER_SECOND_HELP
    ),
) -> None:
    """Jev plays Pokemon. With no subcommand, runs the live tactical loop (same as `run`)."""
    settings = load_settings(
        env_file,
        openai_api_key=openai_api_key,
        openai_base_url=openai_base_url,
        openai_vision_model=openai_vision_model,
        dialog_decode_backend=dialog_decode_backend,
        typesafe_api_key=typesafe_api_key,
    )
    ctx.obj = settings
    if ctx.invoked_subcommand is None:
        _run(
            settings,
            rom_path=rom_path,
            stream_port=stream_port,
            max_calls_per_second=max_calls_per_second,
        )


@app.command("run")
def run_command(
    ctx: typer.Context,
    rom_path: str | None = typer.Option(None, "--rom-path", help=_ROM_PATH_HELP),
    stream_port: int = typer.Option(
        _DEFAULT_STREAM_PORT, "--stream-port", help=_STREAM_PORT_HELP
    ),
    max_calls_per_second: float | None = typer.Option(
        None, "--max-calls-per-second", help=_MAX_CALLS_PER_SECOND_HELP
    ),
) -> None:
    """Boot the ROM and run the live tactical loop until interrupted."""
    _run(
        ctx.obj,
        rom_path=rom_path,
        stream_port=stream_port,
        max_calls_per_second=max_calls_per_second,
    )


@app.command("benchmark")
def benchmark_command(
    ctx: typer.Context,
    rom_path: str | None = typer.Option(
        None, "--rom-path", help="ROM to boot for the benchmark."
    ),
    turns: int = typer.Option(20, "--turns", help="Number of measured turns."),
    warmup: int = typer.Option(
        3, "--warmup", help="Warmup turns run first and discarded before measuring."
    ),
    no_dialog: bool = typer.Option(
        False,
        "--no-dialog",
        help="Skip the vision-fallback dialog decode even if it is configured.",
    ),
) -> None:
    """Measure real per-turn latency and print percentiles (see docs/turn-rate-budget.md)."""
    from jev_plays_pokemon.benchmark import run_benchmark

    run_benchmark(
        turns=turns,
        warmup=warmup,
        rom_path=rom_path,
        no_dialog=no_dialog,
        settings=ctx.obj,
    )


@app.command("watchdog")
def watchdog_command(
    ctx: typer.Context,
    rom_path: str | None = typer.Option(
        None,
        "--rom-path",
        help=f"{_ROM_PATH_HELP} Forwarded to the spawned `run` process.",
    ),
    stream_port: int = typer.Option(
        _DEFAULT_STREAM_PORT,
        "--stream-port",
        help=f"{_STREAM_PORT_HELP} Forwarded to the spawned `run` process.",
    ),
    max_calls_per_second: float | None = typer.Option(
        None,
        "--max-calls-per-second",
        help=f"{_MAX_CALLS_PER_SECOND_HELP} Forwarded to the spawned `run` process.",
    ),
    heartbeat_path: Path = typer.Option(DEFAULT_HEARTBEAT_PATH, "--heartbeat-path"),
    heartbeat_timeout_seconds: float = typer.Option(
        DEFAULT_HEARTBEAT_TIMEOUT_SECONDS, "--heartbeat-timeout-seconds"
    ),
    poll_interval_seconds: float = typer.Option(
        DEFAULT_POLL_INTERVAL_SECONDS, "--poll-interval-seconds"
    ),
) -> None:
    """Run `run` as a monitored subprocess, restarting it on a crash or a hang."""
    settings: Settings = ctx.obj

    run_watchdog(
        lambda: spawn_run_subprocess(
            rom_path=rom_path,
            stream_port=stream_port,
            max_calls_per_second=max_calls_per_second,
            openai_api_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
            openai_vision_model=settings.openai_vision_model,
            dialog_decode_backend=settings.dialog_decode_backend,
            typesafe_api_key=settings.typesafe_api_key,
        ),
        heartbeat_path=heartbeat_path,
        heartbeat_timeout_seconds=heartbeat_timeout_seconds,
        poll_interval_seconds=poll_interval_seconds,
    )


if __name__ == "__main__":
    app()
