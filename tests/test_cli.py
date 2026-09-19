"""Tests for the unified Typer CLI (ADR 0001).

Every invocation passes `--env-file` at a nonexistent tmp path so these
tests never pick up the real repo-root `.env` (which sets `TYPESAFE_API_KEY`
for real use) - the resolved `Settings` in each assertion should reflect
only what the test itself set via CLI flags/env vars.
"""

from typer.testing import CliRunner

from jev_plays_pokemon.cli import app
from jev_plays_pokemon.settings import Settings

runner = CliRunner()


def _no_env_file(tmp_path):
    return str(tmp_path / "does-not-exist.env")


def test_bare_invocation_runs_the_game_with_defaults(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "jev_plays_pokemon.main.main", lambda *a, **kw: calls.append((a, kw))
    )

    result = runner.invoke(app, ["--env-file", _no_env_file(tmp_path)])

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    args, kwargs = calls[0]
    assert args == (None,)
    assert kwargs["stream_port"] == 8080
    assert kwargs["max_calls_per_second"] is None
    # Isolated from the real repo-root `.env` via `--env-file`, so every
    # field should be unset.
    assert kwargs["settings"] == Settings(
        openai_api_key=None,
        openai_base_url=None,
        openai_vision_model=None,
        dialog_decode_backend=None,
        typesafe_api_key=None,
    )


def test_bare_invocation_forwards_rom_path_and_global_overrides(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "jev_plays_pokemon.main.main", lambda *a, **kw: calls.append((a, kw))
    )

    result = runner.invoke(
        app,
        [
            "--env-file",
            _no_env_file(tmp_path),
            "--openai-api-key",
            "sk-cli",
            "--rom-path",
            "x.gb",
        ],
    )

    assert result.exit_code == 0, result.output
    args, kwargs = calls[0]
    assert args == ("x.gb",)
    assert kwargs["settings"].openai_api_key == "sk-cli"


def test_run_subcommand_forwards_its_own_options(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "jev_plays_pokemon.main.main", lambda *a, **kw: calls.append((a, kw))
    )

    result = runner.invoke(
        app,
        [
            "--env-file",
            _no_env_file(tmp_path),
            "--typesafe-api-key",
            "tsk-cli",
            "run",
            "--stream-port",
            "9999",
            "--max-calls-per-second",
            "0.5",
        ],
    )

    assert result.exit_code == 0, result.output
    args, kwargs = calls[0]
    assert args == (None,)
    assert kwargs["stream_port"] == 9999
    assert kwargs["max_calls_per_second"] == 0.5
    assert kwargs["settings"].typesafe_api_key == "tsk-cli"


def test_benchmark_subcommand_forwards_its_options_and_settings(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(
        "jev_plays_pokemon.benchmark.run_benchmark",
        lambda **kw: calls.append(kw),
    )

    result = runner.invoke(
        app,
        [
            "--env-file",
            _no_env_file(tmp_path),
            "--dialog-decode-backend",
            "rapidocr",
            "benchmark",
            "--turns",
            "5",
            "--warmup",
            "1",
            "--no-dialog",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    kwargs = calls[0]
    assert kwargs["turns"] == 5
    assert kwargs["warmup"] == 1
    assert kwargs["no_dialog"] is True
    assert kwargs["settings"].dialog_decode_backend == "rapidocr"


def test_watchdog_subcommand_builds_a_spawn_seam_and_runs_it(monkeypatch, tmp_path):
    watchdog_calls = []
    monkeypatch.setattr(
        "jev_plays_pokemon.cli.run_watchdog",
        lambda spawn, **kw: watchdog_calls.append((spawn, kw)),
    )
    spawn_calls = []
    monkeypatch.setattr(
        "jev_plays_pokemon.cli.spawn_run_subprocess",
        lambda **kw: spawn_calls.append(kw),
    )

    result = runner.invoke(
        app,
        [
            "--env-file",
            _no_env_file(tmp_path),
            "--openai-api-key",
            "sk-cli",
            "watchdog",
            "--rom-path",
            "x.gb",
            "--poll-interval-seconds",
            "1",
        ],
    )

    assert result.exit_code == 0, result.output
    assert len(watchdog_calls) == 1
    spawn, watchdog_kwargs = watchdog_calls[0]
    assert watchdog_kwargs["poll_interval_seconds"] == 1.0

    # run_watchdog never called `spawn()` itself (it's mocked) - invoke it
    # here to check it was built with the right forwarded values.
    spawn()
    assert spawn_calls == [
        {
            "rom_path": "x.gb",
            "stream_port": 8080,
            "max_calls_per_second": None,
            "openai_api_key": "sk-cli",
            "openai_base_url": None,
            "openai_vision_model": None,
            "dialog_decode_backend": None,
            "typesafe_api_key": None,
        }
    ]
