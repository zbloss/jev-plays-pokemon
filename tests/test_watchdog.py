import os
import sys

from jev_plays_pokemon.watchdog import (
    run_watchdog,
    spawn_run_subprocess,
    touch_heartbeat,
)


def _set_mtime(path, when: float) -> None:
    path.write_text("")
    os.utime(path, (when, when))


def test_touch_heartbeat_creates_the_file_if_missing(tmp_path):
    path = tmp_path / "heartbeat.txt"
    assert not path.exists()

    touch_heartbeat(path)

    assert path.exists()


def test_touch_heartbeat_updates_the_mtime_of_an_existing_file(tmp_path):
    path = tmp_path / "heartbeat.txt"
    _set_mtime(path, 100.0)

    touch_heartbeat(path)

    assert path.stat().st_mtime > 100.0


class _FakeProcess:
    """A scripted stand-in for `subprocess.Popen` - `poll_results` is
    consumed one value per `poll()` call; `None` means "still running"."""

    def __init__(self, poll_results: list[int | None]) -> None:
        self._poll_results = list(poll_results)
        self.terminated = False
        self.killed = False
        self.wait_calls: list[float | None] = []

    def poll(self) -> int | None:
        if not self._poll_results:
            return None
        return self._poll_results.pop(0)

    def terminate(self) -> None:
        self.terminated = True

    def kill(self) -> None:
        self.killed = True

    def wait(self, timeout: float | None = None) -> int:
        self.wait_calls.append(timeout)
        return 0


def test_restarts_the_subprocess_on_a_nonzero_exit(tmp_path):
    heartbeat_path = tmp_path / "heartbeat.txt"
    processes = [_FakeProcess([1]), _FakeProcess([1])]
    spawned: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        process = processes[len(spawned)]
        spawned.append(process)
        return process

    run_watchdog(
        spawn,
        heartbeat_path=heartbeat_path,
        time_source=lambda: 0.0,
        sleep=lambda seconds: None,
        max_restarts=2,
    )

    assert len(spawned) == 2
    for process in spawned:
        # Exited on its own - the watchdog has no reason to terminate/kill it.
        assert process.terminated is False
        assert process.killed is False


def test_restarts_the_subprocess_when_the_heartbeat_goes_stale(tmp_path):
    heartbeat_path = tmp_path / "heartbeat.txt"
    _set_mtime(heartbeat_path, 0.0)  # already stale relative to the clock below
    clock = {"now": 0.0}

    def time_source() -> float:
        clock["now"] += 200.0  # jumps well past the timeout on every check
        return clock["now"]

    processes = [_FakeProcess([None] * 10), _FakeProcess([1])]
    spawned: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        process = processes[len(spawned)]
        spawned.append(process)
        return process

    run_watchdog(
        spawn,
        heartbeat_path=heartbeat_path,
        heartbeat_timeout_seconds=120.0,
        time_source=time_source,
        sleep=lambda seconds: None,
        max_restarts=2,
    )

    assert len(spawned) == 2
    assert spawned[0].terminated is True
    assert spawned[0].wait_calls == [5.0]  # default poll_interval_seconds


def test_a_fresh_spawn_gets_a_grace_period_before_a_missing_heartbeat_is_stale(
    tmp_path,
):
    # No heartbeat file exists at all yet - a brand new subprocess that
    # hasn't had time to write its first one isn't hung.
    heartbeat_path = tmp_path / "never-written.txt"
    clock = {"now": 0.0}

    def time_source() -> float:
        clock["now"] += 1.0  # only 1s "passes" per check
        return clock["now"]

    spawned: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        process = _FakeProcess([None] * 1000)
        spawned.append(process)
        return process

    run_watchdog(
        spawn,
        heartbeat_path=heartbeat_path,
        heartbeat_timeout_seconds=120.0,
        time_source=time_source,
        sleep=lambda seconds: None,
        max_polls=20,  # well under the 120s grace period
    )

    assert len(spawned) == 1
    assert spawned[0].terminated is False


def test_a_stale_heartbeat_left_over_from_a_previous_run_does_not_trigger_an_immediate_restart(
    tmp_path,
):
    # The file exists but its mtime is old (e.g. left over from a run the
    # watchdog already restarted once) - the fresh subprocess still gets
    # its own grace period rather than being killed on its very first poll.
    heartbeat_path = tmp_path / "heartbeat.txt"
    _set_mtime(heartbeat_path, -10_000.0)
    clock = {"now": 0.0}

    def time_source() -> float:
        clock["now"] += 1.0
        return clock["now"]

    spawned: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        process = _FakeProcess([None] * 1000)
        spawned.append(process)
        return process

    run_watchdog(
        spawn,
        heartbeat_path=heartbeat_path,
        heartbeat_timeout_seconds=120.0,
        time_source=time_source,
        sleep=lambda seconds: None,
        max_polls=20,
    )

    assert len(spawned) == 1
    assert spawned[0].terminated is False


def test_terminate_timing_out_falls_back_to_kill(tmp_path):
    heartbeat_path = tmp_path / "heartbeat.txt"
    _set_mtime(heartbeat_path, 0.0)

    class _NeverExitsOnTerminate(_FakeProcess):
        def wait(self, timeout: float | None = None) -> int:
            import subprocess

            raise subprocess.TimeoutExpired(cmd="fake", timeout=timeout or 0.0)

    stuck_process = _NeverExitsOnTerminate([None] * 10)
    second_process = _FakeProcess([1])
    processes = [stuck_process, second_process]
    spawned: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        process = processes[len(spawned)]
        spawned.append(process)
        return process

    clock = {"now": 0.0}

    def time_source() -> float:
        clock["now"] += 1000.0  # always past the (already-stale) heartbeat
        return clock["now"]

    run_watchdog(
        spawn,
        heartbeat_path=heartbeat_path,
        heartbeat_timeout_seconds=0.0,  # stale as soon as any time passes
        time_source=time_source,
        sleep=lambda seconds: None,
        max_restarts=2,
    )

    assert stuck_process.terminated is True
    assert stuck_process.killed is True


def test_does_not_restart_while_the_process_is_running_and_the_heartbeat_is_fresh(
    tmp_path,
):
    heartbeat_path = tmp_path / "heartbeat.txt"
    clock = {"now": 0.0}

    def time_source() -> float:
        clock["now"] += 1.0
        return clock["now"]

    _set_mtime(heartbeat_path, 0.0)
    spawned: list[_FakeProcess] = []

    def spawn() -> _FakeProcess:
        process = _FakeProcess([None] * 1000)  # never exits on its own
        spawned.append(process)
        return process

    run_watchdog(
        spawn,
        heartbeat_path=heartbeat_path,
        heartbeat_timeout_seconds=120.0,
        time_source=time_source,
        sleep=lambda seconds: None,
        max_polls=20,  # 20s of simulated time, well under the 120s timeout
    )

    assert len(spawned) == 1
    assert spawned[0].terminated is False


# -- spawn_run_subprocess: the ADR 0001 CLI-forwarding seam --


def _fake_popen(monkeypatch):
    calls: list[list[str]] = []

    def fake(args: list[str]) -> object:
        calls.append(args)
        return object()

    monkeypatch.setattr("jev_plays_pokemon.watchdog.subprocess.Popen", fake)
    return calls


def test_spawn_run_subprocess_targets_the_cli_run_subcommand(monkeypatch):
    calls = _fake_popen(monkeypatch)

    spawn_run_subprocess(
        rom_path=None,
        stream_port=8080,
        max_calls_per_second=None,
    )

    assert calls == [
        [sys.executable, "-m", "jev_plays_pokemon.cli", "run", "--stream-port", "8080"]
    ]


def test_spawn_run_subprocess_forwards_the_rom_path_and_max_calls_per_second_when_given(
    monkeypatch,
):
    calls = _fake_popen(monkeypatch)

    spawn_run_subprocess(
        rom_path="custom.gb",
        stream_port=9090,
        max_calls_per_second=0.5,
    )

    assert calls == [
        [
            sys.executable,
            "-m",
            "jev_plays_pokemon.cli",
            "run",
            "--rom-path",
            "custom.gb",
            "--stream-port",
            "9090",
            "--max-calls-per-second",
            "0.5",
        ]
    ]


def test_spawn_run_subprocess_forwards_display_fps_when_given(monkeypatch):
    calls = _fake_popen(monkeypatch)

    spawn_run_subprocess(
        rom_path=None,
        stream_port=8080,
        max_calls_per_second=None,
        display_fps=30.0,
    )

    assert calls == [
        [
            sys.executable,
            "-m",
            "jev_plays_pokemon.cli",
            "run",
            "--stream-port",
            "8080",
            "--display-fps",
            "30.0",
        ]
    ]


def test_spawn_run_subprocess_forwards_only_the_given_global_settings(monkeypatch):
    calls = _fake_popen(monkeypatch)

    spawn_run_subprocess(
        rom_path=None,
        stream_port=8080,
        max_calls_per_second=None,
        openai_api_key="sk-test",
        typesafe_api_key="tsk-test",
    )

    assert calls == [
        [
            sys.executable,
            "-m",
            "jev_plays_pokemon.cli",
            "--openai-api-key",
            "sk-test",
            "--typesafe-api-key",
            "tsk-test",
            "run",
            "--stream-port",
            "8080",
        ]
    ]
