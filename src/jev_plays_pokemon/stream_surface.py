"""Stream-facing read-only surface: OBS (or any local viewer) polls Jev's live
decision state without codebase access, and can never interfere with it.

Implements #22, part of #14's MVP tactical action-selection loop: a minimal,
read-only view of the decision core's (#21) latest logged decision - chosen
action, confidence, and the completed/current/future objective split (#18) -
over a local HTTP JSON endpoint, so a streaming overlay can follow along.

The wire shape a viewer sees is declared once as ``Snapshot`` below (pydantic
was already in the dependency tree via the vision-fallback path's ``openai``
SDK, and is now a direct dependency). Transport is a FastAPI app served by
uvicorn on its own background thread; ``/video.mjpg`` and ``/viewer`` (later
tickets) land on this same app.

The surface is fed through ``decision.run_turn``'s existing ``on_decision``
seam via ``stream_logger`` below (which also keeps #21's own log line), so
the surface updates as each new decision is logged, with no change to the
decision loop itself.

Read-only is structural, not just by convention: only ``GET`` routes are
defined anywhere on the app, so every other HTTP method against any route -
including ``HEAD`` on the snapshot route, which FastAPI would otherwise
answer automatically - is rejected with no mutation path added. The server
binds to loopback by default, so only a local process - which already has
whatever access the machine grants - can even see the state, let alone write
to it. Nothing read from here is ever fed back into ``run_turn``.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import UTC, datetime
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from jev_plays_pokemon.decision import Decision, DecisionLogger, log_decision
from jev_plays_pokemon.milestones import Milestone

logger = logging.getLogger(__name__)

# The one path the surface serves; every other GET is a 404.
SNAPSHOT_PATH = "/"


class MilestoneRef(BaseModel):
    id: str
    description: str


class MilestoneSplit(BaseModel):
    """The completed/current/future objective split (#18), as exposed."""

    completed: list[MilestoneRef] = Field(default_factory=list)
    current: MilestoneRef | None = None
    future: list[MilestoneRef] = Field(default_factory=list)


class Snapshot(BaseModel):
    """The surface's full wire shape - the contract a stream viewer polls.

    The field defaults *are* the pre-first-decision state: a poller that
    connects before Jev's first choice sees this same shape, never a
    different one.
    """

    decision_count: int = 0
    updated_at: datetime | None = None
    action: str | None = None
    confidence: float | None = None
    milestones: MilestoneSplit = Field(default_factory=MilestoneSplit)


def _serialize_milestone(milestone: Milestone) -> MilestoneRef:
    return MilestoneRef(id=milestone.milestone_id, description=milestone.description)


class StreamSurface:
    """The read-only state a stream viewer sees, updated per logged decision.

    ``record`` is the update side (a ``DecisionLogger``-shaped callable, wired
    into ``decision.run_turn``'s ``on_decision``); ``snapshot`` is the read
    side the HTTP handler serves. They run on different threads in
    production - the decision loop vs. the request handler - hence the lock
    guarding the shared latest-decision state.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: Decision | None = None
        self._decision_count = 0
        self._updated_at: datetime | None = None

    def record(self, decision: Decision) -> None:
        """Update the exposed state to `decision` - the #21 `on_decision` hook."""
        with self._lock:
            self._latest = decision
            self._decision_count += 1
            self._updated_at = datetime.now(UTC)

    def snapshot(self) -> dict[str, Any]:
        """The surface's current state as a JSON-able dict (`Snapshot` dumped).

        Always the same shape - a poller that connects before the first
        decision gets nulls and empty lists, not a different structure.
        """
        with self._lock:
            latest = self._latest
            decision_count = self._decision_count
            updated_at = self._updated_at
        if latest is None:
            return Snapshot().model_dump(mode="json")
        progress = latest.milestone_progress
        return Snapshot(
            decision_count=decision_count,
            updated_at=updated_at,
            action=latest.action,
            confidence=latest.confidence,
            milestones=MilestoneSplit(
                completed=[_serialize_milestone(m) for m in progress.completed],
                current=(
                    _serialize_milestone(progress.current) if progress.current else None
                ),
                future=[_serialize_milestone(m) for m in progress.future],
            ),
        ).model_dump(mode="json")


def _build_app(surface: StreamSurface) -> FastAPI:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)

    @app.get(SNAPSHOT_PATH)
    def get_snapshot() -> JSONResponse:
        logger.debug("GET %s", SNAPSHOT_PATH)
        return JSONResponse(
            content=surface.snapshot(),
            # Pollers (OBS browser sources included) must never see a
            # cached stale decision.
            headers={"Cache-Control": "no-store"},
        )

    # `@app.get` registers only "GET" against this path (verified: unlike
    # plain Starlette routes, FastAPI's `APIRoute` does not implicitly add
    # "HEAD"), so every other method - HEAD included - falls through to
    # FastAPI's own 405 handling with no extra code here.
    return app


class StreamSurfaceServer:
    """Handle to the background uvicorn server.

    Mirrors the small slice of `http.server`'s server API - `server_address`,
    `shutdown()`, `server_close()` - that callers (`main.py`, tests) already
    drive, so the transport swap needed no change on that side.
    """

    def __init__(self, server: uvicorn.Server, thread: threading.Thread) -> None:
        self._server = server
        self._thread = thread

    @property
    def server_address(self) -> tuple[str, int]:
        # `StreamSurfaceServer` is only ever constructed after
        # `start_stream_surface_server` has observed `server.started`, by
        # which point uvicorn has populated exactly one bound listener here.
        sock = self._server.servers[0].sockets[0]
        return sock.getsockname()[:2]

    def shutdown(self) -> None:
        """Stop serving and block until the background thread has exited."""
        self._server.should_exit = True
        self._thread.join()

    def server_close(self) -> None:
        """No-op: `shutdown` above already tears down uvicorn's sockets."""


# Bound on how long startup (including a failed bind) is given before
# `start_stream_surface_server` gives up - `ThreadingHTTPServer`'s bind used
# to fail synchronously in the constructor; uvicorn's happens on its
# background thread, so without a bound a failed bind would hang the caller
# forever instead of raising.
_STARTUP_TIMEOUT_SECONDS = 10.0


def start_stream_surface_server(
    surface: StreamSurface, host: str = "127.0.0.1", port: int = 0
) -> StreamSurfaceServer:
    """Serve `surface` over HTTP on its own daemon thread; return the server.

    `port=0` means the OS assigns an ephemeral port - read it back from
    `server.server_address[1]`; pass a fixed port when a stream tool needs a
    stable URL. Shut the surface down with `server.shutdown()` followed by
    `server.server_close()`.

    Raises `RuntimeError` if the server fails to start (e.g. the port is
    already in use) and `TimeoutError` if it neither starts nor fails within
    `_STARTUP_TIMEOUT_SECONDS`.
    """
    app = _build_app(surface)
    config = uvicorn.Config(app, host=host, port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, name="stream-surface", daemon=True)
    thread.start()

    deadline = time.monotonic() + _STARTUP_TIMEOUT_SECONDS
    while not server.started:
        if not thread.is_alive():
            raise RuntimeError(
                f"stream surface server failed to start on {host}:{port} "
                "(the port may already be in use)"
            )
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "stream surface server did not start within "
                f"{_STARTUP_TIMEOUT_SECONDS}s"
            )
        time.sleep(0.005)
    return StreamSurfaceServer(server, thread)


def stream_logger(surface: StreamSurface) -> DecisionLogger:
    """An `on_decision` hook that both logs (#21's own line) and updates `surface`.

    Handing ``StreamSurface.record`` to ``run_turn`` as ``on_decision``
    works but *replaces* #21's default ``log_decision`` - the seam is one
    callable, not a chain - so decisions would stop reaching the log. This
    composition keeps both, and is the intended wiring for a live loop.
    """

    def on_decision(decision: Decision) -> None:
        log_decision(decision)
        surface.record(decision)

    return on_decision
