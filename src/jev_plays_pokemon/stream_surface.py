"""Stream-facing read-only surface: OBS (or any local viewer) polls Jev's live
decision state without codebase access, and can never interfere with it.

Implements #22, part of #14's MVP tactical action-selection loop: a minimal,
read-only view of the decision core's (#21) latest logged decision - chosen
action, confidence, and the completed/current/future objective split (#18) -
over a local HTTP JSON endpoint, so a streaming overlay can follow along.

The wire shape a viewer sees is declared once as ``Snapshot`` below (pydantic
was already in the dependency tree via the vision-fallback path's ``openai``
SDK, and is now a direct dependency). Transport is a FastAPI app served by
uvicorn on its own background thread; ``/video.mjpg`` (this module, below)
and ``/viewer`` (a later ticket) land on this same app.

``/video.mjpg`` serves the frame-capture component's (``frame_capture.py``,
#48) cached JPEG bytes as an MJPEG (``multipart/x-mixed-replace``) stream,
hand-rolled on Starlette's ``StreamingResponse`` per #41's transport decision
- no MJPEG-serving or OpenCV-coupled third-party library is adopted. The
route only ever re-yields whatever ``FrameCapture.read()`` currently has
cached, on its own delivery cadence; it never triggers or waits for a
capture itself, so any number of concurrent viewers share the one capture
timer's work with no extra encode cost per viewer and no artificial cap on
how many can connect.

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

import asyncio
import logging
import threading
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import uvicorn
from fastapi import FastAPI
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from jev_plays_pokemon.decision import Decision, DecisionLogger, log_decision
from jev_plays_pokemon.frame_capture import FrameCapture
from jev_plays_pokemon.milestones import Milestone

logger = logging.getLogger(__name__)

# The two paths the surface serves; every other GET is a 404.
SNAPSHOT_PATH = "/"
VIDEO_PATH = "/video.mjpg"

# Arbitrary, fixed boundary token for the multipart stream - never
# negotiated, so it's just a constant both the header and each part's
# marker line reference.
_MJPEG_BOUNDARY = "frame"

# How often the route re-yields whatever's cached, independent of the
# frame-capture component's own tick rate (`frame_capture.py`'s
# ~10fps/`_CAPTURE_INTERVAL_SECONDS`) - this loop never triggers a capture,
# it only decides how often to re-check the cache.
_STREAM_POLL_INTERVAL_SECONDS = 0.1


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


def _no_frame_source() -> Image.Image:
    # Only reachable if something calls `FrameCapture.capture()` on the
    # default placeholder below - `read()` never does, so a server started
    # with no `frame_capture` argument just serves an eternally-empty
    # `/video.mjpg` (no capture wired yet is #51's job) rather than crashing.
    raise RuntimeError("no frame source wired for this stream surface (see #51)")


async def _mjpeg_parts(frame_capture: FrameCapture) -> AsyncIterator[bytes]:
    """Re-yield `frame_capture`'s cached JPEG bytes as multipart parts.

    Never triggers or waits for a capture - each iteration just re-checks
    whatever `read()` currently has cached, on this loop's own cadence
    (`_STREAM_POLL_INTERVAL_SECONDS`), so any number of concurrent viewers
    of `/video.mjpg` share the one capture timer's work. Runs until the
    client disconnects, at which point Starlette cancels this generator.
    """
    while True:
        jpeg_bytes = frame_capture.read()
        if jpeg_bytes is not None:
            yield (
                (
                    f"--{_MJPEG_BOUNDARY}\r\n"
                    "Content-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpeg_bytes)}\r\n\r\n"
                ).encode()
                + jpeg_bytes
                + b"\r\n"
            )
        await asyncio.sleep(_STREAM_POLL_INTERVAL_SECONDS)


def _build_app(surface: StreamSurface, frame_capture: FrameCapture | None) -> FastAPI:
    app = FastAPI(openapi_url=None, docs_url=None, redoc_url=None)
    video_source = (
        frame_capture if frame_capture is not None else FrameCapture(_no_frame_source)
    )

    @app.get(SNAPSHOT_PATH)
    def get_snapshot() -> JSONResponse:
        logger.debug("GET %s", SNAPSHOT_PATH)
        return JSONResponse(
            content=surface.snapshot(),
            # Pollers (OBS browser sources included) must never see a
            # cached stale decision.
            headers={"Cache-Control": "no-store"},
        )

    @app.get(VIDEO_PATH)
    def get_video() -> StreamingResponse:
        logger.debug("GET %s", VIDEO_PATH)
        return StreamingResponse(
            _mjpeg_parts(video_source),
            media_type=f"multipart/x-mixed-replace; boundary={_MJPEG_BOUNDARY}",
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
    surface: StreamSurface,
    host: str = "127.0.0.1",
    port: int = 0,
    frame_capture: FrameCapture | None = None,
) -> StreamSurfaceServer:
    """Serve `surface` over HTTP on its own daemon thread; return the server.

    `port=0` means the OS assigns an ephemeral port - read it back from
    `server.server_address[1]`; pass a fixed port when a stream tool needs a
    stable URL. Shut the surface down with `server.shutdown()` followed by
    `server.server_close()`.

    `frame_capture` (`frame_capture.py`, #48) backs `/video.mjpg` - pass the
    same instance a background `FrameCaptureTimer` is filling for a live
    feed. Omitted, `/video.mjpg` still exists but never yields a frame,
    since nothing wires a real capture until #51.

    Raises `RuntimeError` if the server fails to start (e.g. the port is
    already in use) and `TimeoutError` if it neither starts nor fails within
    `_STARTUP_TIMEOUT_SECONDS`.
    """
    app = _build_app(surface, frame_capture)
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
