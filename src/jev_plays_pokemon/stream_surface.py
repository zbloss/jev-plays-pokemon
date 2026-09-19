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

``/viewer`` (#50) serves the combined "Dossier Sidebar" page (#42's chosen
layout, prototyped on the throwaway ``prototype/overlay-design-42`` branch):
the live feed pinned left via a plain ``<img src="/video.mjpg">`` - the
browser renders a held-open MJPEG multipart response as a live-updating
image natively, no client-side JS needed to drive the video half - and a
fixed right-hand sidebar kept live by an inline script that polls
``GET /`` on a fixed interval and re-renders the action, a radial confidence
gauge, and the completed/current/future milestone split. The script's own
default state mirrors ``Snapshot``'s field defaults exactly, and is rendered
synchronously before the first poll resolves, so the page never has a
broken or half-populated moment before Jev's first decision - the same
guarantee ``Snapshot`` already gives JSON pollers, extended to this page.

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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

from jev_plays_pokemon.decision import Decision, DecisionLogger, log_decision
from jev_plays_pokemon.frame_capture import FrameCapture
from jev_plays_pokemon.milestones import Milestone

logger = logging.getLogger(__name__)

# The three paths the surface serves; every other GET is a 404.
SNAPSHOT_PATH = "/"
VIDEO_PATH = "/video.mjpg"
VIEWER_PATH = "/viewer"

# How often the /viewer sidebar's inline script re-polls SNAPSHOT_PATH - a
# fixed internal constant (not configurable), chosen to comfortably keep
# pace with the tactical loop's decision cadence.
_VIEWER_POLL_INTERVAL_MS = 750

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


# The combined "Dossier Sidebar" page (#42's chosen layout): video pinned
# left, a fixed right-hand sidebar kept live by the inline script below.
# `__VIDEO_PATH__` / `__SNAPSHOT_PATH__` / `__POLL_INTERVAL_MS__` are
# substituted from the constants above via plain `str.replace` (not
# `.format`/f-string) so the JS's own `{`/`}` never need escaping.
_VIEWER_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Jev — Live</title>
<style>
  :root {
    --ink: #e9e6da;
    --ink-dim: #a9a58f;
    --panel: #14161b;
    --panel-line: #2c303a;
    --gb-3: #9bbc0f;
    --gold: #f2b134;
  }
  * { box-sizing: border-box; }
  html, body {
    margin: 0;
    height: 100%;
    background: #000;
    color: var(--ink);
    font-family: "IBM Plex Sans", ui-sans-serif, system-ui, sans-serif;
  }
  .stage { display: flex; height: 100vh; width: 100vw; }
  .video { width: 64%; height: 100%; object-fit: contain; background: #000; }
  .sidebar {
    flex: 1;
    min-width: 0;
    background: rgba(14, 15, 12, 0.92);
    border-left: 1px solid var(--panel-line);
    padding: 16px 14px;
    display: flex;
    flex-direction: column;
    gap: 14px;
    overflow: auto;
  }
  .eyebrow {
    font-size: 11px;
    letter-spacing: 0.1em;
    color: var(--gold);
    text-transform: uppercase;
  }
  .action { font-size: 20px; font-weight: 700; line-height: 1.2; }
  .gauge { align-self: center; position: relative; width: 88px; height: 88px; margin: 4px 0; }
  .gauge svg { width: 100%; height: 100%; transform: rotate(-90deg); }
  .gauge .track { fill: none; stroke: var(--panel-line); stroke-width: 6; }
  .gauge .fill {
    fill: none;
    stroke: var(--gold);
    stroke-width: 6;
    stroke-linecap: round;
    transition: stroke-dashoffset 0.2s ease;
  }
  .gauge .pct {
    position: absolute;
    inset: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 15px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
  }
  hr { border: 0; border-top: 1px solid var(--panel-line); margin: 0; }
  .milestones { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 6px; font-size: 13px; }
  .milestones li { color: var(--ink-dim); }
  .milestones li.done { text-decoration: line-through; opacity: 0.5; }
  .milestones li.now {
    color: var(--gb-3);
    background: rgba(155, 188, 15, 0.1);
    border-left: 3px solid var(--gold);
    padding: 4px 6px;
    margin-left: -3px;
    border-radius: 2px;
    font-weight: 600;
  }
  .milestones li.future { opacity: 0.45; }
</style>
</head>
<body>
  <div class="stage">
    <img class="video" src="__VIDEO_PATH__" alt="Jev's live feed">
    <aside class="sidebar">
      <div>
        <div class="eyebrow">Now deciding</div>
        <div class="action" id="action">Awaiting first decision</div>
      </div>
      <div class="gauge">
        <svg viewBox="0 0 60 60">
          <circle class="track" cx="30" cy="30" r="26"></circle>
          <circle class="fill" id="gauge-fill" cx="30" cy="30" r="26"></circle>
        </svg>
        <span class="pct" id="pct">0%</span>
      </div>
      <hr>
      <ul class="milestones" id="milestones"></ul>
    </aside>
  </div>
<script>
(function () {
  "use strict";
  var POLL_INTERVAL_MS = __POLL_INTERVAL_MS__;
  var SNAPSHOT_PATH = "__SNAPSHOT_PATH__";
  // Mirrors Snapshot's own field defaults - the pre-first-decision state,
  // rendered synchronously below before the first poll ever resolves so
  // this page is never broken or half-populated before Jev's first choice.
  var DEFAULT_SNAPSHOT = {
    action: null,
    confidence: null,
    milestones: { completed: [], current: null, future: [] }
  };

  var actionEl = document.getElementById("action");
  var gaugeFill = document.getElementById("gauge-fill");
  var pctEl = document.getElementById("pct");
  var milestonesEl = document.getElementById("milestones");
  var CIRCUMFERENCE = 2 * Math.PI * 26;
  gaugeFill.style.strokeDasharray = String(CIRCUMFERENCE);

  function escapeHtml(text) {
    var div = document.createElement("div");
    div.textContent = text;
    return div.innerHTML;
  }

  function render(snapshot) {
    actionEl.textContent = snapshot.action || "Awaiting first decision";

    var confidence = snapshot.confidence || 0;
    pctEl.textContent = Math.round(confidence * 100) + "%";
    gaugeFill.style.strokeDashoffset = String(CIRCUMFERENCE - CIRCUMFERENCE * confidence);

    var milestones = snapshot.milestones || { completed: [], current: null, future: [] };
    var items = [];
    (milestones.completed || []).forEach(function (milestone) {
      items.push('<li class="done">' + escapeHtml(milestone.description) + "</li>");
    });
    if (milestones.current) {
      items.push('<li class="now">' + escapeHtml(milestones.current.description) + "</li>");
    } else {
      items.push('<li class="now">No current milestone</li>');
    }
    (milestones.future || []).forEach(function (milestone) {
      items.push('<li class="future">' + escapeHtml(milestone.description) + "</li>");
    });
    milestonesEl.innerHTML = items.join("");
  }

  function poll() {
    fetch(SNAPSHOT_PATH, { cache: "no-store" })
      .then(function (response) { return response.json(); })
      .then(render)
      .catch(function () {
        // Transient fetch error: keep the last good render rather than
        // clobbering it with a broken one.
      });
  }

  render(DEFAULT_SNAPSHOT);
  poll();
  setInterval(poll, POLL_INTERVAL_MS);
})();
</script>
</body>
</html>
"""

_VIEWER_HTML = (
    _VIEWER_TEMPLATE.replace("__VIDEO_PATH__", VIDEO_PATH)
    .replace("__SNAPSHOT_PATH__", SNAPSHOT_PATH)
    .replace("__POLL_INTERVAL_MS__", str(_VIEWER_POLL_INTERVAL_MS))
)


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

    @app.get(VIEWER_PATH)
    def get_viewer() -> HTMLResponse:
        logger.debug("GET %s", VIEWER_PATH)
        return HTMLResponse(content=_VIEWER_HTML)

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
    port: int = 8000,
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
