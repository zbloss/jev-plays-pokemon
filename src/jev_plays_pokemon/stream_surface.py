"""Stream-facing read-only surface: OBS (or any local viewer) polls Jev's live
decision state without codebase access, and can never interfere with it.

Implements #22, part of #14's MVP tactical action-selection loop: a minimal,
read-only view of the decision core's (#21) latest logged decision - chosen
action, confidence, and the completed/current/future objective split (#18) -
over a local HTTP JSON endpoint, so a streaming overlay can follow along.

Chosen shape (per this ticket's "implementer's choice, consistent with
minimal and read-only"): Python's stdlib ``http.server``, no new dependency.
The surface is fed through ``decision.run_turn``'s existing ``on_decision``
seam via ``stream_logger`` below (which also keeps #21's own log line), so
the surface updates as each new decision is logged, with no change to the
decision loop itself.

Read-only is structural, not just by convention: the HTTP handler implements
``do_GET`` only (``http.server`` answers every other method with ``501 Not
Implemented`` on its own), and the server binds to loopback by default, so
only a local process - which already has whatever access the machine
grants - can even see the state, let alone write to it. Nothing read from
here is ever fed back into ``run_turn``.
"""

from __future__ import annotations

import json
import logging
import threading
from datetime import UTC, datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from jev_plays_pokemon.decision import Decision, DecisionLogger, log_decision
from jev_plays_pokemon.milestones import Milestone

logger = logging.getLogger(__name__)

# The one path the surface serves; every other GET is a 404.
SNAPSHOT_PATH = "/"


def _serialize_milestone(milestone: Milestone) -> dict[str, str]:
    return {"id": milestone.milestone_id, "description": milestone.description}


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
        self._updated_at: str | None = None

    def record(self, decision: Decision) -> None:
        """Update the exposed state to `decision` - the #21 `on_decision` hook."""
        with self._lock:
            self._latest = decision
            self._decision_count += 1
            self._updated_at = datetime.now(UTC).isoformat()

    def snapshot(self) -> dict[str, Any]:
        """The surface's current state as a JSON-able dict.

        Always the same shape - a poller that connects before the first
        decision gets nulls and empty lists, not a different structure.
        """
        with self._lock:
            latest = self._latest
            decision_count = self._decision_count
            updated_at = self._updated_at
        if latest is None:
            progress = None
        else:
            progress = latest.milestone_progress
        return {
            "decision_count": decision_count,
            "updated_at": updated_at,
            "action": latest.action if latest else None,
            "confidence": latest.confidence if latest else None,
            "milestones": {
                "completed": (
                    [_serialize_milestone(m) for m in progress.completed]
                    if progress
                    else []
                ),
                "current": (
                    _serialize_milestone(progress.current)
                    if progress and progress.current
                    else None
                ),
                "future": (
                    [_serialize_milestone(m) for m in progress.future]
                    if progress
                    else []
                ),
            },
        }


class _StreamServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], surface: StreamSurface) -> None:
        self.surface = surface
        super().__init__(address, _ReadOnlyRequestHandler)


class _ReadOnlyRequestHandler(BaseHTTPRequestHandler):
    """Serves the snapshot; implements GET only.

    No do_POST/do_PUT/do_DELETE/... exists, so `BaseHTTPRequestHandler`
    rejects every other method with 501 on its own - the surface has no
    write path to misuse.
    """

    server: _StreamServer

    def do_GET(self) -> None:
        if self.path != SNAPSHOT_PATH:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        body = json.dumps(self.server.surface.snapshot()).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        # Pollers (OBS browser sources included) must never see a cached
        # stale decision.
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        logger.debug("%s - %s", self.address_string(), format % args)


def start_stream_surface_server(
    surface: StreamSurface, host: str = "127.0.0.1", port: int = 0
) -> ThreadingHTTPServer:
    """Serve `surface` over HTTP on its own daemon thread; return the server.

    `port=0` means the OS assigns an ephemeral port - read it back from
    `server.server_address[1]`; pass a fixed port when a stream tool needs a
    stable URL. Shut the surface down with `server.shutdown()` followed by
    `server.server_close()`.
    """
    server = _StreamServer((host, port), surface)
    threading.Thread(
        target=server.serve_forever, name="stream-surface", daemon=True
    ).start()
    return server


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
