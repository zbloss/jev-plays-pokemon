"""Frame capture: a cached, JPEG-encoded copy of the live PyBoy screen.

Part of #36 (Watch It Live - visual viewer for streaming), building on #41's
architecture decision: a pull-based ~10fps capture reads whatever screen
image the caller exposes, independent of the tactical decision loop's own
~3.3 turns/s cadence (`docs/turn-rate-budget.md`) - it never blocks on, or is
blocked by, that loop. A later ticket wires the frame source to
`pyboy.screen.image` and serves the cached bytes over a `GET /video.mjpg`
endpoint on `stream_surface.py`'s FastAPI app.

Structurally mirrors `StreamSurface` (`stream_surface.py`): an update side
(`capture`, driven by the background timer) and a read side (`read`), guarded
by one lock, since they run on different threads in production - the timer
thread vs. the request handler serving `/video.mjpg`. Starting the timer is
likewise a separate function (`start_frame_capture`) taking a `FrameCapture`,
mirroring `start_stream_surface_server`'s relationship to `StreamSurface`.
"""

from __future__ import annotations

import io
import logging
import threading
from collections.abc import Callable

from PIL import Image

logger = logging.getLogger(__name__)

# Fixed, not configurable - #41's decision. Trades a little visible
# artifacting for smaller/cheaper-to-encode frames at 160x144, ~10 times/sec.
_JPEG_QUALITY = 75

# ~10fps (~100ms), per #41 - independent of, and well under, the tactical
# decision loop's own ~3.3 turns/s budget (`docs/turn-rate-budget.md`).
_CAPTURE_INTERVAL_SECONDS = 0.1

# A zero-argument callable returning the current screen image - injected
# rather than a direct PyBoy reference, so this module never holds (or
# ticks) an emulator itself; production wires this to `pyboy.screen.image`.
FrameSource = Callable[[], Image.Image]


def _encode_jpeg(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
    return buffer.getvalue()


class FrameCapture:
    """The latest JPEG-encoded frame, refreshed one tick at a time.

    `capture` is the update side: read one frame from `source`, JPEG-encode
    it, and cache the bytes - exactly once per call, regardless of how many
    times `read` is called in between. `read` is the read side: it only
    ever returns whatever's cached, never triggering or waiting for a
    capture itself, so a slow reader can never stall the capture timer and a
    slow capture can never block a reader.
    """

    def __init__(self, source: FrameSource) -> None:
        self._source = source
        self._lock = threading.Lock()
        self._jpeg_bytes: bytes | None = None

    def capture(self) -> None:
        """Read one frame from `source` and cache its JPEG encoding."""
        image = self._source()
        encoded = _encode_jpeg(image)
        with self._lock:
            self._jpeg_bytes = encoded

    def read(self) -> bytes | None:
        """The most recently cached JPEG bytes, or `None` before the first `capture`.

        `None` is the documented not-yet-captured state - a caller polling
        before the first tick gets this, not an exception or stale garbage.
        """
        with self._lock:
            return self._jpeg_bytes


class FrameCaptureTimer:
    """Handle to the background daemon thread driving `FrameCapture.capture`.

    Mirrors `StreamSurfaceServer`'s shape (`stream_surface.py`): a small
    handle exposing `stop()`, so callers never need to know the timer's
    innards to control it.
    """

    def __init__(self, thread: threading.Thread, stop_event: threading.Event) -> None:
        self._thread = thread
        self._stop_event = stop_event

    def stop(self) -> None:
        """Stop the timer and block until the background thread has exited."""
        self._stop_event.set()
        self._thread.join()


def start_frame_capture(
    capture: FrameCapture, interval_seconds: float = _CAPTURE_INTERVAL_SECONDS
) -> FrameCaptureTimer:
    """Run `capture.capture()` on its own background daemon thread, forever.

    Ticks at a fixed ~10fps (`interval_seconds`, default
    `_CAPTURE_INTERVAL_SECONDS`) independent of, and never blocking on, the
    tactical decision loop's own cadence.

    A `source` that raises is logged and the loop continues onto the next
    tick rather than killing the thread - a dead capture timer would
    otherwise silently freeze the feed with no error visible to a viewer.
    """
    stop_event = threading.Event()

    def _run() -> None:
        while not stop_event.is_set():
            try:
                capture.capture()
            except Exception:
                logger.exception("frame capture tick failed")
            stop_event.wait(interval_seconds)

    thread = threading.Thread(target=_run, name="frame-capture", daemon=True)
    thread.start()
    return FrameCaptureTimer(thread, stop_event)
