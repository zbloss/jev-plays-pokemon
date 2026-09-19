import contextlib
import dataclasses
import http.client
import io
import json
import logging
import re
import threading
import time
from datetime import datetime
from email.message import Message
from http import HTTPStatus

import pytest
from PIL import Image

from jev_plays_pokemon.decision import (
    NAVIGATION_MACRO_ACTION,
    Decision,
    run_turn,
)
from jev_plays_pokemon.frame_capture import FrameCapture, start_frame_capture
from jev_plays_pokemon.game_state import BattleState, GameState
from jev_plays_pokemon.milestones import Milestone, MilestoneProgress, MilestoneTarget
from jev_plays_pokemon.stream_surface import (
    VIDEO_PATH,
    VIEWER_PATH,
    StreamSurface,
    start_stream_surface_server,
    stream_logger,
)


def _game_state(**overrides) -> GameState:
    base = GameState(
        party=(),
        money=0,
        inventory=(),
        badges=(),
        event_flags=frozenset(),
        battle=BattleState(
            in_battle=False,
            battle_type="none",
            opponent_species=None,
            opponent_level=None,
        ),
        dialog_open=False,
        map_id=40,
        map_name="Oaks Lab",
        player_x=4,
        player_y=5,
    )
    return dataclasses.replace(base, **overrides)


class _FakeChoiceAnswer:
    def __init__(self, choice: str, confidence: float) -> None:
        self.choice = choice
        self.confidence = confidence


class _FakeJevClient:
    """Scripted fake standing in for `typesafe_sdk.TypeSafeClient` - hands
    out the next (choice, confidence) per call, no real TypeSafe API call."""

    def __init__(self, scripted: list[tuple[str, float]]) -> None:
        self._scripted = scripted
        self._calls = 0

    def system_one(self, state, questions):
        choice, confidence = self._scripted[self._calls]
        self._calls += 1

        class _Result:
            def __init__(self, choices: dict) -> None:
                self.choices = choices

        return _Result({"action": _FakeChoiceAnswer(choice, confidence)})


def _milestone(milestone_id: str) -> Milestone:
    return Milestone(
        milestone_id=milestone_id,
        description=f"{milestone_id} description",
        target=MilestoneTarget(map_id=40, map_name="Oaks Lab"),
    )


def _decision(
    action: str,
    confidence: float,
    completed: tuple[Milestone, ...],
    current: Milestone | None,
    future: tuple[Milestone, ...],
) -> Decision:
    return Decision(
        action=action,
        confidence=confidence,
        milestone_progress=MilestoneProgress(
            completed=completed, current=current, future=future
        ),
    )


def test_snapshot_exposes_a_stable_empty_shape_before_any_decision():
    # Stream consumers can poll at any time, including before the first
    # decision exists - the shape must not change once decisions start.
    assert StreamSurface().snapshot() == {
        "decision_count": 0,
        "updated_at": None,
        "action": None,
        "confidence": None,
        "milestones": {"completed": [], "current": None, "future": []},
    }


def test_snapshot_reflects_the_most_recent_recorded_decision():
    surface = StreamSurface()
    surface.record(_decision("a", 0.9, (), _milestone("got_starter"), ()))
    surface.record(
        _decision(
            "down",
            0.42,
            (_milestone("got_starter"),),
            _milestone("got_oaks_parcel"),
            (_milestone("got_pokedex"), _milestone("boulder_badge")),
        )
    )

    snapshot = surface.snapshot()

    assert snapshot["action"] == "down"
    assert snapshot["confidence"] == 0.42
    assert snapshot["decision_count"] == 2
    # Freshness marker a poller can check, as parseable UTC.
    assert datetime.fromisoformat(snapshot["updated_at"]).tzinfo is not None
    assert snapshot["milestones"]["completed"] == [
        {"id": "got_starter", "description": "got_starter description"}
    ]
    assert snapshot["milestones"]["current"] == {
        "id": "got_oaks_parcel",
        "description": "got_oaks_parcel description",
    }
    assert [m["id"] for m in snapshot["milestones"]["future"]] == [
        "got_pokedex",
        "boulder_badge",
    ]


def test_surface_reflects_each_decision_logged_by_the_decision_core():
    # This ticket's acceptance test: drive #21's real `run_turn` with a
    # scripted fake Jev client and a fake state source (no PyBoy, ROM, or
    # TypeSafe call anywhere), and watch the surface's exposed state track
    # the sequence of logged decisions as it happens.
    surface = StreamSurface()
    pre_starter = _game_state()
    # Event flag 34 = got_starter complete (see milestones.py).
    post_starter = _game_state(event_flags=frozenset({34}))
    states = [pre_starter, post_starter]
    client = _FakeJevClient([("a", 0.91), (NAVIGATION_MACRO_ACTION, 0.55)])

    run_turn(lambda: states[0], client, lambda action: None, on_decision=surface.record)
    first = surface.snapshot()

    run_turn(lambda: states[1], client, lambda action: None, on_decision=surface.record)
    second = surface.snapshot()

    assert first["decision_count"] == 1
    assert first["action"] == "a"
    assert first["confidence"] == 0.91
    assert first["milestones"]["completed"] == []
    assert first["milestones"]["current"]["id"] == "got_starter"

    assert second["decision_count"] == 2
    assert second["action"] == NAVIGATION_MACRO_ACTION
    assert second["confidence"] == 0.55
    assert [m["id"] for m in second["milestones"]["completed"]] == ["got_starter"]
    assert second["milestones"]["current"]["id"] == "got_oaks_parcel"
    assert second["milestones"]["future"][0]["id"] == "got_pokedex"


def _request(
    server, method: str = "GET", path: str = "/"
) -> tuple[int, Message, bytes]:
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.server_address[1], timeout=5
    )
    try:
        connection.request(method, path)
        response = connection.getresponse()
        return response.status, response.headers, response.read()
    finally:
        connection.close()


class _FakeFrameSource:
    """Call-counting fake standing in for `pyboy.screen.image` (mirrors
    `test_frame_capture.py`'s own fake) - no real PyBoy anywhere."""

    def __init__(self, frame: Image.Image) -> None:
        self.frame = frame
        self.calls = 0

    def __call__(self) -> Image.Image:
        self.calls += 1
        return self.frame


def _solid_image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (160, 144), color)


_CONTENT_LENGTH_RE = re.compile(rb"Content-Length: (\d+)", re.IGNORECASE)


@contextlib.contextmanager
def _video_connection(server, path: str = VIDEO_PATH, timeout: float = 5):
    """One real HTTP connection to `path`, closed on exit.

    `/video.mjpg`'s response never ends on its own, so every caller against
    it - a header-only check or a frame-demultiplexing read - needs the
    same open/close shape around a possibly-partial read; shared here so
    that shape exists once.
    """
    connection = http.client.HTTPConnection(
        "127.0.0.1", server.server_address[1], timeout=timeout
    )
    try:
        connection.request("GET", path)
        yield connection.getresponse()
    finally:
        connection.close()


def _read_mjpeg_frames(server, count: int, path: str = VIDEO_PATH) -> list[bytes]:
    """Open one real HTTP connection to `path` and demultiplex `count` parts.

    Reads only as many bytes off the wire as needed to collect `count`
    frames, then closes the connection - the route's generator loops
    forever otherwise, so this is what lets a real-HTTP test against it
    terminate.
    """
    with _video_connection(server, path) as response:
        assert response.status == HTTPStatus.OK
        boundary = response.headers.get_param("boundary")
        marker = f"--{boundary}\r\n".encode()

        frames: list[bytes] = []
        buffer = b""
        while len(frames) < count:
            chunk = response.read(4096)
            if not chunk:
                break
            buffer += chunk
            while len(frames) < count:
                start = buffer.find(marker)
                if start == -1:
                    break
                rest = buffer[start + len(marker) :]
                header_end = rest.find(b"\r\n\r\n")
                if header_end == -1:
                    break
                match = _CONTENT_LENGTH_RE.search(rest[:header_end])
                assert match is not None
                content_length = int(match.group(1))
                body_start = header_end + 4
                body_end = body_start + content_length
                if len(rest) < body_end + 2:
                    break
                frames.append(rest[body_start:body_end])
                buffer = rest[body_end + 2 :]
        return frames


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_start_fails_fast_when_the_port_is_already_in_use():
    # A bind failure must surface as an exception to the caller, not hang -
    # `ThreadingHTTPServer`'s bind used to fail synchronously in its
    # constructor; the uvicorn-backed server binds on a background thread,
    # so this must be actively detected rather than left to a timeout.
    # (uvicorn's own background thread also raises `SystemExit` internally
    # on the failed bind - pytest reports that as an unhandled thread
    # exception by default; it's the expected shape of this failure, not a
    # bug, so it's silenced here rather than for the whole suite.)
    surface = StreamSurface()
    holder = start_stream_surface_server(surface)
    try:
        taken_port = holder.server_address[1]
        with pytest.raises(RuntimeError):
            start_stream_surface_server(StreamSurface(), port=taken_port)
    finally:
        holder.shutdown()
        holder.server_close()


def test_served_endpoint_returns_the_live_snapshot_as_json():
    surface = StreamSurface()
    server = start_stream_surface_server(surface)
    try:
        # Bound to loopback by default - only local viewers can see it.
        assert server.server_address[0] == "127.0.0.1"

        surface.record(_decision("a", 0.91, (), _milestone("got_starter"), ()))

        status, headers, body = _request(server)

        served = json.loads(body)
        assert status == 200
        assert served["action"] == "a"
        assert served["milestones"]["current"]["id"] == "got_starter"
        assert headers.get_content_type() == "application/json"
        # Pollers (OBS browser sources included) must never see a cached
        # stale decision.
        assert headers.get("Cache-Control") == "no-store"

        # A decision logged after the server started is visible on the next
        # poll, with no restart.
        surface.record(_decision("b", 0.3, (_milestone("got_starter"),), None, ()))
        status, headers, body = _request(server)
        served = json.loads(body)
        assert status == 200
        assert served["decision_count"] == 2
        assert served["milestones"]["current"] is None
    finally:
        server.shutdown()
        server.server_close()


def test_surface_exposes_no_write_mutation_or_command_path():
    # This ticket's acceptance criterion: nothing external can interfere
    # with Jev's decisions - every non-GET method is rejected outright and
    # leaves the surface's state untouched.
    surface = StreamSurface()
    server = start_stream_surface_server(surface)
    try:
        surface.record(_decision("a", 0.9, (), _milestone("got_starter"), ()))
        before = surface.snapshot()

        for method in ("POST", "PUT", "PATCH", "DELETE", "HEAD"):
            status, _, _ = _request(server, method=method)
            assert status == HTTPStatus.METHOD_NOT_ALLOWED, method

        assert surface.snapshot() == before
    finally:
        server.shutdown()
        server.server_close()


def test_only_the_status_path_is_served():
    surface = StreamSurface()
    server = start_stream_surface_server(surface)
    try:
        for path in ("/status.json", "/shutdown", "/../../etc/passwd"):
            status, _, _ = _request(server, path=path)
            assert status == HTTPStatus.NOT_FOUND, path
    finally:
        server.shutdown()
        server.server_close()


def test_stream_logger_updates_the_surface_without_losing_the_decision_log(caplog):
    # on_decision is a single replacement seam, not a chain: the composed
    # hook must keep #21's own unconditional logging AND update the surface.
    surface = StreamSurface()
    state = _game_state()
    client = _FakeJevClient([("left", 0.77)])

    with caplog.at_level(logging.INFO, logger="jev_plays_pokemon.decision"):
        run_turn(
            lambda: state,
            client,
            lambda action: None,
            on_decision=stream_logger(surface),
        )

    assert any("action=left" in record.getMessage() for record in caplog.records)
    assert surface.snapshot()["action"] == "left"
    assert surface.snapshot()["confidence"] == 0.77


def test_poll_requests_are_logged_at_debug_level(caplog):
    surface = StreamSurface()
    server = start_stream_surface_server(surface)
    try:
        with caplog.at_level(logging.DEBUG, logger="jev_plays_pokemon.stream_surface"):
            _request(server)

        assert any("GET /" in record.getMessage() for record in caplog.records)
    finally:
        server.shutdown()
        server.server_close()


def test_video_route_serves_multipart_x_mixed_replace():
    surface = StreamSurface()
    capture = FrameCapture(_FakeFrameSource(_solid_image((10, 20, 30))))
    capture.capture()
    server = start_stream_surface_server(surface, frame_capture=capture)
    try:
        with _video_connection(server) as response:
            assert response.status == HTTPStatus.OK
            assert response.headers.get_content_type() == "multipart/x-mixed-replace"
            assert response.headers.get_param("boundary") is not None
    finally:
        server.shutdown()
        server.server_close()


def test_video_route_with_no_frame_capture_starts_but_never_yields_a_frame():
    # `start_stream_surface_server`'s `frame_capture` is optional - #51 (not
    # this ticket) wires a real one into `main.py`. Until then the route
    # must still exist and respond, just with nothing ever cached to yield.
    surface = StreamSurface()
    server = start_stream_surface_server(surface)
    try:
        with _video_connection(server, timeout=1) as response:
            assert response.status == HTTPStatus.OK
            assert response.headers.get_content_type() == "multipart/x-mixed-replace"
            with pytest.raises(TimeoutError):
                response.read(1)
    finally:
        server.shutdown()
        server.server_close()


def test_video_route_streams_frames_the_fake_source_produced():
    # This ticket's real-HTTP acceptance test: start the real FastAPI/uvicorn
    # app against a frame-capture object wired to a fake source, hit it with
    # a real HTTP client, and demultiplex the body back into JPEG frames.
    surface = StreamSurface()
    source = _FakeFrameSource(_solid_image((200, 30, 90)))
    capture = FrameCapture(source)
    capture.capture()
    server = start_stream_surface_server(surface, frame_capture=capture)
    try:
        frames = _read_mjpeg_frames(server, count=2)

        assert len(frames) == 2
        for frame_bytes in frames:
            assert frame_bytes == capture.read()
            decoded = Image.open(io.BytesIO(frame_bytes))
            decoded.load()
            assert decoded.size == (160, 144)
    finally:
        server.shutdown()
        server.server_close()


def test_two_concurrent_viewers_never_trigger_extra_capture_or_encode_work():
    # This ticket's other acceptance criterion, checked against a *real*
    # background `FrameCaptureTimer` rather than a manually-invoked
    # `.capture()` - so this actually verifies "the capture object's own
    # tick rate" (the spec's phrasing), not just that reads never call
    # `.capture()` in isolation. The timer is stopped before either viewer
    # connects, freezing the tick count, so the assertion below is exact
    # rather than a timing-dependent range.
    surface = StreamSurface()
    source = _FakeFrameSource(_solid_image((1, 2, 3)))
    capture = FrameCapture(source)
    timer = start_frame_capture(capture, interval_seconds=0.01)
    deadline = time.monotonic() + 5
    while source.calls < 2 and time.monotonic() < deadline:
        time.sleep(0.005)
    timer.stop()
    ticks_before_viewers = source.calls
    assert ticks_before_viewers >= 2

    server = start_stream_surface_server(surface, frame_capture=capture)
    try:
        results: dict[str, list[bytes]] = {}

        def _collect(key: str) -> None:
            results[key] = _read_mjpeg_frames(server, count=3)

        readers = [
            threading.Thread(target=_collect, args=("a",)),
            threading.Thread(target=_collect, args=("b",)),
        ]
        for reader in readers:
            reader.start()
        for reader in readers:
            reader.join(timeout=10)

        assert len(results["a"]) == 3
        assert len(results["b"]) == 3
        assert all(frame == capture.read() for frame in results["a"] + results["b"])
        # Two viewers, each reading 3 frames off two separate connections,
        # never pushed the tick count past what the (now-stopped) timer had
        # already produced on its own.
        assert source.calls == ticks_before_viewers
    finally:
        server.shutdown()
        server.server_close()


def test_video_route_exposes_no_write_mutation_path():
    surface = StreamSurface()
    capture = FrameCapture(_FakeFrameSource(_solid_image((4, 5, 6))))
    capture.capture()
    server = start_stream_surface_server(surface, frame_capture=capture)
    try:
        for method in ("POST", "PUT", "PATCH", "DELETE", "HEAD"):
            status, _, _ = _request(server, method=method, path=VIDEO_PATH)
            assert status == HTTPStatus.METHOD_NOT_ALLOWED, method
    finally:
        server.shutdown()
        server.server_close()


def test_viewer_route_serves_html_referencing_the_video_feed():
    surface = StreamSurface()
    server = start_stream_surface_server(surface)
    try:
        status, headers, body = _request(server, path=VIEWER_PATH)

        assert status == HTTPStatus.OK
        assert headers.get_content_type() == "text/html"
        html = body.decode()
        assert VIDEO_PATH in html
    finally:
        server.shutdown()
        server.server_close()


def test_viewer_route_renders_the_pre_first_decision_default_state():
    # This ticket's acceptance test: hit /viewer on a fresh StreamSurface
    # (no decision recorded yet) with a real HTTP request and confirm it
    # renders successfully, matching Snapshot's own field defaults rather
    # than erroring or half-populating.
    surface = StreamSurface()
    server = start_stream_surface_server(surface)
    try:
        status, _headers, body = _request(server, path=VIEWER_PATH)

        assert status == HTTPStatus.OK
        html = body.decode()
        assert "Awaiting first decision" in html
        assert "No current milestone" in html
        # The default snapshot embedded for the script's initial render
        # mirrors Snapshot()'s own field defaults exactly.
        assert "action: null" in html
        assert "confidence: null" in html
        assert "completed: [], current: null, future: []" in html
    finally:
        server.shutdown()
        server.server_close()


def test_viewer_route_exposes_no_write_mutation_path():
    surface = StreamSurface()
    server = start_stream_surface_server(surface)
    try:
        for method in ("POST", "PUT", "PATCH", "DELETE", "HEAD"):
            status, _, _ = _request(server, method=method, path=VIEWER_PATH)
            assert status == HTTPStatus.METHOD_NOT_ALLOWED, method
    finally:
        server.shutdown()
        server.server_close()


def test_snapshot_route_is_unaffected_by_the_video_route_existing():
    surface = StreamSurface()
    capture = FrameCapture(_FakeFrameSource(_solid_image((7, 8, 9))))
    capture.capture()
    server = start_stream_surface_server(surface, frame_capture=capture)
    try:
        surface.record(_decision("a", 0.91, (), _milestone("got_starter"), ()))
        status, headers, body = _request(server)

        served = json.loads(body)
        assert status == HTTPStatus.OK
        assert served["action"] == "a"
        assert headers.get_content_type() == "application/json"
    finally:
        server.shutdown()
        server.server_close()
