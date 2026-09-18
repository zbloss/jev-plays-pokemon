import http.client
import json
import logging
from datetime import datetime
from email.message import Message
from http import HTTPStatus

from jev_plays_pokemon.decision import (
    NAVIGATION_MACRO_ACTION,
    Decision,
    run_turn,
)
from jev_plays_pokemon.game_state import BattleState, GameState
from jev_plays_pokemon.milestones import Milestone, MilestoneProgress, MilestoneTarget
from jev_plays_pokemon.stream_surface import (
    StreamSurface,
    start_stream_surface_server,
    stream_logger,
)


def _game_state(**overrides) -> GameState:
    defaults = {
        "party": (),
        "money": 0,
        "inventory": (),
        "badges": (),
        "event_flags": frozenset(),
        "battle": BattleState(
            in_battle=False,
            battle_type="none",
            opponent_species=None,
            opponent_level=None,
        ),
        "dialog_open": False,
        "map_id": 40,
        "map_name": "Oaks Lab",
        "player_x": 4,
        "player_y": 5,
    }
    defaults.update(overrides)
    return GameState(**defaults)


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
            assert status == HTTPStatus.NOT_IMPLEMENTED, method

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
