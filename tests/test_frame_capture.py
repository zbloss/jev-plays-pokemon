import io

import pytest
from PIL import Image

from jev_plays_pokemon.frame_capture import (
    DEFAULT_DISPLAY_FPS,
    FrameCapture,
    interval_seconds_for_fps,
    start_frame_capture,
)


class _FakeSource:
    """Call-counting fake standing in for `pyboy.screen.image` - hands back
    whatever `frame` is currently set to, no real PyBoy anywhere."""

    def __init__(self, frame: Image.Image) -> None:
        self.frame = frame
        self.calls = 0

    def __call__(self) -> Image.Image:
        self.calls += 1
        return self.frame

    def __getattr__(self, name: str):
        # Acceptance criterion: `FrameCapture` must never call anything
        # resembling `pyboy.tick()` - accessing `.tick` on the source fails
        # loudly rather than silently no-op-ing, unlike a real PyBoy
        # instance which would have such a method to call.
        if name == "tick":
            raise AssertionError("FrameCapture must never call .tick()")
        raise AttributeError(name)


def _solid_image(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (160, 144), color)


def test_read_reports_not_yet_captured_before_any_capture():
    capture = FrameCapture(_FakeSource(_solid_image((255, 0, 0))))

    assert capture.read() is None


def test_read_returns_jpeg_bytes_derived_from_the_source_after_one_capture():
    source = _FakeSource(_solid_image((255, 0, 0)))
    capture = FrameCapture(source)

    capture.capture()
    jpeg_bytes = capture.read()

    assert jpeg_bytes is not None
    # JPEG magic bytes / EOI marker - confirms this is really a JPEG encode
    # of the source's frame, not the raw source or an empty placeholder.
    assert jpeg_bytes.startswith(b"\xff\xd8")
    assert jpeg_bytes.endswith(b"\xff\xd9")
    decoded = Image.open(io.BytesIO(jpeg_bytes))
    decoded.load()
    assert decoded.size == (160, 144)


def test_read_is_cached_and_never_invokes_the_source_or_re_encodes():
    source = _FakeSource(_solid_image((0, 255, 0)))
    capture = FrameCapture(source)
    capture.capture()
    assert source.calls == 1

    first = capture.read()
    second = capture.read()
    third = capture.read()

    assert first == second == third
    # No extra source calls from repeated reads - the source is only ever
    # invoked by `capture`, never by `read`.
    assert source.calls == 1


def test_a_second_capture_with_a_changed_frame_updates_the_cached_bytes():
    source = _FakeSource(_solid_image((0, 0, 255)))
    capture = FrameCapture(source)
    capture.capture()
    first = capture.read()

    source.frame = _solid_image((255, 255, 0))
    capture.capture()
    second = capture.read()

    assert first != second
    assert source.calls == 2


def test_frame_capture_never_calls_anything_resembling_pyboy_tick():
    capture = FrameCapture(_FakeSource(_solid_image((10, 20, 30))))

    capture.capture()

    assert capture.read() is not None


def test_default_display_fps_is_60():
    assert DEFAULT_DISPLAY_FPS == 60.0


def test_interval_seconds_for_fps_is_the_reciprocal_of_the_rate():
    assert interval_seconds_for_fps(60.0) == pytest.approx(1.0 / 60.0)
    assert interval_seconds_for_fps(10.0) == pytest.approx(0.1)


def test_interval_seconds_for_fps_rejects_zero_or_negative():
    with pytest.raises(ValueError):
        interval_seconds_for_fps(0.0)
    with pytest.raises(ValueError):
        interval_seconds_for_fps(-5.0)


def test_start_frame_capture_returns_a_handle_that_stops_cleanly():
    # Capture correctness (the acceptance criteria above) is exercised via
    # direct `capture.capture()` calls, per the ticket's own test
    # methodology ("no real timer in the tests"); this only checks that the
    # background-thread handle starts and stops without hanging - a hang
    # here would fail the test via its own timeout.
    capture = FrameCapture(_FakeSource(_solid_image((1, 2, 3))))

    timer = start_frame_capture(capture, interval_seconds=0.01)
    timer.stop()
