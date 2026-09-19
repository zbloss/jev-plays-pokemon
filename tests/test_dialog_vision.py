from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pyboy import PyBoy

from jev_plays_pokemon.dialog_vision import (
    VisionConfigError,
    capture_screen,
    decode_dialog_text,
    load_vision_client_and_model,
)

ROM_PATH = Path(__file__).resolve().parent.parent / "pokemon_red.gb"


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeOpenAIClient:
    """Stands in for `openai.OpenAI` - no real network call, per this ticket."""

    def __init__(self, content: str = "PROFESSOR OAK: Hello there!") -> None:
        self.chat = _FakeChat(content)


def test_decode_dialog_text_returns_the_fake_clients_transcription():
    client = _FakeOpenAIClient("PROFESSOR OAK: Hello there!")
    screen = Image.new("RGB", (160, 144), color="white")

    text = decode_dialog_text(client, "test-vision-model", screen)

    assert text == "PROFESSOR OAK: Hello there!"


def test_decode_dialog_text_strips_surrounding_whitespace():
    client = _FakeOpenAIClient("  PROFESSOR OAK: Hello there!  \n")
    screen = Image.new("RGB", (160, 144), color="white")

    text = decode_dialog_text(client, "test-vision-model", screen)

    assert text == "PROFESSOR OAK: Hello there!"


def test_decode_dialog_text_sends_the_model_and_an_inline_png_image():
    client = _FakeOpenAIClient()
    screen = Image.new("RGB", (160, 144), color="white")

    decode_dialog_text(client, "test-vision-model", screen)

    kwargs = client.chat.completions.last_kwargs
    assert kwargs is not None
    assert kwargs["model"] == "test-vision-model"
    content = kwargs["messages"][0]["content"]
    image_part = next(part for part in content if part["type"] == "image_url")
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_load_vision_client_and_model_targets_real_openai_by_default(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("OPENAI_VISION_MODEL", "gpt-test-vision")

    client, model = load_vision_client_and_model()

    assert model == "gpt-test-vision"
    assert str(client.base_url) == "https://api.openai.com/v1/"


def test_load_vision_client_and_model_requires_api_key_for_real_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_VISION_MODEL", "gpt-test-vision")

    with pytest.raises(VisionConfigError, match="OPENAI_API_KEY"):
        load_vision_client_and_model()


def test_load_vision_client_and_model_targets_custom_endpoint_without_api_key(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_VISION_MODEL", "qwen3-flash-next")

    client, model = load_vision_client_and_model()

    assert model == "qwen3-flash-next"
    assert str(client.base_url) == "http://localhost:8000/v1/"


def test_load_vision_client_and_model_passes_through_a_provided_key_for_custom_endpoint(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "custom-key")
    monkeypatch.setenv("OPENAI_VISION_MODEL", "qwen3-flash-next")

    client, _ = load_vision_client_and_model()

    assert client.api_key == "custom-key"


def test_load_vision_client_and_model_requires_a_model(monkeypatch):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "custom-key")
    monkeypatch.delenv("OPENAI_VISION_MODEL", raising=False)

    with pytest.raises(VisionConfigError, match="OPENAI_VISION_MODEL"):
        load_vision_client_and_model()


def test_load_vision_client_and_model_prefers_explicit_overrides_over_env(
    monkeypatch,
):
    monkeypatch.setenv("OPENAI_BASE_URL", "http://env-endpoint/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    monkeypatch.setenv("OPENAI_VISION_MODEL", "env-model")

    client, model = load_vision_client_and_model(
        base_url="http://explicit-endpoint/v1",
        api_key="explicit-key",
        model="explicit-model",
    )

    assert model == "explicit-model"
    assert str(client.base_url) == "http://explicit-endpoint/v1/"
    assert client.api_key == "explicit-key"


# `pokemon_red.gb` is gitignored (see `tests/test_game_state.py`), so this
# only runs where a local checkout has added its own copy.
@pytest.mark.skipif(not ROM_PATH.exists(), reason=f"{ROM_PATH} not present locally")
def test_capture_screen_reads_a_real_rendered_frame_under_the_null_window():
    # A rendered frame needs a render=True tick: PyBoy leaves screen.ndarray
    # stale (uniform) under render=False, so capturing before one would hand
    # the decoder a blank image while still reading the correct size/mode.
    pyboy = PyBoy(str(ROM_PATH), window="null")
    pyboy.set_emulation_speed(0)
    # Boot forward unrendered (buffer stays stale), then render until a frame
    # rasterizes real content - Gen 1's intro has blank-white stretches, so a
    # single fixed frame is not guaranteed to have pixels (see emulator.py's
    # render_current_frame for why a batched render tick won't render at all).
    for _ in range(60):
        pyboy.tick(1, False)
    screen = None
    for _ in range(240):
        pyboy.tick(1, True)
        candidate = capture_screen(pyboy)
        if float(np.std(np.asarray(candidate))) > 0.0:
            screen = candidate
            break

    assert screen is not None, (
        "no rendered frame rasterized content under the null window"
    )
    assert isinstance(screen, Image.Image)
    assert screen.size == (160, 144)
    assert screen.mode == "RGB"
    assert float(np.std(np.asarray(screen))) > 0.0
    pyboy.stop(save=False)
