import pytest
from PIL import Image

from jev_plays_pokemon import dialog_decode
from jev_plays_pokemon.dialog_decode import DialogDecodeConfigError, load_dialog_decoder
from jev_plays_pokemon.dialog_vision import VisionConfigError

_SCREEN = Image.new("RGB", (160, 144), color="white")
_FAKE_CLIENT = object()
_FAKE_ENGINE = object()


def _fake_vision_client_and_model():
    return _FAKE_CLIENT, "test-vision-model"


def _fake_ocr_engine():
    return _FAKE_ENGINE


def test_default_backend_is_vision_llm(monkeypatch):
    monkeypatch.delenv("DIALOG_DECODE_BACKEND", raising=False)
    monkeypatch.setattr(
        dialog_decode, "load_vision_client_and_model", _fake_vision_client_and_model
    )
    calls = []
    monkeypatch.setattr(
        dialog_decode,
        "decode_dialog_text",
        lambda client, model, screen: calls.append((client, model, screen)) or "ok",
    )

    decoder = load_dialog_decoder()
    result = decoder(_SCREEN)

    assert result == "ok"
    assert calls == [(_FAKE_CLIENT, "test-vision-model", _SCREEN)]


def test_explicit_vision_llm_backend_selects_the_vision_path(monkeypatch):
    monkeypatch.setenv("DIALOG_DECODE_BACKEND", "vision-llm")
    monkeypatch.setattr(
        dialog_decode, "load_vision_client_and_model", _fake_vision_client_and_model
    )
    monkeypatch.setattr(
        dialog_decode, "decode_dialog_text", lambda client, model, screen: "vision-result"
    )

    decoder = load_dialog_decoder()

    assert decoder(_SCREEN) == "vision-result"


def test_rapidocr_backend_is_case_insensitive_and_trims_whitespace(monkeypatch):
    monkeypatch.setenv("DIALOG_DECODE_BACKEND", "  RapidOCR  ")
    monkeypatch.setattr(dialog_decode, "load_ocr_engine", _fake_ocr_engine)
    calls = []
    monkeypatch.setattr(
        dialog_decode,
        "decode_dialog_text_ocr",
        lambda engine, screen: calls.append((engine, screen)) or "ocr-result",
    )

    decoder = load_dialog_decoder()
    result = decoder(_SCREEN)

    assert result == "ocr-result"
    assert calls == [(_FAKE_ENGINE, _SCREEN)]


def test_unrecognized_backend_raises_config_error(monkeypatch):
    monkeypatch.setenv("DIALOG_DECODE_BACKEND", "tesseract")

    with pytest.raises(DialogDecodeConfigError, match="tesseract"):
        load_dialog_decoder()


def test_vision_llm_config_errors_still_propagate(monkeypatch):
    monkeypatch.delenv("DIALOG_DECODE_BACKEND", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_VISION_MODEL", raising=False)

    with pytest.raises(VisionConfigError, match="OPENAI_VISION_MODEL"):
        load_dialog_decoder()
