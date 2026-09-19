import logging

import pytest
from PIL import Image

from jev_plays_pokemon import dialog_decode
from jev_plays_pokemon.dialog_decode import (
    DialogDecodeConfigError,
    load_dialog_decoder,
    load_dialog_decoder_or_none,
)
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
        dialog_decode,
        "decode_dialog_text",
        lambda client, model, screen: "vision-result",
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


def test_explicit_backend_override_beats_the_environment_variable(monkeypatch):
    monkeypatch.setenv("DIALOG_DECODE_BACKEND", "vision-llm")
    monkeypatch.setattr(dialog_decode, "load_ocr_engine", _fake_ocr_engine)
    monkeypatch.setattr(
        dialog_decode,
        "decode_dialog_text_ocr",
        lambda engine, screen: "ocr-result",
    )

    decoder = load_dialog_decoder(backend="rapidocr")

    assert decoder(_SCREEN) == "ocr-result"


def test_explicit_vision_overrides_are_forwarded_to_load_vision_client_and_model(
    monkeypatch,
):
    captured: dict = {}

    def fake_load_vision_client_and_model(**kwargs):
        captured.update(kwargs)
        return _FAKE_CLIENT, "resolved-model"

    monkeypatch.setattr(
        dialog_decode,
        "load_vision_client_and_model",
        fake_load_vision_client_and_model,
    )
    monkeypatch.setattr(
        dialog_decode,
        "decode_dialog_text",
        lambda client, model, screen: "ok",
    )

    decoder = load_dialog_decoder(
        base_url="http://explicit/v1", api_key="explicit-key", model="explicit-model"
    )

    assert decoder(_SCREEN) == "ok"
    assert captured == {
        "base_url": "http://explicit/v1",
        "api_key": "explicit-key",
        "model": "explicit-model",
    }


def test_vision_llm_config_errors_still_propagate(monkeypatch):
    monkeypatch.delenv("DIALOG_DECODE_BACKEND", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_VISION_MODEL", raising=False)

    with pytest.raises(VisionConfigError, match="OPENAI_VISION_MODEL"):
        load_dialog_decoder()


# -- load_dialog_decoder_or_none: the loop's "degrade if unconfigured" wrapper --


def test_load_or_none_returns_a_decoder_when_the_backend_is_configured(monkeypatch):
    monkeypatch.delenv("DIALOG_DECODE_BACKEND", raising=False)
    monkeypatch.setattr(
        dialog_decode, "load_vision_client_and_model", _fake_vision_client_and_model
    )
    monkeypatch.setattr(
        dialog_decode,
        "decode_dialog_text",
        lambda client, model, screen: "ok",
    )

    decoder = load_dialog_decoder_or_none()

    assert decoder is not None
    assert decoder(_SCREEN) == "ok"


def test_load_or_none_returns_none_when_the_backend_is_unconfigured(
    monkeypatch, caplog
):
    # An unconfigured vision backend is a normal live-run state, not a crash:
    # the loop should still run, just without dialog text.
    monkeypatch.delenv("DIALOG_DECODE_BACKEND", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_VISION_MODEL", raising=False)

    with caplog.at_level(logging.WARNING, logger="jev_plays_pokemon.dialog_decode"):
        assert load_dialog_decoder_or_none() is None

    assert any("not configured" in r.getMessage() for r in caplog.records)


def test_load_or_none_returns_none_for_an_unrecognized_backend(monkeypatch):
    monkeypatch.setenv("DIALOG_DECODE_BACKEND", "tesseract")

    assert load_dialog_decoder_or_none() is None


def test_load_or_none_forwards_explicit_overrides_to_load_dialog_decoder(monkeypatch):
    captured: dict = {}

    def fake_load_dialog_decoder(**kwargs):
        captured.update(kwargs)
        return lambda screen: "ok"

    monkeypatch.setattr(dialog_decode, "load_dialog_decoder", fake_load_dialog_decoder)

    decoder = load_dialog_decoder_or_none(backend="rapidocr", model="explicit-model")

    assert decoder is not None
    assert decoder(_SCREEN) == "ok"
    assert captured == {"backend": "rapidocr", "model": "explicit-model"}


def test_load_or_none_does_not_swallow_a_non_config_error(monkeypatch):
    # Only the two "not configured" errors are downgraded to None; any other
    # exception (a real bug) must still propagate rather than be silenced.
    def explode():
        raise RuntimeError("unexpected")

    monkeypatch.setattr(dialog_decode, "load_dialog_decoder", explode)

    with pytest.raises(RuntimeError, match="unexpected"):
        load_dialog_decoder_or_none()
