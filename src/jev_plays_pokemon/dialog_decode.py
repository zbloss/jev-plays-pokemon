"""Env-driven backend selection between the RapidOCR and vision-LLM dialog
decoders.

Both `dialog_ocr.decode_dialog_text_ocr` and `dialog_vision.decode_dialog_text`
satisfy the same contract - a rendered dialog screen (PIL image) in,
transcribed text out - so which one actually runs is a deployment choice, not
a code change, matching this project's existing no-hardcoded-provider pattern
(see `dialog_vision.py`'s `OPENAI_*` vars).

`DIALOG_DECODE_BACKEND` selects the backend:

- unset, empty, or `"rapidocr"`: the local RapidOCR path, no LLM call
  (default - RapidOCR's accuracy against Pokemon Red's bitmap dialog font was
  validated per issue #31, so the vision-LLM path is no longer the default).
- `"vision-llm"`: the vision-LLM path.

Any other value raises `DialogDecodeConfigError` immediately, rather than
silently falling back to a default.
"""

import logging
import os
from collections.abc import Callable

from PIL import Image

from jev_plays_pokemon.dialog_ocr import decode_dialog_text_ocr, load_ocr_engine
from jev_plays_pokemon.dialog_vision import (
    VisionConfigError,
    decode_dialog_text,
    load_vision_client_and_model,
)

logger = logging.getLogger(__name__)

_BACKEND_ENV = "DIALOG_DECODE_BACKEND"
_VISION_LLM = "vision-llm"
_RAPIDOCR = "rapidocr"

DialogDecoder = Callable[[Image.Image], str]


class DialogDecodeConfigError(RuntimeError):
    """`DIALOG_DECODE_BACKEND` names a backend this project doesn't have."""


def load_dialog_decoder(
    *,
    backend: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> DialogDecoder:
    """Build the configured decode callable: rendered dialog screen in, text out.

    `backend` overrides `DIALOG_DECODE_BACKEND` when given (see
    `settings.Settings`, ADR 0001); `base_url`/`api_key`/`model` are forwarded
    to `load_vision_client_and_model` only when the resolved backend is the
    vision LLM, and only when actually given - an explicit `None` is never
    passed through, so a bare `load_dialog_decoder()` call keeps behaving
    exactly as if no CLI/`.env` layer existed.

    Each backend's own config errors (e.g. `dialog_vision.VisionConfigError`
    for a missing `OPENAI_VISION_MODEL`) still propagate from here rather than
    being swallowed.
    """
    resolved_backend = (
        (backend or os.environ.get(_BACKEND_ENV) or _RAPIDOCR).strip().lower()
    )

    if resolved_backend == _VISION_LLM:
        vision_overrides = {
            key: value
            for key, value in {
                "base_url": base_url,
                "api_key": api_key,
                "model": model,
            }.items()
            if value is not None
        }
        client, resolved_model = load_vision_client_and_model(**vision_overrides)
        return lambda screen: decode_dialog_text(client, resolved_model, screen)

    if resolved_backend == _RAPIDOCR:
        engine = load_ocr_engine()
        return lambda screen: decode_dialog_text_ocr(engine, screen)

    raise DialogDecodeConfigError(
        f"{_BACKEND_ENV}={resolved_backend!r} is not a recognized dialog-decode "
        f"backend; expected {_VISION_LLM!r} or {_RAPIDOCR!r}."
    )


def load_dialog_decoder_or_none(
    *,
    backend: str | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> DialogDecoder | None:
    """Build the configured decoder, or ``None`` if it isn't configured.

    The default vision backend raises `VisionConfigError` when
    `OPENAI_VISION_MODEL` (etc.) aren't set, and `DialogDecodeConfigError` for
    an unknown `DIALOG_DECODE_BACKEND` - both mean "no dialog decode for this
    run," a normal state for a loop that doesn't want dialog text yet. Turn
    those two config errors into `None` (with a warning) so the loop runs
    without dialog text instead of crashing on startup; any other exception
    (a real bug) still propagates rather than being silently downgraded. A
    live run gains dialog text by setting the backend's env vars (or the
    matching CLI flag/`.env` entry) and restarting.
    """
    overrides = {
        key: value
        for key, value in {
            "backend": backend,
            "base_url": base_url,
            "api_key": api_key,
            "model": model,
        }.items()
        if value is not None
    }
    try:
        return load_dialog_decoder(**overrides)
    except (VisionConfigError, DialogDecodeConfigError):
        logger.warning(
            "dialog-decode backend not configured; running without dialog text",
            exc_info=True,
        )
        return None
