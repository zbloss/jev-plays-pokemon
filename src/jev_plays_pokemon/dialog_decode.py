"""Env-driven backend selection between the vision-LLM and RapidOCR dialog
decoders.

Both `dialog_vision.decode_dialog_text` and `dialog_ocr.decode_dialog_text_ocr`
satisfy the same contract - a rendered dialog screen (PIL image) in,
transcribed text out - so which one actually runs is a deployment choice, not
a code change, matching this project's existing no-hardcoded-provider pattern
(see `dialog_vision.py`'s `OPENAI_*` vars).

`DIALOG_DECODE_BACKEND` selects the backend:

- unset, empty, or `"vision-llm"`: the existing vision-LLM path (default).
  Per issue #31, this stays the default until RapidOCR's accuracy against
  Pokemon Red's bitmap dialog font is demonstrated.
- `"rapidocr"`: the local RapidOCR path, no LLM call.

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


def load_dialog_decoder() -> DialogDecoder:
    """Build the configured decode callable: rendered dialog screen in, text out.

    Each backend's own config errors (e.g. `dialog_vision.VisionConfigError`
    for a missing `OPENAI_VISION_MODEL`) still propagate from here rather than
    being swallowed.
    """
    backend = (os.environ.get(_BACKEND_ENV) or _VISION_LLM).strip().lower()

    if backend == _VISION_LLM:
        client, model = load_vision_client_and_model()
        return lambda screen: decode_dialog_text(client, model, screen)

    if backend == _RAPIDOCR:
        engine = load_ocr_engine()
        return lambda screen: decode_dialog_text_ocr(engine, screen)

    raise DialogDecodeConfigError(
        f"{_BACKEND_ENV}={backend!r} is not a recognized dialog-decode "
        f"backend; expected {_VISION_LLM!r} or {_RAPIDOCR!r}."
    )


def load_dialog_decoder_or_none() -> DialogDecoder | None:
    """Build the configured decoder, or ``None`` if it isn't configured.

    The default vision backend raises `VisionConfigError` when
    `OPENAI_VISION_MODEL` (etc.) aren't set, and `DialogDecodeConfigError` for
    an unknown `DIALOG_DECODE_BACKEND` - both mean "no dialog decode for this
    run," a normal state for a loop that doesn't want dialog text yet. Turn
    those two config errors into `None` (with a warning) so the loop runs
    without dialog text instead of crashing on startup; any other exception
    (a real bug) still propagates rather than being silently downgraded. A
    live run gains dialog text by setting the backend's env vars and restarting.
    """
    try:
        return load_dialog_decoder()
    except (VisionConfigError, DialogDecodeConfigError):
        logger.warning(
            "dialog-decode backend not configured; running without dialog text",
            exc_info=True,
        )
        return None
