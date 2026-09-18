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

import os
from collections.abc import Callable

from PIL import Image

from jev_plays_pokemon.dialog_ocr import decode_dialog_text_ocr, load_ocr_engine
from jev_plays_pokemon.dialog_vision import (
    decode_dialog_text,
    load_vision_client_and_model,
)

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
