"""Vision-fallback decode of dialog/NPC text via an OpenAI-compatible endpoint.

Gen 1's proprietary text encoding isn't decoded anywhere in the ecosystem
(see `docs/research/pokemon-red-ram-map.md`), so `game_state.py`'s RAM-based
extraction can report *that* a dialog box is open (`GameState.dialog_open`,
via the tile-238 check) but not decode its text content. This module reads
that same open dialog box off the rendered screen and asks a vision model to
transcribe it - the one field `game_state.py` can't reliably decode itself.

Per `CONTEXT.md`'s "Game state" entry, this is the sole vision-fallback path
in the project: it is triggered automatically whenever `GameState.dialog_open`
is true, never by a per-turn Jev decision, and it is never used for menu/item
names (those go through `lookup.item_name`'s static table instead).

TypeSafe's System One model (Jev) is text-only and has no vision capability,
so the decode call here is a plain OpenAI-compatible chat-completions
request, entirely independent of the `typesafe_sdk` client used for Jev's
tactical decisions elsewhere - this module never imports or calls it.

Provider configuration is env-driven and swappable (this is a public
package, so no provider or credentials are hardcoded):

- `OPENAI_BASE_URL`: base URL of an OpenAI-compatible endpoint (e.g. a
  self-hosted `sglang` server). Unset/empty targets the real OpenAI API.
- `OPENAI_API_KEY`: required when `OPENAI_BASE_URL` is unset/empty, since the
  real OpenAI API always needs one. Optional against a custom endpoint
  (self-hosted OpenAI-compatible servers often don't enforce auth), but
  still passed through to the client when provided.
- `OPENAI_VISION_MODEL`: model name/id to request. No provider-specific
  default is hardcoded.

Call `load_vision_client_and_model()` once and reuse its `(client, model)`
result for every `decode_dialog_text` call, rather than rebuilding the
client per turn.
"""

import base64
import io
import os
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from openai import OpenAI
from PIL import Image
from pyboy import PyBoy

_DIALOG_DECODE_INSTRUCTIONS = (
    "This is a screenshot from Pokemon Red/Blue with an open dialog or NPC "
    "text box. Transcribe the dialog text exactly as written. Reply with "
    "only the transcribed text, no commentary or surrounding quotes."
)


class VisionConfigError(RuntimeError):
    """The OpenAI-compatible vision endpoint isn't configured correctly."""


class _Message(Protocol):
    @property
    def content(self) -> str | None: ...


class _Choice(Protocol):
    @property
    def message(self) -> _Message: ...


class _ChatCompletion(Protocol):
    @property
    def choices(self) -> Sequence[_Choice]: ...


class _Completions(Protocol):
    # A read-only property of `Callable[..., Any]`, not a `create(**kwargs)`
    # method member: the openai SDK's `create` takes fixed keyword-only params
    # (plus stream-mode overloads returning `Stream[ChatCompletionChunk]`), so
    # no concrete `OpenAI` can satisfy an arbitrary-kwargs method member or a
    # writable attribute member. The response is narrowed to `_ChatCompletion`
    # at the use site in `decode_dialog_text` instead.
    @property
    def create(self) -> Callable[..., Any]: ...


class _Chat(Protocol):
    @property
    def completions(self) -> _Completions: ...


class VisionClient(Protocol):
    """Structural shape of `openai.OpenAI` that `decode_dialog_text` needs.

    Lets tests exercise this module's request/response handling against a
    fake/injected client (per this ticket's test-coverage requirement)
    without depending on the concrete `openai.OpenAI` class.
    """

    @property
    def chat(self) -> _Chat: ...


def load_vision_client_and_model(
    *,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[OpenAI, str]:
    """Build the OpenAI-compatible client and model id.

    Each parameter, left as `None`, falls back to its matching env var
    exactly as before; an explicit value always wins - the same
    resolve-or-env-fallback convention `typesafe_sdk` uses, letting a CLI
    flag/`.env` value (see `settings.Settings`, ADR 0001) override the
    environment without this function needing to know CLI/`.env` exist.
    """
    base_url = base_url or os.environ.get("OPENAI_BASE_URL") or None
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or None
    model = model or os.environ.get("OPENAI_VISION_MODEL") or None

    if model is None:
        raise VisionConfigError(
            "OPENAI_VISION_MODEL must be set to the vision model to request."
        )
    if base_url is None and api_key is None:
        raise VisionConfigError(
            "OPENAI_API_KEY must be set when OPENAI_BASE_URL is unset/empty "
            "(targeting the real OpenAI API, which always requires a key)."
        )

    # A self-hosted OpenAI-compatible server often doesn't enforce auth, but
    # the SDK itself still requires a non-None api_key to construct a client.
    client = OpenAI(base_url=base_url, api_key=api_key or "not-required")
    return client, model


def capture_screen(pyboy: PyBoy) -> Image.Image:
    """Read the current rendered screen as a PIL image.

    Built from `pyboy.screen.ndarray` rather than `pyboy.screen.image`: the
    latter returns `None` under PyBoy's `window="null"` backend (used by
    this project's tests and any headless run), while `ndarray` is populated
    regardless of window backend.
    """
    return Image.fromarray(pyboy.screen.ndarray, mode="RGBA").convert("RGB")


def decode_dialog_text(client: VisionClient, model: str, screen: Image.Image) -> str:
    """Decode the dialog/NPC text box rendered in `screen` via a vision-model call."""
    buffer = io.BytesIO()
    screen.save(buffer, format="PNG")
    data_url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )

    response: _ChatCompletion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": _DIALOG_DECODE_INSTRUCTIONS},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
    )
    return (response.choices[0].message.content or "").strip()
