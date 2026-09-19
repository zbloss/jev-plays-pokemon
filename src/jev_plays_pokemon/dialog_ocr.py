"""RapidOCR-backed decode of dialog/NPC text - a local alternative to
`dialog_vision.py`'s vision-LLM path.

`decode_dialog_text_ocr` matches `dialog_vision.decode_dialog_text`'s input/
output contract - a rendered dialog screen (PIL image) in, transcribed text
out - so either backend can sit behind `dialog_decode.load_dialog_decoder`.

Unlike the vision-LLM path, RapidOCR runs entirely locally: no endpoint, no
API key, no per-call network request. `RapidOCR.__call__` accepts a
`PIL.Image.Image` directly and handles the RGB-to-BGR conversion itself, so
`decode_dialog_text_ocr` passes `screen` straight through rather than
converting it to a numpy array first.
"""

from typing import Any, Protocol

from PIL import Image
from rapidocr import RapidOCR


class _OCRResult(Protocol):
    @property
    def txts(self) -> tuple[str, ...] | None: ...


class OCREngine(Protocol):
    """Structural shape of `rapidocr.RapidOCR` that `decode_dialog_text_ocr` needs.

    Lets tests exercise this module's response handling against a fake/
    injected engine (per this ticket's test-coverage requirement) without
    loading the real RapidOCR model.

    `RapidOCR.__call__`'s own annotations are narrower than what it accepts:
    it declares `img_content: str | ndarray | bytes | Path` even though its
    `LoadImage` step accepts `PIL.Image.Image` too, and returns the union of
    every dispatch mode's output type even though the default det+cls+rec
    call always yields a `RapidOCROutput` (which is the only member with
    `txts`). So no protocol member signature can be both true to this
    module's usage (PIL image in, `txts` out) and satisfied by the concrete
    class; the call is typed `Any` and narrowed to `_OCRResult` at the use
    site instead.
    """

    def __call__(self, img_content: Any) -> Any: ...


def load_ocr_engine() -> RapidOCR:
    """Build the local RapidOCR engine.

    Unlike `dialog_vision.load_vision_client_and_model`, there's no env-driven
    configuration to validate here: RapidOCR needs no endpoint or credentials,
    just its bundled model files.
    """
    return RapidOCR()


def decode_dialog_text_ocr(engine: OCREngine, screen: Image.Image) -> str:
    """Decode the dialog/NPC text box rendered in `screen` via a local RapidOCR pass.

    A dialog box can render as multiple detected text lines; they're joined
    with newlines to keep the multi-line shape rather than collapsing it to a
    single run-on line.
    """
    result: _OCRResult = engine(screen)
    return "\n".join(result.txts or ()).strip()
